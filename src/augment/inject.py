"""
inject.py
=========
group-aware augmentation의 저장-포맷-무관 공통 로직 (MTM/CoFormer 드라이버가 공유).
원본: MTM/analysis/augment_common.py — 이 파일이 논문의 핵심 방법론(group-aware
missingness augmentation)이므로 로직은 원본과 byte-for-byte 동일하게 유지했다.
sys.path.insert 기반 import만 상대 import로 교체했다.

[W5 — overlay 정의 명시] "low에 high의 결측 패턴을 무작정 덮어씌우면(naive union) 결측
시간이 오히려 늘어나 버린다"(예: low 1-2시 + high 템플릿 1-10시 결측 -> union하면
11시간 결측)는 우려에 대한 답: 이 모듈은 naive union을 하지 않는다. 대신

  1. target-rate 기반: 목표는 "그 (cell,label) 그룹의 high 결측률(tf)"이고,
     새로 주입할 양은 need = tf - existing(그 샘플의 현재 결측률) 만큼으로 미리 정한다.
  2. intersection 존중(no-removal): 기존에 결측이던 슬롯은 항상 그대로 유지하고
     (절대 관측값으로 되돌리지 않음), 새로 추가하는 슬롯만 `~existing`(현재 결측이
     아닌 곳) 중에서 고른다 -> 기존 결측과 새 주입이 겹쳐도 "겹친 만큼"만 손해볼 뿐
     이중으로 늘어나지 않는다.
  3. budget-bound(no-overshoot): burst 분기는 매 배치마다 남은 need만큼만 길이를
     제한(`L = min(runlen_draw, remaining)`)하고, bin 분기는 남은 예산에 비례한
     확률(`prob=(tf-ecur)/(1-ecur)`)로만 추가한다 -> 최종 결측률이 tf를 초과하지 않는다.

원본 augment_missing_prototype.py는 block마다 모든 subject에 동일한 목표 결측률을
주입했다(high 전체 window pool에서 rng.choice). missing_pattern_report.py가 밝힌
"cell(성별x나이) x label(환자/대조)별로 결측 구조가 다르다"는 발견을 반영하기 위해,
여기서는 목표 pool을 (cell,label) 단위로 세분화한다. 표본이 작은(n<MIN_N) (cell,label)은
block(하위 feature group) 전역 pool로 fallback — 원본 스크립트와 동일 동작으로 축소된다.

"모양"(run-length/diurnal/cohesion)은 여전히 block 전역 파라미터(group_profile.json)를
쓴다 — 리포트 결정사항: 세분화는 "얼마나"(occurrence)에만 적용.

이 모듈은 dense (N,T,C) NaN-grid 표현에서만 동작한다. CoFormer(packed)는 호출 전
unpack, 호출 후 repack을 스스로 처리한다(build.py의 augment_coformer()).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from .grouping import BLOCKS, FEATURE_PERIODS_HOURS
from .profile import block_slot_missing, block_bin_missing, collect_runs, build_demo_lookup

MIN_N = 10
COHESIVE_THR = 0.6  # augment_missing_prototype.py와 동일


def build_subject_cell_label(sids, Y, cell_axes=("Sex", "AgeGroup"), use_label=True):
    """윈도우 단위 subject_id 배열 + label(0/1) 배열 -> subject 단위 DataFrame
    (index=subject_id, cols=Sex/AgeGroup/cell/label). subject 대표 라벨 = phq9 평균>=0.5.

    W6 ablation용 파라미터 (lmm_effect_size.py의 effect size 순서 age > sex > patient를
    따라 순차로 켠다):
      cell_axes: 셀을 구성할 인구통계 축. ("Sex","AgeGroup")=기본(전체), ("AgeGroup",)=나이만,
                 ()=축 없음(단일 "ALL" 셀, uniform 주입과 동일해짐).
      use_label: False면 환자/대조 구분을 끄고 label=0 고정(모든 subject가 같은 target pool
                 공유) -> "환자/대조 축 끄기" ablation.
    subject 표본 집합은 항상 Sex/AgeGroup 둘 다 있는 subject로 고정(ablation 간 N 비교가
    공정하도록, cell_axes가 그 중 일부만 써도 표본 자체는 안 바뀜)."""
    df = pd.DataFrame({"subject_id": sids, "y": Y})
    subj_label = (df.groupby("subject_id")["y"].mean() >= 0.5).astype(int).rename("label")
    demo = build_demo_lookup()
    out = demo.join(subj_label, how="inner")
    out = out.dropna(subset=["Sex", "AgeGroup"])
    if cell_axes:
        out["cell"] = out[list(cell_axes)].astype(str).agg("_".join, axis=1)
    else:
        out["cell"] = "ALL"
    if not use_label:
        out["label"] = 0
    return out


def assign_window_cell_label(sids, subj_lookup):
    """윈도우 단위 subject_id 배열 -> (cell array[object], label array[float, NaN=unknown])."""
    lut_cell = subj_lookup["cell"].to_dict()
    lut_label = subj_lookup["label"].to_dict()
    cell_arr = np.array([lut_cell.get(s) for s in sids], dtype=object)
    label_arr = np.array([lut_label.get(s, np.nan) for s in sids], dtype=float)
    return cell_arr, label_arr


def _build_pools(occ_per_window, cell_arr_high, label_arr_high, subj_lookup_high, min_n):
    """high의 window별 occurrence(1D) -> {(cell,label): 그 subgroup의 occurrence pool}.
    subj_lookup_high(subject 단위, build_subject_cell_label 출력)에서 (cell,label)별
    **subject 수**(window 수 아님)를 세어 min_n 미만이면 dict에서 생략 -> 호출자가 global
    pool로 fallback. cell_axes/use_label ablation에도 그대로 동작(CSV 사전계산 불필요)."""
    pools = {}
    counts = subj_lookup_high.groupby(["cell", "label"]).size()
    for (cell, label), n in counts.items():
        if n < min_n:
            continue
        m = (cell_arr_high == cell) & (label_arr_high == label)
        if m.sum() > 0:
            pools[(cell, label)] = occ_per_window[m]
    return pools


def _rank_match_targets(ecur_arr, cell_arr, label_arr, target_pools, global_pool, rng):
    """(cell,label) 그룹별로 low 샘플들을 현재 결측률(ecur) 순위로 정렬해 high pool의
    같은 분위수 값을 목표(tf)로 배정. 독립 랜덤추출(rng.choice)의 "이미 그 그룹에서
    결측이 높은 저(low)샘플이 우연히 낮은 목표를 뽑아 못 낮추는" 비대칭을 완화한다:
    낮은 순위(적게 결측)는 낮은 목표를, 높은 순위(많이 결측)는 높은 목표를 받는다."""
    Nl = len(ecur_arr)
    tf_arr = np.empty(Nl, dtype=np.float64)
    groups = defaultdict(list)
    unknown = []
    for i in range(Nl):
        c, l = cell_arr[i], label_arr[i]
        if c is None or (isinstance(l, float) and np.isnan(l)):
            unknown.append(i)
        else:
            groups[(c, l)].append(i)
    for key, idx in groups.items():
        idx = np.array(idx)
        pool = target_pools.get(key, global_pool)
        sub_ecur = ecur_arr[idx]
        order = np.argsort(sub_ecur)
        n = len(idx)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        pct = (ranks + 0.5) / n * 100
        sorted_pool = np.sort(pool)
        tf_arr[idx] = np.percentile(sorted_pool, pct)
    if unknown:
        tf_arr[np.array(unknown)] = rng.choice(global_pool, size=len(unknown))
    return tf_arr


def inject_group_aware(Xl_aug, maskl, hoursl, cell_arr_low, label_arr_low,
                       maskh, hoursh, cell_arr_high, label_arr_high,
                       feats, prof, subj_lookup_high, rng, min_n=MIN_N, mode="auto"):
    """low(filled) dense 배열에 (cell,label)별 차등 결측 주입. in-place로 Xl_aug 수정.
    augment_missing_prototype.inject()의 리팩터판: target pool만 (cell,label) 서브그룹으로
    세분화하고, 나머지(run-length/diurnal/cohesion 기반 burst 또는 bin Bernoulli) 로직은 동일.

    subj_lookup_high: build_subject_cell_label()의 반환값(high 쪽) — cell_axes/use_label로
    셀 정의를 바꾸면(W6 ablation) 이 함수 수정 없이 그대로 다른 그룹화로 동작한다.

    mode: "auto"(기본, 기존 동작과 byte-identical) — 블록별 p==1 and cohesive일 때만 burst,
    나머지는 bin Bernoulli. "bernoulli_only" — burst 조건을 무시하고 항상 bin Bernoulli
    (rate-only 베이스라인용: 셀 구분/목표 결측률은 유지하되 burst 모양 재현만 끈다).
    target pool(global_pool/target_pools, high에서 유도)은 두 분기가 같은 (bfeats, p)로
    계산하므로 mode를 바꿔도 셀별 목표 결측률의 출처는 동일 — build.py의 검증 빌드가
    블록별 최종 결측률을 실측 비교한다."""
    if mode not in ("auto", "bernoulli_only"):
        raise ValueError(f"unknown mode: {mode!r}")
    Nl, T, _ = Xl_aug.shape
    burst_blocks = []  # D: burst가 실제 적용된 블록 목록(로그+반환용, "패턴 재현" 적용 범위)

    for bn, bfeats in BLOCKS.items():
        p = FEATURE_PERIODS_HOURS[bfeats[0]]
        cols = [feats.index(f) for f in bfeats if f in feats]
        coh_field = prof["groups"][bn]["within_cohesion_high"]
        coh = coh_field.get("mean") if isinstance(coh_field, dict) else coh_field
        cohesive = (coh is None) or (coh >= COHESIVE_THR)

        sl_h = block_slot_missing(maskh, feats, bfeats)
        bin_h = block_bin_missing(sl_h, p, T)
        global_pool = bin_h.mean(axis=1)
        target_pools = _build_pools(global_pool, cell_arr_high, label_arr_high, subj_lookup_high, min_n)
        spb = max(1, round(p / 1))
        nb = bin_h.shape[1]

        use_burst = (mode == "auto") and (p == 1 and cohesive)
        if use_burst:
            burst_blocks.append(bn)
        print(f"  [inject] block={bn} p={p}h cohesive={cohesive} mode={mode} -> "
              f"{'burst' if use_burst else 'bernoulli'}")
        if use_burst:  # burst 주입 (셀별 target, 모양은 block 전역)
            runlen_pool = collect_runs(bin_h)
            if len(runlen_pool) == 0:
                runlen_pool = np.array([1])
            diur = np.array(prof["groups"][bn]["diurnal_high"])
            # burst(시간단위 hr/rr/temp/ppg) 블록은 이미 양(occurrence) 매칭이 좋았고
            # (bugfix만으로 근접), rank 매칭을 적용하면 필요 주입량이 줄어 diurnal(시간대)
            # 신호가 옅어지는 부작용이 있었다 -> 이 분기는 기존 랜덤 추출 유지(하이브리드 확정판).
            # (트로프-보호용 zone-capping을 실험했으나 A_hr_rr는 개선되고 B1_temp/B2_ppg는
            # 12-13시 경계에서 인위적 단절/오버슈트가 생겨 기각 -> 원래 방식으로 복귀)
            sl_l = block_slot_missing(maskl, feats, bfeats)
            for i in range(Nl):
                existing = sl_l[i].copy()
                pool = target_pools.get((cell_arr_low[i], label_arr_low[i]), global_pool)
                tf = rng.choice(pool)
                need = int(round((tf - existing.mean()) * T))
                if need <= 0:
                    continue
                w = diur[hoursl[i]]
                w = w / w.sum() if w.sum() > 0 else None
                newmiss = existing.copy()
                attempts = 0
                # overshoot 방지: 남은 need만큼만 burst 길이 제한 -> 목표 초과 주입 안 함
                while (newmiss.sum() - existing.sum()) < need and attempts < 60:
                    remaining = need - (newmiss.sum() - existing.sum())
                    L = min(int(rng.choice(runlen_pool)), max(1, remaining))
                    start = int(rng.choice(T, p=w))
                    newmiss[start:start + L] = True
                    attempts += 1
                add = np.where(newmiss & ~existing)[0]
                if len(add):
                    Xl_aug[i][np.ix_(add, cols)] = np.nan
        else:  # coarse / 비응집 / mode=bernoulli_only 강제 -> bin Bernoulli (셀별 target)
            feat_iter = [cols] if cohesive else [[c] for c in cols]
            for fcols in feat_iter:
                sub_feats = [feats[c] for c in fcols]
                sl_l = block_slot_missing(maskl, feats, sub_feats)
                bin_l = block_bin_missing(sl_l, p, T)
                sl_h2 = block_slot_missing(maskh, feats, sub_feats)
                bin_h2 = block_bin_missing(sl_h2, p, T)
                sub_global = bin_h2.mean(axis=1)
                sub_pools = _build_pools(sub_global, cell_arr_high, label_arr_high, subj_lookup_high, min_n)
                ecur_all = bin_l.mean(axis=1)
                tf_all = _rank_match_targets(ecur_all, cell_arr_low, label_arr_low,
                                             sub_pools, sub_global, rng)
                for i in range(Nl):
                    existing = bin_l[i]
                    ecur = existing.mean()
                    tf = tf_all[i]
                    if tf <= ecur:
                        continue
                    avail = np.where(~existing)[0]
                    if len(avail) == 0:
                        continue
                    # 확률적 주입: 각 available bin을 prob로 결측 -> E[occurrence]=tf (round overshoot 제거)
                    prob = (tf - ecur) / (1.0 - ecur) if ecur < 1.0 else 0.0
                    pick = avail[rng.random(len(avail)) < prob]
                    for b in pick:
                        s, e = b * spb, min((b + 1) * spb, T)
                        Xl_aug[i][np.ix_(np.arange(s, e), fcols)] = np.nan
    print(f"  [inject] burst 적용 블록(패턴 재현): {burst_blocks if burst_blocks else '(없음)'}")
    return Xl_aug
