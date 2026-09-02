"""
profile.py
==========
low_missing vs high_missing 데이터에서 그룹(A~F)/디바이스별 결측 패턴을 프로파일링.
→ 실험2 증강의 "그룹별 증강 기준"을 만들기 위한 사전 분석. `build_group_profile()`의
반환 dict가 `inject.inject_group_aware()`의 `prof` 인자로 그대로 소비된다.

원본: MTM/analysis/group_missing_profile.py + MTM/analysis/imbalance_by_subgroup.py
(build_demo_lookup/encode_sex/AGE_BINS/AGE_LABELS만 — inject.py가 build_demo_lookup을
필요로 하는데 imbalance_by_subgroup.py 전체는 이 마이그레이션 대상이 아니라서 여기로
옮겨왔다).

이 파일은 matplotlib/seaborn을 전혀 쓰지 않는다(순수 numpy/pandas/scipy). 원본의 그림
생성 코드(_box_stats_low_high의 boxplot, _dep_heatmap, 모든 plt./sns. 호출)는 전부
드롭했다 — report figure이지 `inject_group_aware()`가 실제로 소비하는 산출물이
아니기 때문. 통계치 자체(Mann-Whitney U, Cliff's delta, KS-test, bootstrap CI 등)는
그대로 유지해 profile dict에 채워 넣는다. 향후 src/analysis/의 별도 figure 스크립트가
`build_group_profile()`이 반환하는 dict에서 이 그림들(group_occurrence.png,
runlength_highfreq.png, diurnal_highfreq.png, cross_dependency_high.png,
device_occurrence.png)을 재생성할 수 있다.

`missing_pattern_groups.py`(Layer1/2/3 통계검정+그림, control/patient 결측 패턴 비교)는
이 파일로 옮기지 않았다 — `inject_group_aware`/`build.py`가 런타임에 전혀 참조하지 않는
validation-only 스크립트이고, `compute_sample_missrate`를 자체 재정의하는 등 이 패키지의
canonical 정의(`grouping.py`)와 중복이라 별도 `src/analysis/` 마이그레이션 대상으로 남겨둔다.
"""

import json
import pickle
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp

from ..paths import DATA_ROOT
from .grouping import FEATURE_PERIODS_HOURS, BLOCKS, compute_sample_missrate

N_BOOT = 2000
BOOT_SEED = 42

AGE_BINS = [0, 30, 40, 50, 60, 150]
AGE_LABELS = ["<30s", "30s", "40s", "50s", "60s+"]

# 디바이스 그룹 (첨부 표 기준). E는 기기 혼합(Smartband/Smartphone) → 자체 버킷.
DEVICE_GROUPS = {
    "Smartband": ["hr", "rr", "spo2", "core_temp", "skin_temp",
                  "bp_sys", "bp_dia", "glucose", "hrv"],
    "Smartphone": ["light_sensor", "proximity"],
    "Daily_agg(E)": ["step", "distance", "screen_time", "wake_time", "sleep_time",
                     "deep_sleep_time", "rem_sleep_time", "light_sleep_time", "total_sleep_time"],
    "MobileApp(F)": ["EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress"],
}


# =====================================================
# 인구통계 lookup (원본: imbalance_by_subgroup.py)
# =====================================================

def encode_sex(x):
    """step3_mtm_v2.py의 수정판과 동일 로직 (회차별 인코딩 불일치 대응)."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x in ("남", "1"):
        return "Male"
    if x in ("여", "0", "2"):
        return "Female"
    return np.nan


def decode_age(x):
    """step3_mtm.py와 동일하게 정제 나이 또는 생년월일을 만 나이로 변환한다."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    x = str(x).strip()
    dt = pd.to_datetime(x, errors="coerce")
    if pd.notna(dt):
        today = pd.Timestamp.today()
        return float(today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day)))
    try:
        return float(x)
    except Exception:
        return np.nan


def build_demo_lookup():
    """subject_id -> (Sex, Age, AgeGroup) lookup. age_years 우선, 없으면 Age를 변환."""
    survey_all = pickle.load(open(DATA_ROOT / "cache" / "survey_all.pkl", "rb"))
    first = survey_all.drop_duplicates("ID", keep="first").set_index("ID")
    age = first["age_years"] if "age_years" in first.columns else first["Age"].apply(decode_age)
    demo = pd.DataFrame({
        "Sex": first["Sex"].apply(encode_sex),
        "Age": age,
    })
    demo["AgeGroup"] = pd.cut(demo["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return demo


# =====================================================
# 통계 유틸
# =====================================================

def cliffs_delta_from_u(U, n1, n2):
    """rank-biserial = 2U/(n1 n2) - 1 (Cliff's delta). 부호: 양수면 group1(high)이 더 큼."""
    return 2.0 * U / (n1 * n2) - 1.0


def mwu_test(low_vals, high_vals):
    """subject-level Mann-Whitney U (high vs low) + Cliff's delta. NaN 제거."""
    a = np.asarray(low_vals); a = a[~np.isnan(a)]
    b = np.asarray(high_vals); b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return {"n_low": len(a), "n_high": len(b), "U": np.nan, "p_value": np.nan,
                "cliffs_delta": np.nan, "significant": False}
    U, p = mannwhitneyu(b, a, alternative="two-sided")
    return {"n_low": len(a), "n_high": len(b), "U": float(U), "p_value": float(p),
            "cliffs_delta": float(cliffs_delta_from_u(U, len(b), len(a))),
            "significant": bool(p < 0.05)}


def _group_stats(low_series, high_series):
    """{group_name: pd.Series(subject-level, 0~1)} 두 dict -> subject-level MWU + Cliff's
    delta DataFrame (통계만, 플로팅 없음). 원본 _box_stats_low_high()의 boxplot을 뺀 판."""
    rows = []
    for n in low_series.keys():
        stat = mwu_test(low_series[n].values, high_series[n].values)
        stat["group"] = n
        stat["mean_low"] = float(np.nanmean(low_series[n].values))
        stat["mean_high"] = float(np.nanmean(high_series[n].values))
        rows.append(stat)
    return pd.DataFrame(rows)[["group", "n_low", "n_high", "mean_low", "mean_high",
                               "U", "p_value", "cliffs_delta", "significant"]]


# =====================================================
# 로드 / 저수준 결측 지표
# =====================================================

def load_group(name, target, dataset_base=None):
    dataset_base = (Path(dataset_base) if dataset_base is not None
                    else DATA_ROOT / "mtm" / target / "datasets")
    d = np.load(dataset_base / name / "processed_data" / f"1_{target}.npz", allow_pickle=True)
    feats = json.load(open(dataset_base / name / "feature_columns.json"))
    sids = np.array(json.load(open(dataset_base / name / "subject_ids.json")))
    smids = json.load(open(dataset_base / name / "sample_ids.json"))
    X = np.concatenate([d["train_x"], d["val_x"], d["test_x"]], axis=0)
    return X, feats, sids, smids


def parse_wstart(sample_ids):
    return np.array([int(s.rsplit("_s", 1)[1]) for s in sample_ids], dtype=np.int64)


def block_slot_missing(mask, feats, block_feats):
    """block의 slot별 결측 지표: 그 그룹 feature가 (그 slot에서) 전부 NaN이면 True. (N,T)"""
    cols = [feats.index(f) for f in block_feats if f in feats]
    return mask[:, :, cols].all(axis=2)


def block_bin_missing(slot_missing, period_h, T=72, unit_h=1):
    """slot 결측지표(N,T)를 그룹 native bin(period_h) 단위로 접음: bin 전체 결측이면 True. (N, nbins)"""
    spb = max(1, round(period_h / unit_h))
    nb = ceil(T / spb)
    out = np.zeros((slot_missing.shape[0], nb), dtype=bool)
    for b in range(nb):
        s, e = b * spb, min((b + 1) * spb, T)
        out[:, b] = slot_missing[:, s:e].all(axis=1)
    return out


def collect_runs(bin_missing):
    """(N,nbins) bool → 연속 True run 길이 리스트."""
    runs = []
    for row in bin_missing:
        if not row.any():
            continue
        padded = np.concatenate(([0], row.astype(int), [0]))
        d = np.diff(padded)
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        runs.extend((ends - starts).tolist())
    return np.array(runs)


def subject_occurrence(bin_missing, sids):
    """윈도우별 bin 결측률 → subject 평균."""
    win_rate = bin_missing.mean(axis=1)
    return pd.Series(win_rate).groupby(sids).mean()


def _hour_rate(slot_missing, hours):
    h = hours.ravel()
    m = slot_missing.ravel().astype(np.float64)
    cnt = np.bincount(h, minlength=24)
    tot = np.bincount(h, weights=m, minlength=24)
    return tot / np.maximum(cnt, 1)


# =====================================================
# profile 계산 (원본 main()의 플로팅-제외 리팩터판)
# =====================================================

def build_group_profile(target, low_name="low_missing", high_name="high_missing", dataset_base=None):
    """low_missing vs high_missing에서 그룹(A~F)/디바이스별 결측 프로파일을 계산해
    dict로 반환한다 (원본이 group_profile.json에 쓰던 것과 동일한 구조).
    occurrence(subject 단위) / run-length(고빈도 1h 그룹) / diurnal(고빈도 그룹) /
    within-cohesion / cross-group dependency 를 포함한다. 그림은 그리지 않는다."""
    Xl, feats, sidl, smidl = load_group(low_name, target, dataset_base)
    Xh, _, sidh, smidh = load_group(high_name, target, dataset_base)
    T = Xl.shape[1]
    maskl, maskh = np.isnan(Xl), np.isnan(Xh)
    hoursl = (parse_wstart(smidl)[:, None] + np.arange(T)[None, :]) % 24
    hoursh = (parse_wstart(smidh)[:, None] + np.arange(T)[None, :]) % 24

    # per-feature 주기보정 결측률 (heatmap과 동일 정의) — occurrence_perfeat용
    Ml = compute_sample_missrate(maskl, feats, 60)
    Mh = compute_sample_missrate(maskh, feats, 60)

    def perfeat_occ_series(M, sids, block_feats):
        """subject-level Series (not reduced to a single mean) — 통계검정용."""
        cols = [feats.index(f) for f in block_feats if f in feats]
        return pd.Series(M[:, cols].mean(axis=1)).groupby(sids).mean()

    def perfeat_occ(M, sids, block_feats):
        return perfeat_occ_series(M, sids, block_feats).mean()

    block_names = list(BLOCKS.keys())
    profile = {"_meta": {"T": T, "unit_hours": 1, "target": target,
                         "note": "occurrence/runlength are period-bin based; run-length in bins"},
               "groups": {}, "devices": {}}

    # 그룹별 slot/bin 결측 지표 미리 계산
    slot_low, slot_high, bin_low, bin_high, period_of = {}, {}, {}, {}, {}
    for bn in block_names:
        p = FEATURE_PERIODS_HOURS[[f for f in BLOCKS[bn]][0]]
        period_of[bn] = p
        sl_l = block_slot_missing(maskl, feats, BLOCKS[bn])
        sl_h = block_slot_missing(maskh, feats, BLOCKS[bn])
        slot_low[bn], slot_high[bn] = sl_l, sl_h
        bin_low[bn] = block_bin_missing(sl_l, p, T)
        bin_high[bn] = block_bin_missing(sl_h, p, T)

    # ── 1. occurrence (subject 단위, Mann-Whitney U + Cliff's delta) ──
    occ_low_series = {bn: perfeat_occ_series(Ml, sidl, BLOCKS[bn]) for bn in block_names}
    occ_high_series = {bn: perfeat_occ_series(Mh, sidh, BLOCKS[bn]) for bn in block_names}
    group_occ_stats = _group_stats(occ_low_series, occ_high_series)
    print("\n[group occurrence] Mann-Whitney U + Cliff's delta (subject-level)")
    print(group_occ_stats.to_string(index=False))

    # ── 2. run-length (고빈도 1h 그룹): KS-test + 분위수 ────
    hf_blocks = [bn for bn in block_names if period_of[bn] == 1]
    QUANTILES = [10, 25, 50, 75, 90, 99]
    runlen_quantiles = {}
    runlen_ks = {}
    for bn in hf_blocks:
        runs_l = collect_runs(bin_low[bn])
        runs_h = collect_runs(bin_high[bn])
        ks_stat, ks_p = ks_2samp(runs_l, runs_h)
        runlen_ks[bn] = {"ks_stat": float(ks_stat), "p_value": float(ks_p)}
        runlen_quantiles[bn] = {
            "low": {f"p{q}": float(np.percentile(runs_l, q)) for q in QUANTILES},
            "high": {f"p{q}": float(np.percentile(runs_h, q)) for q in QUANTILES},
        }
    print("\n[run-length] KS-test (low vs high distribution) + quantiles(hours)")
    for bn in hf_blocks:
        print(f"  {bn:10s} KS D={runlen_ks[bn]['ks_stat']:.3f} p={runlen_ks[bn]['p_value']:.2e}  "
              f"low.p50/p90={runlen_quantiles[bn]['low']['p50']:.0f}/{runlen_quantiles[bn]['low']['p90']:.0f}h  "
              f"high.p50/p90={runlen_quantiles[bn]['high']['p50']:.0f}/{runlen_quantiles[bn]['high']['p90']:.0f}h")

    # ── 3. diurnal (고빈도 그룹, high만 profile에 필요) ────────
    diurnal_high = {}
    for bn in hf_blocks:
        dh = _hour_rate(slot_high[bn], hoursh)
        diurnal_high[bn] = dh.tolist()

    # ── 4. within-group cohesion (subject-level cluster bootstrap CI) ───
    cohesion = {}
    cohesion_ci = {}
    uniq_sidh = np.unique(sidh)
    sid_to_rows = {s: np.where(sidh == s)[0] for s in uniq_sidh}
    for bn in block_names:
        cols = [feats.index(f) for f in BLOCKS[bn] if f in feats]
        if len(cols) < 2:
            cohesion[bn] = None
            cohesion_ci[bn] = None
            continue
        fn = maskh[:, :, cols]                      # (N,T,C)
        any_miss = fn.any(axis=2).sum(axis=1)        # (N,)
        all_miss = fn.all(axis=2).sum(axis=1)        # (N,)
        subj_any = pd.Series(any_miss).groupby(sidh).sum()
        subj_all = pd.Series(all_miss).groupby(sidh).sum()
        cohesion[bn] = float(subj_all.sum() / max(subj_any.sum(), 1))

        rng = np.random.default_rng(BOOT_SEED)
        subs = subj_any.index.values
        boots = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(subs), size=len(subs))
            a = subj_any.values[pick].sum(); b = subj_all.values[pick].sum()
            boots.append(b / max(a, 1))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cohesion_ci[bn] = {"mean": cohesion[bn], "ci_lo": float(lo), "ci_hi": float(hi)}
    print("\n[within-group cohesion] P(all missing | any missing), high_missing, subject cluster-bootstrap 95% CI")
    for bn in block_names:
        if cohesion_ci[bn] is None:
            continue
        c = cohesion_ci[bn]
        print(f"  {bn:10s} {c['mean']:.3f}  [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]")

    # ── 5. cross-group dependency: slot-level P(Y missing|X missing), subject cluster-bootstrap CI
    S = {bn: slot_high[bn] for bn in block_names}   # (N,T) per block
    dep = np.zeros((len(block_names), len(block_names)))
    dep_ci_halfwidth = np.zeros((len(block_names), len(block_names)))
    rng = np.random.default_rng(BOOT_SEED)
    for i, bx in enumerate(block_names):
        for j, by in enumerate(block_names):
            x_all, xy_all = S[bx].reshape(-1), (S[bx] & S[by]).reshape(-1)
            dep[i, j] = xy_all.mean() / max(x_all.mean(), 1e-9)
    x_subj_sum = {bn: pd.Series(S[bn].sum(axis=1)).groupby(sidh).sum() for bn in block_names}
    for i, bx in enumerate(block_names):
        for j, by in enumerate(block_names):
            xy_sum_per_win = (S[bx] & S[by]).sum(axis=1)
            xy_subj_sum = pd.Series(xy_sum_per_win).groupby(sidh).sum()
            xa = x_subj_sum[bx].values
            xya = xy_subj_sum.reindex(x_subj_sum[bx].index, fill_value=0).values
            boots = []
            for _ in range(500):  # 64셀 x 500 = 계산량 감안해 축소
                pick = rng.integers(0, len(xa), size=len(xa))
                boots.append(xya[pick].sum() / max(xa[pick].sum(), 1e-9))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            dep_ci_halfwidth[i, j] = (hi - lo) / 2
    print(f"\n[cross-dependency] subject cluster-bootstrap 95% CI 반폭 평균 = "
          f"{dep_ci_halfwidth[~np.eye(len(block_names), dtype=bool)].mean()*100:.1f}%p "
          f"(대각선 제외, 값은 반환 dict의 cross_dependency_high.bootstrap_ci_halfwidth 참고)")

    # ── profile dict 채우기 ─────────────────────────────────────
    occ_stats_by_group = group_occ_stats.set_index("group").to_dict("index")
    for bn in block_names:
        runs_h = collect_runs(bin_high[bn])
        occ_l = subject_occurrence(bin_low[bn], sidl).mean()
        occ_h = subject_occurrence(bin_high[bn], sidh).mean()
        profile["groups"][bn] = {
            "period_hours": int(period_of[bn]),
            "features": BLOCKS[bn],
            "occurrence_perfeat_low": float(perfeat_occ(Ml, sidl, BLOCKS[bn])),
            "occurrence_perfeat_high": float(perfeat_occ(Mh, sidh, BLOCKS[bn])),
            "occurrence_mwu": occ_stats_by_group.get(bn),
            "occurrence_joint_low": float(occ_l),
            "occurrence_joint_high": float(occ_h),
            "runlength_high_bins": {
                "mean": float(runs_h.mean()) if len(runs_h) else 0.0,
                "median": float(np.median(runs_h)) if len(runs_h) else 0.0,
                "p90": float(np.percentile(runs_h, 90)) if len(runs_h) else 0.0,
                "quantiles": runlen_quantiles.get(bn),
                "ks_test_low_vs_high": runlen_ks.get(bn),
            },
            "diurnal_high": diurnal_high.get(bn),
            "within_cohesion_high": cohesion_ci.get(bn),
        }
    profile["cross_dependency_high"] = {
        "blocks": block_names, "matrix_P(col|row)": dep.round(3).tolist(),
        "bootstrap_ci_halfwidth": dep_ci_halfwidth.round(3).tolist(),
        "note": "cluster bootstrap(subject 재표본, n_boot=500) 95% CI 반폭"}

    # ── device 축 occurrence (Mann-Whitney U + Cliff's delta) ─
    dev_low_series = {dv: perfeat_occ_series(Ml, sidl, df) for dv, df in DEVICE_GROUPS.items()}
    dev_high_series = {dv: perfeat_occ_series(Mh, sidh, df) for dv, df in DEVICE_GROUPS.items()}
    dev_occ_stats = _group_stats(dev_low_series, dev_high_series)
    print("\n[device occurrence] Mann-Whitney U + Cliff's delta (subject-level)")
    print(dev_occ_stats.to_string(index=False))
    dev_stats_by_group = dev_occ_stats.set_index("group").to_dict("index")
    for dv, dfeats in DEVICE_GROUPS.items():
        p = FEATURE_PERIODS_HOURS[dfeats[0]]
        profile["devices"][dv] = {
            "features": dfeats, "period_hours": int(p),
            "occurrence_perfeat_low": float(dev_low_series[dv].mean()),
            "occurrence_perfeat_high": float(dev_high_series[dv].mean()),
            "occurrence_mwu": dev_stats_by_group.get(dv),
        }

    return profile


if __name__ == "__main__":
    import argparse
    from ..paths import DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["phq9", "gad7"], required=True)
    args = ap.parse_args()
    profile = build_group_profile(target=args.target)
    out_dir = DATA_ROOT / "mtm" / args.target / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "group_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print(f"\n  saved: {out_dir / 'group_profile.json'}")
    print("Done.")
