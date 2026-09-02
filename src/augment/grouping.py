"""
grouping.py
===========
baseline 데이터셋(step3 산출물)을 결측률 기준으로 low_missing/high_missing 두 그룹으로
나눈다. 이 모듈은 src/augment 패키지 전체의 canonical single-source-of-truth다:
FEATURE_PERIODS_HOURS / BLOCKS / apply_period_fill / compute_sample_missrate는 원본
저장소에서 여러 분석 스크립트(group_missing_profile.py, imbalance_by_subgroup.py,
missing_pattern_groups.py, augment_distribution_test.py)가 각자 재정의하던 것을 여기
하나로 모은 것 — 패키지 내 다른 모듈은 전부 여기서 import한다(재정의 금지).

파이프라인 (사용자 확정 순서):
  1. ±14일 슬라이스는 이미 baseline 생성 단계에서 끝남 (72h 윈도우 샘플)
  2. 측정주기별 값 채우기 (apply_period_fill)
        - 수집주기 bin 안에 값이 하나라도 있으면 그 bin 전체를 관측값으로 채움
        - 정규화 후 채우기 == 정규화 전 채우기 (선형 변환이라 수학적으로 동일)
  3. subject-level 결측률 측정 (주기 bin 기준, fill 여부와 무관하게 동일)
  4. 결측률 순위 기준 median split (rank-based) → low / high 두 그룹
  5. 학습 표본 수를 두 그룹이 같도록 맞춤 (많은 쪽을 랜덤 서브샘플링)
  6. low_missing / high_missing 데이터셋 저장

원본: MTM/grouping_by_missing.py. figure 함수(plot_histogram / plot_group_mean_missing)는
matplotlib 의존이라 이 패키지에서 제외했다 — src/augment는 numpy/pandas만 사용해야
한다(HARD CONSTRAINT: torch 금지 패키지이자, 이 파일은 그 상위 규칙으로 matplotlib도
빼서 순수 수치 로직만 남겼다).

사용법:
  from src.augment.grouping import split_low_high
  split_low_high(DATA_ROOT / "mtm" / "mtm_v3_unit60_w72_s24")
"""

import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DATA_ROOT

# 각 피처가 한 번 관측될 것으로 기대되는 주기(시간). 이 bin이 통째로 NaN이면 결측 1.
FEATURE_PERIODS_HOURS = {
    "hr": 1, "rr": 1, "core_temp": 1, "skin_temp": 1,
    "bp_sys": 1, "bp_dia": 1, "glucose": 1,
    "spo2": 24, "hrv": 24, "light_sensor": 24, "proximity": 24,
    "step": 24, "distance": 24, "screen_time": 24,
    "wake_time": 24, "sleep_time": 24, "deep_sleep_time": 24,
    "rem_sleep_time": 24, "light_sleep_time": 24, "total_sleep_time": 24,
    "EMA_Anxiety": 168, "EMA_Depression": 168, "EMA_Sleep": 168, "EMA_Stress": 168,
}

# 측정주기 그룹 (CLAUDE.md A~F, figure용 색/블록)
BLOCKS = {
    "A_hr_rr": ["hr", "rr"],
    "B1_temp": ["core_temp", "skin_temp"],
    "B2_ppg": ["bp_sys", "bp_dia", "glucose"],
    "A_spo2": ["spo2"],
    "C_hrv": ["hrv"],
    "D_event": ["light_sensor", "proximity"],
    "E_daily": ["step", "distance", "screen_time", "wake_time", "sleep_time",
                "deep_sleep_time", "rem_sleep_time", "light_sleep_time", "total_sleep_time"],
    "F_ema": ["EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress"],
}

SEED = 42


# =====================================================
# 측정주기별 값 채우기
# =====================================================

def apply_period_fill(X, feats, unit_minutes=60):
    """X:(N,T,C) — 수집주기 bin 안에 값이 하나라도 있으면 bin 전체를 그 관측값(nanmean)으로 채움.
    정규화가 이미 적용된 배열에 대해서도 결과가 동일함(정규화는 선형변환)."""
    N, T, C = X.shape
    hps = unit_minutes / 60
    filled = X.copy()
    for c, f in enumerate(feats):
        ph = FEATURE_PERIODS_HOURS.get(f, int(hps))
        spb = max(1, round(ph / hps))
        if spb <= 1:
            continue  # 1h-slot 피처는 채울 것 없음
        nb = ceil(T / spb)
        for b in range(nb):
            s, e = b * spb, min((b + 1) * spb, T)
            chunk = X[:, s:e, c]                       # (N, binlen)
            allnan = np.isnan(chunk).all(axis=1)       # (N,)
            with np.errstate(invalid="ignore"):
                binmean = np.nanmean(chunk, axis=1)    # (N,)
            filled[:, s:e, c] = np.where(
                allnan[:, None], filled[:, s:e, c], binmean[:, None])
    return filled


# =====================================================
# subject-level 결측률
# =====================================================

def compute_sample_missrate(mask, feats, unit_minutes):
    """mask:(N,T,C) bool(True=NaN) -> M:(N,C) 주기 bin 기준 결측률."""
    N, T, C = mask.shape
    hps = unit_minutes / 60
    M = np.zeros((N, C), dtype=np.float32)
    for c, f in enumerate(feats):
        ph = FEATURE_PERIODS_HOURS.get(f, int(hps))
        spb = max(1, round(ph / hps))
        nb = ceil(T / spb)
        binmiss = np.zeros((N, nb), dtype=bool)
        for b in range(nb):
            s, e = b * spb, min((b + 1) * spb, T)
            binmiss[:, b] = mask[:, s:e, c].all(axis=1)
        M[:, c] = binmiss.mean(axis=1)
    return M


# =====================================================
# low/high split (원본 main()의 리팩터판 — 재사용 가능한 함수)
# =====================================================

def split_low_high(data_dir, target, subset=1, out_base=None, seed=SEED):
    """baseline npz(data_dir/processed_data/{subset}.npz)를 결측률 기준으로
    low_missing/high_missing 두 그룹으로 나눠 out_base 아래에 저장한다.

    data_dir : baseline 데이터셋 디렉터리 (feature_columns.json/subject_ids.json/
               sample_ids.json/meta.json + processed_data/{subset}.npz 포함).
    out_base : low_missing/high_missing을 저장할 부모 디렉터리. 기본값은
               data_dir.parent (원본 스크립트와 동일한 레이아웃: DATA_ROOT/"mtm"/...).
    """
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    data_dir = Path(data_dir)
    feats = json.load(open(data_dir / "feature_columns.json"))
    subject_ids = np.array(json.load(open(data_dir / "subject_ids.json")))
    sample_ids = json.load(open(data_dir / "sample_ids.json"))
    meta = json.loads((data_dir / "meta.json").read_text())
    unit_minutes = meta["unit_minutes"]

    out_base = Path(out_base) if out_base is not None else data_dir.parent

    base_npz = data_dir / "processed_data" / f"{subset}_{target}.npz"
    d = np.load(base_npz, allow_pickle=True)
    n_train, n_val = len(d["train_y"]), len(d["val_y"])
    n_test = len(d["test_y"])

    X = np.concatenate([d["train_x"], d["val_x"], d["test_x"]], axis=0)
    Y = np.concatenate([d["train_y"], d["val_y"], d["test_y"]], axis=0)

    # ── 결측률 측정 (fill 여부와 무관) ─────────────────────────
    print("[*] computing period-adjusted missing rate...")
    mask = np.isnan(X)
    M = compute_sample_missrate(mask, feats, unit_minutes)
    sample_score = M.mean(axis=1)

    df = pd.DataFrame({"subject_id": subject_ids, "score": sample_score, "y": Y})
    subj = df.groupby("subject_id").agg(score=("score", "mean"), y=("y", "max"))

    # ── rank-based median split ────────────────────────────────
    subj_sorted = subj.sort_values("score", kind="stable")
    half = len(subj_sorted) // 2
    low_subjects = set(subj_sorted.iloc[:half].index)
    subj["group"] = np.where(subj.index.isin(low_subjects), "low", "high")
    median = float(subj_sorted.iloc[half]["score"])

    print("\n[*] group summary (subject-level, before sample-count matching)")
    for gname in ["low", "high"]:
        g = subj[subj["group"] == gname]
        print(f"  {gname:4s}: n_subj={len(g)}  patient_ratio={g['y'].mean():.3f}  "
              f"score=[{g['score'].min():.3f}, {g['score'].max():.3f}]  mean={g['score'].mean():.3f}")

    # ── 그룹 소속(subject_id 리스트)을 모델 공용으로 저장 ─────────
    # CoFormer 등 다른 모델의 grouping이 결측률을 다시 계산하지 않고 이 소속을 그대로
    # 재사용 -> 두 모델이 완전히 동일한 사람 집합으로 비교됨.
    membership_path = data_dir.parent.parent / "group_membership.json"
    membership = {
        "source": str(data_dir),
        "target": target,
        "median_score": median,
        "low_subjects": sorted(str(s) for s in low_subjects),
        "high_subjects": sorted(str(s) for s in subj.index[~subj.index.isin(low_subjects)]),
    }
    membership_path.write_text(json.dumps(membership, indent=2, ensure_ascii=False))
    print(f"  saved group membership -> {membership_path}")

    # ── 측정주기별 값 채우기 (학습 데이터용) ───────────────────
    print("\n[*] applying period-fill to training data...")
    X_filled = apply_period_fill(X, feats, unit_minutes)

    subj_group_of = subj["group"].to_dict()
    sample_group = np.array([subj_group_of[s] for s in subject_ids])

    # ── split 경계별 그룹 인덱스 + 표본 수 맞추기 ──────────────
    rng = np.random.default_rng(seed)
    split_bounds = {"train": (0, n_train),
                    "val": (n_train, n_train + n_val),
                    "test": (n_train + n_val, n_train + n_val + n_test)}

    group_split_idx = {"low": {}, "high": {}}
    print("\n[*] sample-count matching per split")
    for split, (s, e) in split_bounds.items():
        seg_group = sample_group[s:e]
        low_idx = np.where(seg_group == "low")[0] + s
        high_idx = np.where(seg_group == "high")[0] + s
        matched_n = min(len(low_idx), len(high_idx))
        if len(low_idx) > matched_n:
            low_idx = np.sort(rng.choice(low_idx, matched_n, replace=False))
        if len(high_idx) > matched_n:
            high_idx = np.sort(rng.choice(high_idx, matched_n, replace=False))
        group_split_idx["low"][split] = low_idx
        group_split_idx["high"][split] = high_idx
        print(f"  {split:5s}: low={len(low_idx)}  high={len(high_idx)}  (target={matched_n})")

    # ── 그룹별 데이터셋 저장 ───────────────────────────────────
    npz_names = [f"{subset}_{target}.npz"]

    for gname, out_name in [("low", "low_missing"), ("high", "high_missing")]:
        out_dir = out_base / out_name
        (out_dir / "processed_data").mkdir(parents=True, exist_ok=True)
        idx = group_split_idx[gname]

        (out_dir / "feature_columns.json").write_text(json.dumps(feats, ensure_ascii=False, indent=2))
        keep_all = np.concatenate([idx["train"], idx["val"], idx["test"]])
        (out_dir / "sample_ids.json").write_text(
            json.dumps([sample_ids[i] for i in keep_all], ensure_ascii=False))
        (out_dir / "subject_ids.json").write_text(
            json.dumps([subject_ids[i] for i in keep_all], ensure_ascii=False))

        shapes = {}
        for npz_name in npz_names:
            dd = np.load(data_dir / "processed_data" / npz_name, allow_pickle=True)
            # X는 fill된 전체 배열에서 인덱싱, y/stat은 원본 npz에서
            Xf_all = X_filled  # (N,T,C) filled, train->val->test 순
            filtered = {
                "train_x": Xf_all[idx["train"]], "train_y": dd["train_y"][idx["train"]],
                "train_stat": dd["train_stat"][idx["train"]],
                "val_x": Xf_all[idx["val"]], "val_y": dd["val_y"][idx["val"] - n_train],
                "val_stat": dd["val_stat"][idx["val"] - n_train],
                "test_x": Xf_all[idx["test"]], "test_y": dd["test_y"][idx["test"] - n_train - n_val],
                "test_stat": dd["test_stat"][idx["test"] - n_train - n_val],
            }
            np.savez(out_dir / "processed_data" / npz_name, **filtered)
            shapes = {k: v.shape for k, v in filtered.items() if k.endswith("_x")}
            print(f"  saved {out_dir / 'processed_data' / npz_name}  "
                  f"train={shapes['train_x'][0]} val={shapes['val_x'][0]} test={shapes['test_x'][0]}")

        g = subj[subj["group"] == gname]
        group_meta = dict(meta)
        group_meta["data_version"] = "v3"
        group_meta["missing_group"] = out_name
        group_meta["missing_split_method"] = "subject-level rank-based median split (period-adjusted), sample-count matched, period-filled"
        group_meta["median_threshold"] = median
        group_meta["train_x_shape"] = list(shapes["train_x"])
        group_meta["val_x_shape"] = list(shapes["val_x"])
        group_meta["test_x_shape"] = list(shapes["test_x"])
        group_meta["unique_patients"] = len(g)
        group_meta["patient_ratio"] = float(g["y"].mean())
        group_meta["missing_score_range"] = [float(g["score"].min()), float(g["score"].max())]
        group_meta["missing_score_mean"] = float(g["score"].mean())
        (out_dir / "meta.json").write_text(json.dumps(group_meta, indent=2, ensure_ascii=False))
        print(f"  -> {out_dir}")

    print("\nDone.")


def split_coformer_low_high(data_dir, target, out_base=None, seed=SEED):
    """MTM에서 확정한 target별 subject low/high 소속을 CoFormer packed 데이터에 적용한다.

    각 split에서 두 그룹의 window 수를 동일하게 맞추고, period-fill 후 packed 포맷으로
    저장한다. 따라서 모델 간 low/high의 사람 정의는 같고 window 구성만 모델별로 다르다.
    """
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    from .packing import load_packed, parse_wstart_min, unpack_dense, repack_from_dense

    data_dir = Path(data_dir)
    out_base = Path(out_base) if out_base is not None else data_dir.parent
    packed = load_packed(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    if meta.get("target") != target:
        raise ValueError(f"dataset target={meta.get('target')} does not match {target}")

    membership_path = DATA_ROOT / "mtm" / target / "group_membership.json"
    membership = json.loads(membership_path.read_text())
    low_subjects = set(map(str, membership["low_subjects"]))
    high_subjects = set(map(str, membership["high_subjects"]))
    sample_group = np.array([
        "low" if str(sid) in low_subjects else "high" if str(sid) in high_subjects else "unknown"
        for sid in packed["sids"]
    ])
    if np.any(sample_group == "unknown"):
        unknown = np.unique(packed["sids"][sample_group == "unknown"])
        raise ValueError(f"{len(unknown)} CoFormer subjects are absent from {membership_path}")

    unit_minutes = int(meta["unit_minutes"])
    T = int(meta["window_slots"])
    wstart = parse_wstart_min(packed["smids"])
    dense = unpack_dense(packed["array"], packed["time"], packed["mask"],
                         wstart, T, unit_minutes)
    dense = apply_period_fill(dense, packed["feats"], unit_minutes)

    rng = np.random.default_rng(seed)
    selected = {"low": {}, "high": {}}
    print("[*] applying MTM subject membership and matching window counts per split")
    for split_name, base_idx in zip(("train", "val", "test"), packed["split"]):
        base_idx = np.asarray(base_idx, dtype=np.int64)
        low_idx = base_idx[sample_group[base_idx] == "low"]
        high_idx = base_idx[sample_group[base_idx] == "high"]
        matched_n = min(len(low_idx), len(high_idx))
        if len(low_idx) > matched_n:
            low_idx = np.sort(rng.choice(low_idx, matched_n, replace=False))
        if len(high_idx) > matched_n:
            high_idx = np.sort(rng.choice(high_idx, matched_n, replace=False))
        selected["low"][split_name] = low_idx
        selected["high"][split_name] = high_idx
        print(f"  {split_name:5s}: low={len(low_idx)} high={len(high_idx)}")

    optional_json = {}
    for filename in ("origins.json", "w_start_days.json"):
        path = data_dir / filename
        if path.exists():
            optional_json[filename] = json.loads(path.read_text())

    for group, out_name in (("low", "low_missing"), ("high", "high_missing")):
        out_dir = out_base / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        idx_parts = [selected[group][name] for name in ("train", "val", "test")]
        keep = np.concatenate(idx_parts)
        array, time_arr, mask = repack_from_dense(
            dense[keep], wstart[keep], unit_minutes, packed["array"].shape[2])
        counts = [len(x) for x in idx_parts]
        split = np.empty(3, dtype=object)
        start = 0
        for i, count in enumerate(counts):
            split[i] = np.arange(start, start + count)
            start += count

        np.save(out_dir / "array.npy", array)
        np.save(out_dir / "time.npy", time_arr)
        np.save(out_dir / "mask.npy", mask)
        np.save(out_dir / "gt.npy", packed["gt"][keep])
        np.save(out_dir / "static.npy", packed["static"][keep])
        np.save(out_dir / "split.npy", split, allow_pickle=True)
        (out_dir / "feature_columns.json").write_text(
            json.dumps(packed["feats"], ensure_ascii=False, indent=2))
        (out_dir / "subject_ids.json").write_text(
            json.dumps(packed["sids"][keep].tolist(), ensure_ascii=False))
        (out_dir / "sample_ids.json").write_text(
            json.dumps([packed["smids"][i] for i in keep], ensure_ascii=False))
        for filename, values in optional_json.items():
            (out_dir / filename).write_text(
                json.dumps([values[i] for i in keep], ensure_ascii=False))

        group_subjects = low_subjects if group == "low" else high_subjects
        group_meta = dict(meta)
        group_meta.update({
            "missing_group": out_name,
            "missing_split_method": "MTM target-specific subject membership, per-split window-count matched, period-filled",
            "membership_source": str(membership_path),
            "array_shape": list(array.shape),
            "train": counts[0], "val": counts[1], "test": counts[2],
            "unique_patients": len(group_subjects),
        })
        (out_dir / "meta.json").write_text(
            json.dumps(group_meta, indent=2, ensure_ascii=False))
        print(f"  saved {out_name} -> {out_dir} {array.shape}")

    return out_base / "low_missing", out_base / "high_missing"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["mtm", "coformer"], default="mtm")
    ap.add_argument("--target", choices=["phq9", "gad7"], required=True)
    ap.add_argument("--datapath", type=str, default=None)
    ap.add_argument("--subset", type=int, default=1)
    args = ap.parse_args()
    default = DATA_ROOT / args.format / args.target / "datasets" / "baseline"
    datapath = args.datapath or str(default)
    if args.format == "mtm":
        split_low_high(datapath, target=args.target, subset=args.subset)
    else:
        split_coformer_low_high(datapath, target=args.target)
