"""
step2_arrays.py v3 (formerly step2_coformer.py)
================================================
Run as: `python -m src.preprocess.step2_arrays` from repo root

변경사항:
- window=3일(72slot), stride=1일(24slot) sliding
- 14일을 앞9일/뒤5일로 나눔
  앞9일 → 7개 샘플 → train
  뒤5일 → 3개 샘플 → 1개 val, 2개 test
- subject 단위 분리 없음
- daily 24slot, EMA 168slot 채우기 유지
"""

import argparse
import json
import pickle
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..paths import DATA_ROOT

warnings.filterwarnings("ignore")

# =====================================================
# 설정
# =====================================================

# 캐시 위치: step1_prepare.py가 쓰는 통합 캐시 (step3와 동일 경로 — drift 없음)
CACHE_DIR      = DATA_ROOT / "cache"

# 24-feature 모드: 분당센서 mean 집계만(11) + daily 13 = 24
# False면 기존 46-feature(분당센서 mean/min/max + daily)
MEAN_ONLY_24 = True

# 분당 센서 목록에 있지만 실측 주기가 daily 인 피처.
# 하루 평균으로 압축한 뒤 daily 와 동일하게 24슬롯으로 채운다.
#   근거(관측이 있는 윈도우 기준, 관측일당 관측 시간 수):
#     hr/rr 14.7~14.8h, core_temp/skin_temp/bp/glucose 21.4~22.1h  → 연속 측정
#     spo2 6.3h, hrv 2.3h, proximity 1.8h                          → 하루 1회꼴
#   light_sensor 는 관측값의 4.18%가 -1 센티널이고 긴 오른쪽 꼬리를 가진다.
#   unsigned 변환된 -1은 아래 sentinel guard가 제거하고, 전체 feature의 긴 꼬리는
#   저장 직전 train-only percentile clip + z-score로 처리한다.
MINUTE_TO_DAILY = ["spo2", "hrv", "proximity"]
MINUTE_TO_DAILY_FILL = 24

CHUNK_MINUTES = 60

WINDOW_DAYS   = 3
STRIDE_DAYS   = 1
WINDOW_SLOTS  = WINDOW_DAYS * 24   # 72
STRIDE_SLOTS  = STRIDE_DAYS * 24   # 24

TRAIN_DAYS    = 9
VALTEST_DAYS  = 5
TOTAL_DAYS    = TRAIN_DAYS + VALTEST_DAYS  # 14

RANDOM_SEED   = 42
EPS = 1e-6
CLIP_PERCENTILES = (0.05, 99.95)
WRAPPED_NEGATIVE_SENTINEL_MIN = 1e18

DAILY_FEATURES = [
    "step", "wake_time", "deep_sleep_time", "rem_sleep_time", "screen_time",
    "light_sleep_time", "distance", "sleep_time", "total_sleep_time",
    "EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress",
]

FILL_SLOTS = {
    "step": 24, "wake_time": 24, "deep_sleep_time": 24,
    "rem_sleep_time": 24, "screen_time": 24, "light_sleep_time": 24,
    "distance": 24, "sleep_time": 24, "total_sleep_time": 24,
    "EMA_Anxiety": 168, "EMA_Depression": 168,
    "EMA_Sleep": 168, "EMA_Stress": 168,
}

# =====================================================
# 유틸
# =====================================================

def to_naive_ts(x):
    if pd.isna(x):
        return pd.NaT
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts

def dt_to_cum_minutes(dt, origin):
    return int((to_naive_ts(dt) - to_naive_ts(origin)).total_seconds() // 60)

def safe_numeric(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def encode_sex(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    # raw 코드북: 1=남(전 회차 공통), 0/2=여
    # 1회차 fixed 파일만 0/1로 재인코딩되어 있고(0=여,1=남), 2/3회차는 1/2(1=남,2=여)
    if x in ("남", "1"):
        return 0.0
    if x in ("여", "0", "2"):
        return 1.0
    return np.nan

def parse_age(x):
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

def parse_numeric_val(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    x = str(x).strip().replace(" ", "")
    if "," in x and "." not in x and x.count(",") == 1:
        x = x.replace(",", ".")
    try:
        return float(x)
    except Exception:
        return np.nan

def fill_nan_with_col_mean(arr):
    out = arr.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        if np.isnan(col).all():
            out[:, j] = 0.0 # 여기서
        else:
            out[np.isnan(col), j] = np.nanmean(col)
    return out

def clean_sensor_values(values, feature):
    """Numeric conversion plus a defensive guard for wrapped negative sentinels."""
    values = pd.to_numeric(values, errors="coerce")
    if feature == "light_sensor":
        values = values.where(
            (values >= 0) & (values < WRAPPED_NEGATIVE_SENTINEL_MIN)
        )
    return values.dropna()

def normalize_packed_arrays(arrays, masks):
    """Normalize observed values with train-only robust statistics; keep padding zero."""
    train_array, _, _ = arrays
    train_mask, _, _ = masks
    n_features = train_array.shape[1]
    slots = np.arange(train_array.shape[2])[None, :]
    clip_lo = np.zeros(n_features, dtype=np.float64)
    clip_hi = np.zeros(n_features, dtype=np.float64)
    means = np.zeros(n_features, dtype=np.float64)
    stds = np.ones(n_features, dtype=np.float64)

    for feature_idx in range(n_features):
        train_valid = slots < train_mask[:, feature_idx, None]
        train_values = train_array[:, feature_idx, :][train_valid]
        train_values = train_values[np.isfinite(train_values)]
        if train_values.size == 0:
            continue

        lo, hi = np.percentile(train_values, CLIP_PERCENTILES)
        clipped = np.clip(train_values, lo, hi)
        mean = float(clipped.mean())
        std = float(clipped.std())
        if not np.isfinite(std) or std < EPS:
            std = 1.0

        clip_lo[feature_idx], clip_hi[feature_idx] = lo, hi
        means[feature_idx], stds[feature_idx] = mean, std
        for array, mask in zip(arrays, masks):
            valid = slots < mask[:, feature_idx, None]
            column = array[:, feature_idx, :]
            column[valid] = (np.clip(column[valid], lo, hi) - mean) / std
            column[~valid] = 0.0

    return {
        "method": "train-observed percentile clip then z-score",
        "clip_percentiles": list(CLIP_PERCENTILES),
        "clip_lo": clip_lo.tolist(),
        "clip_hi": clip_hi.tolist(),
        "mean": means.tolist(),
        "std": stds.tolist(),
    }

def slice_window(df, date_col, start_date, end_date, is_daily):
    if is_daily:
        end_date = end_date - pd.Timedelta(days=1)
    else:
        end_date = end_date - pd.Timedelta(minutes=1)
    return df.loc[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()

# =====================================================
# 집계
# =====================================================

def aggregate_chunk(minute_df, dt_col, feature_cols, origin, chunk_minutes, agg_list):
    out = {f"{c}_{a}": [] for c in feature_cols for a in agg_list}
    if len(minute_df) == 0:
        return out
    df = minute_df.copy().set_index(dt_col)
    grouped = df.resample(f"{chunk_minutes}min", label="right", closed="right")
    for end_time, g in grouped:
        if len(g) == 0:
            continue
        end_min = dt_to_cum_minutes(pd.Timestamp(end_time), origin)
        for col in feature_cols:
            if col not in g.columns:
                continue
            vals = clean_sensor_values(g[col], col)
            if len(vals) == 0:
                continue
            for agg_name in agg_list:
                val = vals.mean() if agg_name == "mean" else getattr(vals, agg_name)()
                if pd.notna(val):
                    out[f"{col}_{agg_name}"].append((end_min, float(val)))
    return out

def build_minute_daily_series(minute_df, feature_cols, origin):
    """분당 센서지만 실측 주기가 daily 인 피처(MINUTE_TO_DAILY)를 하루 평균으로
    압축한 뒤 build_daily_series 와 동일하게 24슬롯으로 채운다.

    시간 단위로 두면 관측이 하루 중 일부 시각에만 몰려(예: spo2 는 관측의 92.5%가
    00~08시) 결측률이 과대평가되고, 모델 입력에서도 72슬롯 중 2~6개만 채워진
    희소 채널이 된다.
    """
    out = {c: [] for c in feature_cols}
    if len(minute_df) == 0 or not feature_cols:
        return out

    df = minute_df.copy()
    df["_date"] = pd.to_datetime(df["datetime"]).dt.normalize()

    for date, g in df.groupby("_date"):
        day_start_min = int(
            (pd.Timestamp(date) - pd.Timestamp(origin)).total_seconds() // 60
        )
        for c in feature_cols:
            if c not in g.columns:
                continue
            vals = clean_sensor_values(g[c], c)
            if len(vals) == 0:
                continue
            val = float(vals.mean())
            for slot_offset in range(MINUTE_TO_DAILY_FILL):
                slot_min = day_start_min + slot_offset * CHUNK_MINUTES
                if slot_min >= 0:
                    out[c].append((slot_min, val))
    return out


def build_daily_series(daily_df, origin):
    out = {c: [] for c in DAILY_FEATURES}
    for _, row in daily_df.iterrows():
        day_start_min = int(
            (pd.Timestamp(row["date"]) - pd.Timestamp(origin)).total_seconds() // 60
        )
        for c in DAILY_FEATURES:
            val = safe_numeric(row.get(c, np.nan))
            if pd.isna(val):
                continue
            fill = FILL_SLOTS.get(c, 24)
            for slot_offset in range(fill):
                slot_min = day_start_min + slot_offset * CHUNK_MINUTES
                if slot_min >= 0:
                    out[c].append((slot_min, float(val)))
    return out

# =====================================================
# sliding window 추출
# =====================================================

def extract_windows(all_series, canonical_features, origin, n_days):
    """n_days 구간에서 window=72slot, stride=24slot으로 sliding"""
    total_slots = n_days * 24
    windows = []

    for w_start in range(0, total_slots - WINDOW_SLOTS + 1, STRIDE_SLOTS):
        w_end = w_start + WINDOW_SLOTS
        w_start_min = w_start * CHUNK_MINUTES
        w_end_min   = w_end   * CHUNK_MINUTES

        array_2d = np.zeros((len(canonical_features), WINDOW_SLOTS), dtype=np.float32)
        time_2d  = np.full((len(canonical_features), WINDOW_SLOTS), -1, dtype=np.float32)
        mask_1d  = np.zeros((len(canonical_features),), dtype=np.int32)

        has_data = False
        for i, feat in enumerate(canonical_features):
            seq = [(t, v) for t, v in all_series.get(feat, [])
                   if w_start_min <= t < w_end_min]
            seq = sorted(seq, key=lambda x: x[0])
            # time을 window 내 상대 slot으로 변환
            seq_rel = [(t - w_start_min, v) for t, v in seq]
            seq_rel = [(t // CHUNK_MINUTES, v) for t, v in seq_rel
                       if 0 <= t // CHUNK_MINUTES < WINDOW_SLOTS]
            mask_1d[i] = len(seq_rel)
            if seq_rel:
                has_data = True
            for t_idx, (slot, value) in enumerate(seq_rel[:WINDOW_SLOTS]):
                array_2d[i, t_idx] = value
                time_2d[i, t_idx]  = float(w_start_min + slot * CHUNK_MINUTES)

        if has_data:
            windows.append({
                "array": array_2d,
                "time":  time_2d,
                "mask":  mask_1d,
                "w_start_day": w_start // 24,
            })

    return windows

# =====================================================
# 샘플 생성
# =====================================================

def build_samples(sid, survey_date, direction, row,
                  minute_df_cache, daily_df_cache, minute_feat_cache,
                  canonical_features, agg_list):
    survey_day = to_naive_ts(survey_date).normalize()

    if direction == "before":
        start_date = survey_day - timedelta(days=TOTAL_DAYS)
        end_date   = survey_day
    else:
        start_date = survey_day + timedelta(days=1)
        end_date   = survey_day + timedelta(days=TOTAL_DAYS + 1)

    origin = start_date
    train_end_date  = start_date + timedelta(days=TRAIN_DAYS)
    valtest_end_date = end_date

    # 데이터 슬라이싱
    daily_slice = pd.DataFrame()
    if sid in daily_df_cache:
        daily_slice = slice_window(daily_df_cache[sid], "date",
                                   start_date, end_date, is_daily=True)

    minute_slice = pd.DataFrame()
    minute_feats = []
    if sid in minute_df_cache:
        minute_slice = slice_window(minute_df_cache[sid], "datetime",
                                    start_date, end_date, is_daily=False)
        minute_feats = [c for c in minute_feat_cache[sid]
                        if c in minute_slice.columns
                        and minute_slice[c].notna().sum() > 0]

    # 분당 센서 중 daily 로 전환할 것과 시간 집계로 남길 것을 분리
    minute_feats_hourly = [c for c in minute_feats if c not in MINUTE_TO_DAILY]
    minute_feats_daily  = [c for c in minute_feats if c in MINUTE_TO_DAILY]

    # series 생성 (14일 전체)
    minute_series = aggregate_chunk(
        minute_slice, "datetime", minute_feats_hourly, origin, CHUNK_MINUTES, agg_list
    ) if len(minute_slice) > 0 and len(minute_feats_hourly) > 0 else {}

    minute_daily_series = build_minute_daily_series(
        minute_slice, minute_feats_daily, origin
    ) if len(minute_slice) > 0 and len(minute_feats_daily) > 0 else {}

    daily_series = build_daily_series(
        daily_slice, origin
    ) if len(daily_slice) > 0 else {}

    all_series = {**minute_series, **minute_daily_series, **daily_series}

    if sum(len(v) for v in all_series.values()) == 0:
        return [], []

    # 앞 9일 → train windows
    train_windows  = extract_windows(all_series, canonical_features, origin, TRAIN_DAYS)

    # 뒤 5일 (offset: 9일 이후) → valtest windows
    # 뒤 5일 series만 필터링
    train_min = TRAIN_DAYS * 24 * CHUNK_MINUTES
    valtest_series = {
        feat: [(t - train_min, v) for t, v in seqs if t >= train_min]
        for feat, seqs in all_series.items()
    }
    valtest_windows = extract_windows(valtest_series, canonical_features, origin, VALTEST_DAYS)

    static = [
        encode_sex(row.get("Sex")),
        parse_age(row.get("Age")),
        parse_numeric_val(row.get("Height")),
        parse_numeric_val(row.get("Weight")),
    ]

    def make_sample(w, split, idx):
        w_start_min = w["w_start_day"] * 24 * CHUNK_MINUTES
        return {
            "subject_id": sid,
            "sample_id":  f"{sid}_{survey_date.date()}_{direction}_{split}_{idx}_s{w_start_min}",
            "split":      split,
            "array":      w["array"],
            "time":       w["time"],
            "mask":       w["mask"],
            "phq9_label": row["phq9_label"],
            "gad7_label": row["gad7_label"],
            "static":     static,
            "origin_date": str(pd.Timestamp(origin).date()),
            "w_start_day": w["w_start_day"],
        }

    train_samples = [make_sample(w, "train", i) for i, w in enumerate(train_windows)]

    # 뒤 5일 3개 → 1개 val, 2개 test
    valtest_samples = []
    for i, w in enumerate(valtest_windows):
        split = "val" if i == 0 else "test"
        valtest_samples.append(make_sample(w, split, i))

    return train_samples, valtest_samples

# =====================================================
# CLI / 메인
# =====================================================

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["phq9", "gad7"], required=True)
    parser.add_argument("--output", default=None,
                        help="기본값: data/coformer/<target>/datasets/baseline")
    return parser.parse_args()


def main():
    args = build_args()
    target = args.target
    label_col = f"{target}_label"
    score_col = "PHQ9_Score" if target == "phq9" else "GAD7_Score"
    numpy_output = (Path(args.output) if args.output else
                    DATA_ROOT / "coformer" / target / "datasets" / "baseline")
    print("=" * 60)
    print(f"  step2_arrays.py v3 — target={target}")
    print(f"  window={WINDOW_DAYS}일({WINDOW_SLOTS}slot), stride={STRIDE_DAYS}일({STRIDE_SLOTS}slot)")
    print(f"  앞{TRAIN_DAYS}일=train, 뒤{VALTEST_DAYS}일=val/test")
    print("=" * 60)

    if numpy_output.exists() and any(numpy_output.iterdir()):
        raise FileExistsError(f"output already exists; refusing to mix files: {numpy_output}")
    numpy_output.mkdir(parents=True, exist_ok=True)

    print("\n  캐시 로드 중...")
    with open(CACHE_DIR / "survey_all.pkl", "rb") as f:
        survey_all = pickle.load(f)
    with open(CACHE_DIR / "daily_df_cache.pkl", "rb") as f:
        daily_df_cache = pickle.load(f)
    with open(CACHE_DIR / "minute_df_cache.pkl", "rb") as f:
        minute_df_cache = pickle.load(f)
    with open(CACHE_DIR / "minute_feat_cache.pkl", "rb") as f:
        minute_feat_cache = pickle.load(f)
    with open(CACHE_DIR / "feature_info.json", "r") as f:
        feature_info = json.load(f)

    if label_col not in survey_all.columns:
        raise KeyError(f"survey cache has no target label column: {label_col}")
    survey_all = survey_all[survey_all[label_col].notna()].copy()

    # MTM과 CoFormer가 같은 사람 집합을 비교하도록 MTM에서 확정한 target별
    # low/high membership의 합집합을 공통 subject universe로 사용한다.
    membership_path = DATA_ROOT / "mtm" / target / "group_membership.json"
    membership = json.loads(membership_path.read_text())
    allowed_subjects = set(map(str, membership["low_subjects"] + membership["high_subjects"]))
    survey_all = survey_all[survey_all["ID"].astype(str).isin(allowed_subjects)].copy()

    canonical_features = feature_info["canonical_features"]
    agg_list           = feature_info["agg_list"]

    if MEAN_ONLY_24:
        agg_list = ["mean"]
        canonical_features = [c for c in canonical_features
                              if c.endswith("_mean") or c in DAILY_FEATURES]
        print(f"  [24feat] agg={agg_list}, features={len(canonical_features)}개")

    if MINUTE_TO_DAILY:
        # spo2_mean 같은 시간집계 채널을 빼고 bare name 으로 daily 블록 앞에 넣는다.
        # (canonical_features = minute_union + DAILY_FEATURES 순서라 daily 는 항상 뒤쪽)
        drop = {f"{f}_{a}" for f in MINUTE_TO_DAILY for a in agg_list}
        kept = [c for c in canonical_features if c not in drop]
        at = len(kept) - len([c for c in kept if c in DAILY_FEATURES])
        canonical_features = kept[:at] + list(MINUTE_TO_DAILY) + kept[at:]
        print(f"  [daily 전환] {MINUTE_TO_DAILY} -> 하루평균 + {MINUTE_TO_DAILY_FILL}슬롯 fill")
        print(f"  [daily 전환] features={len(canonical_features)}개")

    print(f"  설문: {len(survey_all)}건, features: {len(canonical_features)}개")

    all_train, all_val, all_test = [], [], []
    skipped = 0

    for _, row in tqdm(survey_all.iterrows(), total=len(survey_all), desc="  Processing"):
        sid         = row["ID"]
        survey_date = row["timestamp"]
        try:
            for direction in ["before", "after"]:
                train_s, valtest_s = build_samples(
                    sid, survey_date, direction, row,
                    minute_df_cache, daily_df_cache, minute_feat_cache,
                    canonical_features, agg_list,
                )
                if not train_s and not valtest_s:
                    skipped += 1
                    continue
                all_train.extend(train_s)
                for s in valtest_s:
                    if s["split"] == "val":
                        all_val.append(s)
                    else:
                        all_test.append(s)
        except Exception as e:
            print(f"  [SKIP] {sid}: {e}")

    print(f"\n  train: {len(all_train)}, val: {len(all_val)}, test: {len(all_test)}")
    print(f"  skipped: {skipped}")

    def build_arrays(samples):
        n = len(samples)
        nf = len(canonical_features)
        if n == 0:
            return (
                np.zeros((0, nf, WINDOW_SLOTS), dtype=np.float32),
                np.full((0, nf, WINDOW_SLOTS), -1, dtype=np.float32),
                np.zeros((0, nf), dtype=np.int32),
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                [], [], [], [],
            )
        arr   = np.zeros((n, nf, WINDOW_SLOTS), dtype=np.float32)
        time  = np.full((n, nf, WINDOW_SLOTS), -1, dtype=np.float32)
        mask  = np.zeros((n, nf), dtype=np.int32)
        gt    = np.zeros((n, 1), dtype=np.float32)
        stat  = np.full((n, 4), np.nan, dtype=np.float32)
        sids, smids, origs, wdays = [], [], [], []
        for i, s in enumerate(samples):
            arr[i]  = s["array"]
            time[i] = s["time"]
            mask[i] = s["mask"]
            gt[i,0] = s[label_col]
            stat[i] = np.array(s["static"], dtype=np.float32)
            sids.append(s["subject_id"])
            smids.append(s["sample_id"])
            origs.append(s["origin_date"])
            wdays.append(s["w_start_day"])
        stat = fill_nan_with_col_mean(stat)
        stat[(stat[:,1]<0)|(stat[:,1]>120),1] = np.nan
        stat[(stat[:,2]<100)|(stat[:,2]>250),2] = np.nan
        stat[(stat[:,3]<20)|(stat[:,3]>300),3] = np.nan
        stat = fill_nan_with_col_mean(stat)
        return arr, time, mask, gt, stat, sids, smids, origs, wdays

    print("\n  Numpy 조립...")
    tr = build_arrays(all_train)
    va = build_arrays(all_val)
    te = build_arrays(all_test)

    print("  train 기준 robust normalization ...")
    normalization = normalize_packed_arrays(
        arrays=(tr[0], va[0], te[0]),
        masks=(tr[2], va[2], te[2]),
    )

    # split index
    n_tr = len(all_train)
    n_va = len(all_val)
    n_te = len(all_test)
    train_idx = np.arange(n_tr)
    val_idx   = np.arange(n_tr, n_tr + n_va)
    test_idx  = np.arange(n_tr + n_va, n_tr + n_va + n_te)

    # 합쳐서 저장
    array_3d  = np.concatenate([tr[0], va[0], te[0]], axis=0)
    time_3d   = np.concatenate([tr[1], va[1], te[1]], axis=0)

    mask_2d   = np.concatenate([tr[2], va[2], te[2]], axis=0)
    gt_all    = np.concatenate([tr[3], va[3], te[3]], axis=0)
    static_2d = np.concatenate([tr[4], va[4], te[4]], axis=0)
    sids_all  = tr[5] + va[5] + te[5]
    smids_all = tr[6] + va[6] + te[6]
    origs_all = tr[7] + va[7] + te[7]
    wdays_all = tr[8] + va[8] + te[8]

    np.save(numpy_output / "array.npy",  array_3d)
    np.save(numpy_output / "time.npy",   time_3d)
    np.save(numpy_output / "mask.npy",   mask_2d)
    np.save(numpy_output / "gt.npy",     gt_all)
    np.save(numpy_output / "static.npy", static_2d)
    np.save(numpy_output / "split.npy",
            np.array([train_idx, val_idx, test_idx], dtype=object),
            allow_pickle=True)

    with open(numpy_output / "subject_ids.json", "w") as f:
        json.dump(sids_all, f, ensure_ascii=False)
    with open(numpy_output / "sample_ids.json", "w") as f:
        json.dump(smids_all, f, ensure_ascii=False)
    with open(numpy_output / "origins.json", "w") as f:
        json.dump(origs_all, f, ensure_ascii=False)
    with open(numpy_output / "w_start_days.json", "w") as f:
        json.dump(wdays_all, f, ensure_ascii=False)
    with open(numpy_output / "feature_columns.json", "w") as f:
        json.dump(canonical_features, f, ensure_ascii=False, indent=2)

    meta = {
        "data_version": "v3",
        "target": target,
        "target_label_column": label_col,
        "target_score_column": score_col,
        "target_threshold": 10,
        "subject_universe": "MTM target-specific group membership union",
        "membership_source": str(membership_path),
        "unit_minutes": CHUNK_MINUTES,
        "window_slots": WINDOW_SLOTS, "stride_slots": STRIDE_SLOTS,
        "train_days": TRAIN_DAYS, "valtest_days": VALTEST_DAYS,
        "array_shape": list(array_3d.shape),
        "train": n_tr, "val": n_va, "test": n_te,
        "fill_slots": FILL_SLOTS,
        "normalization": normalization,
    }
    with open(numpy_output / "meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n  array: {array_3d.shape}")
    print(f"  train: {n_tr}, val: {n_va}, test: {n_te}")
    gt_sq = gt_all.squeeze()
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"  {name}: {target.upper()} 0={int(sum(gt_sq[idx]==0))}, 1={int(sum(gt_sq[idx]==1))}")
    print(f"\n  완료! → {numpy_output}")

if __name__ == "__main__":
    main()
