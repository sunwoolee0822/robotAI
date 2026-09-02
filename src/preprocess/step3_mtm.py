"""
step3_mtm.py (formerly step3_mtm_v2.py)
========================================
Run as: `python -m src.preprocess.step3_mtm` from repo root

pickle 캐시 로드 → 시간 단위 슬롯 집계 → 각 window를 독립 sample로 저장

split 방식 (시간 기반, subject 내 분리):
  - before/after 각 14일 중
      앞 9일 window → train
      뒤 5일 window → val/test  (샘플 단위 1:2 랜덤 split)
  - 경계 걸치는 window는 제외 (train/val-test 완전 분리)

입력: CACHE_DIR의 pkl/json 파일 (step1_prepare.py 출력)
출력:
  MTM_OUTPUT_V2/
    processed_data/1.npz  (train_x, train_y, train_stat, val_x, ...)
    feature_columns.json, subject_ids.json, sample_ids.json, meta.json

사용법:
  python -m src.preprocess.step3_mtm --window-units 72 --stride-units 24
"""

import argparse
import json
import pickle
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..paths import DATA_ROOT

warnings.filterwarnings("ignore")


# =====================================================
# 설정
# =====================================================

# 캐시 위치: step1_prepare.py가 쓰는 통합 캐시.
# 원본 저장소에서는 step3가 MTM/dataset/cache_all_new 를, step1/step2가
# sw/data/cache_all 을 가리켜 drift가 있었다 — 마이그레이션 후에는 세 스크립트
# 모두 DATA_ROOT / "cache" 하나만 읽고 쓴다 (step1이 쓰고, step2/step3가 읽음).
CACHE_DIR = DATA_ROOT / "cache"

DAYS_BEFORE  = 14
DAYS_AFTER   = 14
TRAIN_DAYS   = 9    # 앞 9일 → train
VALTEST_DAYS = 5    # 뒤 5일 → val/test

RANDOM_SEED = 42
EPS = 1e-8

DAILY_FEATURES = [
    "step", "wake_time", "deep_sleep_time", "rem_sleep_time", "screen_time",
    "light_sleep_time", "distance", "sleep_time", "total_sleep_time",
    "EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress",
]


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


def safe_numeric(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def encode_sex(x):
    # 회차별 인코딩 불일치 수정판 (sw/step2_coformer.py와 동일 로직):
    # 1회차 fixed 파일: 0=여,1=남 (재인코딩됨) / 2,3회차: 1=남,2=여
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x in ("남", "1"):
        return 0.0
    if x in ("여", "0", "2"):
        return 1.0
    return np.nan


def get_age(row):
    """survey_all['age_years'](정제됨) 우선, 없으면 Age 필드로 폴백."""
    if "age_years" in row.index and pd.notna(row.get("age_years")):
        return float(row["age_years"])
    x = row.get("Age")
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
            out[:, j] = 0.0
        else:
            out[np.isnan(col), j] = np.nanmean(col)
    return out


def slice_by_date(df, date_col, survey_date, days_before, days_after, is_daily):
    survey_day = to_naive_ts(survey_date).normalize()
    start = survey_day - timedelta(days=days_before)
    end   = survey_day + timedelta(days=days_after)
    if not is_daily:
        end = end + timedelta(hours=23, minutes=59)
    return df.loc[(df[date_col] >= start) & (df[date_col] <= end)].copy()


# =====================================================
# [v2 핵심] 시간 슬롯 집계 + sliding window → sample 생성
# =====================================================

def build_slot_series(minute_df, dt_col, minute_feat_cols,
                      daily_df, daily_feat_cols,
                      origin, unit_minutes, canonical_features):
    """
    전체 데이터를 unit_minutes 단위 슬롯으로 집계.

    반환:
        slot_array: np.ndarray shape (N_slots, n_features)
        slot_min, slot_max: 유효 슬롯 범위 (origin 기준 절대 슬롯 번호)
    """
    n_feat   = len(canonical_features)
    feat_idx = {f: i for i, f in enumerate(canonical_features)}
    origin_ts = to_naive_ts(origin)

    slot_vals = {}

    def add_val(slot, fi, v):
        if np.isnan(v):
            return
        slot_vals.setdefault(slot, {}).setdefault(fi, []).append(v)

    # ── minute 데이터 슬롯 집계 ─────────────────────────────────────
    if len(minute_df) > 0 and len(minute_feat_cols) > 0:
        df_s      = minute_df.sort_values(dt_col).reset_index(drop=True)
        dt_series = pd.to_datetime(df_s[dt_col], errors="coerce")
        if dt_series.dt.tz is not None:
            dt_series = dt_series.dt.tz_localize(None)
        cum_min   = ((dt_series - origin_ts).dt.total_seconds() // 60).astype(np.int64).values

        for col in minute_feat_cols:
            if col not in feat_idx:
                continue
            fi   = feat_idx[col]
            vals = pd.to_numeric(df_s[col], errors="coerce").values
            for t, v in zip(cum_min, vals):
                slot = int(t) // unit_minutes
                add_val(slot, fi, float(v) if not np.isnan(v) else np.nan)

    # ── daily 데이터 슬롯 집계 ──────────────────────────────────────
    if len(daily_df) > 0 and len(daily_feat_cols) > 0:
        for _, row in daily_df.iterrows():
            day_end_ts = to_naive_ts(row["date"]) + pd.Timedelta(hours=23, minutes=59)
            cum_end    = int((day_end_ts - origin_ts).total_seconds() // 60)
            slot       = cum_end // unit_minutes
            for col in daily_feat_cols:
                if col not in feat_idx:
                    continue
                v = safe_numeric(row.get(col, np.nan))
                add_val(slot, feat_idx[col], v)

    if not slot_vals:
        return None, None, None

    slot_min = min(slot_vals.keys())
    slot_max = max(slot_vals.keys())
    n_slots  = slot_max - slot_min + 1

    slot_array = np.full((n_slots, n_feat), np.nan, dtype=np.float32)

    for slot, feat_dict in slot_vals.items():
        row_idx = slot - slot_min
        for fi, vals in feat_dict.items():
            slot_array[row_idx, fi] = float(np.nanmean(vals))

    return slot_array, slot_min, slot_max


def extract_windows_from_slots(slot_array, slot_min, slot_max,
                                window_units, stride_units,
                                train_threshold_slot):
    """
    슬롯 배열에서 sliding window로 sample 추출.
    각 window에 split_tag('train' / 'val_test') 부여.

    train 조건    : w_start_abs + window_units <= train_threshold_slot  (완전히 앞 9일 안)
    val_test 조건 : w_start_abs >= train_threshold_slot                 (완전히 뒤 5일 안)
    경계 걸치는 window: skip

    반환:
        list of (w_start_abs, split_tag, array(window_units, n_features))
    """
    windows = []
    n_slots = slot_max - slot_min + 1

    if window_units > n_slots:
        return windows

    for w_start in range(0, n_slots - window_units + 1, stride_units):
        w_end_excl  = w_start + window_units
        w_start_abs = slot_min + w_start        # origin 기준 절대 슬롯
        w_end_abs   = slot_min + w_end_excl     # exclusive

        chunk = slot_array[w_start:w_end_excl]

        if np.isnan(chunk).all():
            continue

        if w_end_abs <= train_threshold_slot:
            tag = "train"
        elif w_start_abs >= train_threshold_slot:
            tag = "val_test"
        else:
            continue  # 경계 걸침 → skip

        windows.append((w_start_abs, tag, chunk.copy()))

    return windows


# =====================================================
# 메인
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="step3_mtm: 시간 기반 train/val/test split"
    )
    parser.add_argument("--target", choices=["phq9", "gad7"], required=True,
                        help="학습 target. 해당 score가 있는 설문만 사용")
    parser.add_argument("--unit-minutes", type=int, default=60,
                        help="기본 시간 단위 (분). default: 60")
    parser.add_argument("--window-units", type=int, default=72,
                        help="window 내 timestamp 수. 72 → 3일 (default: 72)")
    parser.add_argument("--stride-units", type=int, default=24,
                        help="sliding stride (단위 수). 24 → 하루씩 이동 (default: 24)")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="입력 캐시 디렉토리 (step1_prepare.py 출력). "
                             f"default: {CACHE_DIR}")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="출력 디렉토리. default: "
                             "DATA_ROOT/mtm/{target}/datasets/baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    UNIT_MINUTES = args.unit_minutes
    WINDOW_UNITS = args.window_units
    STRIDE_UNITS = args.stride_units
    CACHE_DIR_ARG = args.cache_dir
    target = args.target
    target_label_col = f"{target}_label"
    target_score_col = {"phq9": "PHQ9_Score", "gad7": "GAD7_Score"}[target]

    # train/val_test 경계: origin 기준 절대 슬롯
    # TRAIN_DAYS * 24h * 60min / UNIT_MINUTES
    TRAIN_THRESHOLD = TRAIN_DAYS * 24 * 60 // UNIT_MINUTES

    # target별 baseline을 분리해 metadata/label/sample ID가 서로 섞이지 않게 한다.
    MTM_OUTPUT_V2 = args.output_dir or (
        DATA_ROOT / "mtm" / target / "datasets" / "baseline"
    )

    window_hours = UNIT_MINUTES * WINDOW_UNITS // 60
    stride_hours = UNIT_MINUTES * STRIDE_UNITS // 60

    print("=" * 60)
    print(f"  step3_mtm.py: 시간 기반 split (앞 {TRAIN_DAYS}일=train / 뒤 {VALTEST_DAYS}일=val_test)")
    print(f"  unit={UNIT_MINUTES}min | window={WINDOW_UNITS} units ({window_hours}h) "
          f"| stride={STRIDE_UNITS} units ({stride_hours}h)")
    print(f"  train threshold: slot {TRAIN_THRESHOLD} ({TRAIN_DAYS}일)")
    print(f"  sample shape: ({WINDOW_UNITS}, N_features)")
    print("=" * 60)

    MTM_OUTPUT_V2.mkdir(parents=True, exist_ok=True)
    (MTM_OUTPUT_V2 / "processed_data").mkdir(parents=True, exist_ok=True)

    # ── 캐시 로드 ────────────────────────────────────────────────────
    print("\n  캐시 로드 중...")
    with open(CACHE_DIR_ARG / "survey_all.pkl",        "rb") as f: survey_all        = pickle.load(f)
    with open(CACHE_DIR_ARG / "daily_df_cache.pkl",    "rb") as f: daily_df_cache    = pickle.load(f)
    with open(CACHE_DIR_ARG / "minute_df_cache.pkl",   "rb") as f: minute_df_cache   = pickle.load(f)
    with open(CACHE_DIR_ARG / "minute_feat_cache.pkl", "rb") as f: minute_feat_cache = pickle.load(f)

    eligible = survey_all[target_label_col].notna()
    print(f"  target={target.upper()}: valid surveys={int(eligible.sum())}/{len(survey_all)}")
    survey_all = survey_all.loc[eligible].copy()
    survey_all[target_label_col] = survey_all[target_label_col].astype(int)

    # ── canonical features 결정 ──────────────────────────────────────
    print("\n  canonical features 결정 중...")
    all_minute_feats = set()
    for feats in minute_feat_cache.values():
        all_minute_feats.update(feats)
    minute_canonical   = sorted(all_minute_feats)
    canonical_features = minute_canonical + DAILY_FEATURES
    n_feat             = len(canonical_features)
    print(f"  minute features : {len(minute_canonical)}개")
    print(f"  daily  features : {len(DAILY_FEATURES)}개")
    print(f"  total  features : {n_feat}개")

    # ── 설문별 처리 ──────────────────────────────────────────────────
    print(f"\n  설문별 처리 중...")
    train_samples   = []
    valtest_samples = []

    for _, row in tqdm(survey_all.iterrows(), total=len(survey_all), desc="  Processing"):
        sid         = row["ID"]
        survey_date = row["timestamp"]

        has_minute = sid in minute_df_cache
        has_daily  = sid in daily_df_cache
        if not has_minute and not has_daily:
            continue

        static_val = [
            encode_sex(row.get("Sex")),
            get_age(row),
            parse_numeric_val(row.get("Height")),
            parse_numeric_val(row.get("Weight")),
        ]

        for window_tag, days_bef, days_aft in [("before", DAYS_BEFORE, 0), ("after", 0, DAYS_AFTER)]:
            try:
                daily_slice = pd.DataFrame()
                if has_daily:
                    daily_slice = slice_by_date(
                        daily_df_cache[sid], "date", survey_date, days_bef, days_aft, True)

                minute_slice = pd.DataFrame()
                minute_feats = []
                if has_minute:
                    minute_slice = slice_by_date(
                        minute_df_cache[sid], "datetime", survey_date, days_bef, days_aft, False)
                    minute_feats = [c for c in minute_feat_cache[sid]
                                    if c in minute_slice.columns
                                    and minute_slice[c].notna().sum() > 0]

                daily_feats = [c for c in DAILY_FEATURES
                               if len(daily_slice) > 0
                               and c in daily_slice.columns
                               and daily_slice[c].notna().sum() > 0]

                candidates = []
                if len(minute_slice) > 0:
                    candidates.append(minute_slice["datetime"].min().normalize())
                if len(daily_slice) > 0:
                    candidates.append(pd.Timestamp(daily_slice["date"].min()))
                if not candidates:
                    continue
                origin = min(candidates)

                slot_array, slot_min, slot_max = build_slot_series(
                    minute_df        = minute_slice,
                    dt_col           = "datetime",
                    minute_feat_cols = minute_feats,
                    daily_df         = daily_slice,
                    daily_feat_cols  = daily_feats,
                    origin           = origin,
                    unit_minutes     = UNIT_MINUTES,
                    canonical_features = canonical_features,
                )

                if slot_array is None:
                    continue

                windows = extract_windows_from_slots(
                    slot_array           = slot_array,
                    slot_min             = slot_min,
                    slot_max             = slot_max,
                    window_units         = WINDOW_UNITS,
                    stride_units         = STRIDE_UNITS,
                    train_threshold_slot = TRAIN_THRESHOLD,
                )

                for w_start_abs, tag, arr in windows:
                    entry = {
                        "subject_id": sid,
                        "sample_id":  f"{sid}_{survey_date.date()}_{window_tag}_s{w_start_abs}",
                        "array":      arr,
                        "target_label": int(row[target_label_col]),
                        "static":     static_val,
                    }
                    if tag == "train":
                        train_samples.append(entry)
                    else:
                        valtest_samples.append(entry)

            except Exception as e:
                print(f"  [SKIP] {sid} ({window_tag}): {e}")

    if not train_samples:
        raise RuntimeError("no train samples!")
    if not valtest_samples:
        raise RuntimeError("no val/test samples!")

    print(f"\n  train samples   : {len(train_samples)}개")
    print(f"  val_test samples: {len(valtest_samples)}개")

    # ── val/test 1:2 샘플 단위 랜덤 split ───────────────────────────
    vt_idx_all = list(range(len(valtest_samples)))
    val_idx_local, test_idx_local = train_test_split(
        vt_idx_all, test_size=2/3, random_state=RANDOM_SEED)

    val_samples  = [valtest_samples[i] for i in val_idx_local]
    test_samples = [valtest_samples[i] for i in test_idx_local]

    print(f"  val  samples    : {len(val_samples)}개")
    print(f"  test samples    : {len(test_samples)}개")
    print(f"  sample shape: ({WINDOW_UNITS}, {n_feat})")

    # ── numpy 조립 ───────────────────────────────────────────────────
    print("\n  Numpy 조립 중...")

    def build_arrays(sample_list):
        n = len(sample_list)
        x   = np.full((n, WINDOW_UNITS, n_feat), np.nan, dtype=np.float32)
        y   = np.zeros((n,), dtype=np.int64)
        sta = np.full((n, 4), np.nan, dtype=np.float32)
        sids, smids = [], []
        for i, s in enumerate(sample_list):
            x[i]   = s["array"]
            y[i]   = s["target_label"]
            sta[i] = np.array(s["static"], dtype=np.float32)
            sids.append(s["subject_id"])
            smids.append(s["sample_id"])
        return x, y, sta, sids, smids

    train_x, train_y, train_sta, train_sids, train_smids = build_arrays(train_samples)
    val_x,   val_y,   val_sta,   val_sids,   val_smids   = build_arrays(val_samples)
    test_x,  test_y,  test_sta,  test_sids,  test_smids  = build_arrays(test_samples)

    all_subject_ids = train_sids + val_sids + test_sids
    all_sample_ids  = train_smids + val_smids + test_smids

    nan_pct = np.isnan(train_x).mean() * 100
    print(f"  train_x shape : {train_x.shape}")
    print(f"  NaN ratio (train): {nan_pct:.1f}%")

    # ── static 처리 ──────────────────────────────────────────────────
    def clean_static(sta):
        sta = fill_nan_with_col_mean(sta)
        sta[(sta[:, 1] < 0)   | (sta[:, 1] > 120), 1] = np.nan
        sta[(sta[:, 2] < 100) | (sta[:, 2] > 250), 2] = np.nan
        sta[(sta[:, 3] < 20)  | (sta[:, 3] > 300), 3] = np.nan
        return fill_nan_with_col_mean(sta)

    train_sta = clean_static(train_sta)
    val_sta   = clean_static(val_sta)
    test_sta  = clean_static(test_sta)

    # ── train 기준 정규화 ────────────────────────────────────────────
    # cache_all_new 원본 소스에 센서 오류 극단값이 섞여있음(예: light_sensor
    # max~1.8e19, 물리적으로 불가능 — sw 원본 캐시에도 동일 존재, 공유 버그).
    # clip 없이 nanstd를 구하면 sum-of-squares가 overflow해 std=inf가 되고,
    # (x-mean)/inf = 0.0 이 되어 해당 feature 전체가 결측 아닌 채로 조용히
    # 0으로 죽는다(light_sensor 전 구간이 정확히 0.0으로 확인됨). 정규화 전
    # feature별 [0.05, 99.95] percentile로 클리핑해 이상치만 제거.
    print("  train 기준 정규화 (percentile clip 후) ...")
    clip_lo = np.nanpercentile(train_x, 0.05, axis=(0, 1))
    clip_hi = np.nanpercentile(train_x, 99.95, axis=(0, 1))
    train_x = np.clip(train_x, clip_lo, clip_hi)
    val_x   = np.clip(val_x,   clip_lo, clip_hi)
    test_x  = np.clip(test_x,  clip_lo, clip_hi)

    arr_mean = np.nanmean(train_x, axis=(0, 1))
    arr_std  = np.nanstd(train_x,  axis=(0, 1)) + EPS

    train_x = (train_x - arr_mean) / arr_std
    val_x   = (val_x   - arr_mean) / arr_std
    test_x  = (test_x  - arr_mean) / arr_std

    stat_mean  = train_sta.mean(axis=0)
    stat_std   = train_sta.std(axis=0) + EPS
    train_stat = (train_sta - stat_mean) / stat_std
    val_stat   = (val_sta   - stat_mean) / stat_std
    test_stat  = (test_sta  - stat_mean) / stat_std

    # ── npz 저장 ─────────────────────────────────────────────────────
    save_path = MTM_OUTPUT_V2 / "processed_data" / f"1_{target}.npz"
    np.savez(save_path,
             train_x=train_x, train_y=train_y, train_stat=train_stat,
             val_x=val_x,     val_y=val_y,     val_stat=val_stat,
             test_x=test_x,   test_y=test_y,   test_stat=test_stat)

    # ── 메타 저장 ────────────────────────────────────────────────────
    with open(MTM_OUTPUT_V2 / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(canonical_features, f, ensure_ascii=False, indent=2)
    with open(MTM_OUTPUT_V2 / "subject_ids.json", "w", encoding="utf-8") as f:
        json.dump(all_subject_ids, f, ensure_ascii=False, indent=2)
    with open(MTM_OUTPUT_V2 / "sample_ids.json", "w", encoding="utf-8") as f:
        json.dump(all_sample_ids, f, ensure_ascii=False, indent=2)

    meta = {
        "version":           "v3 (target-specific; 시간 기반 split)",
        "data_version":      "v3",
        "target":            target,
        "target_score_column": target_score_col,
        "target_threshold":  10,
        "split_method":      "time-based per subject (train_days=9, valtest_days=5)",
        "unit_minutes":      UNIT_MINUTES,
        "window_units":      WINDOW_UNITS,
        "stride_units":      STRIDE_UNITS,
        "window_hours":      window_hours,
        "stride_hours":      stride_hours,
        "train_days":        TRAIN_DAYS,
        "valtest_days":      VALTEST_DAYS,
        "train_threshold_slot": TRAIN_THRESHOLD,
        "sample_shape":      f"({WINDOW_UNITS}, {n_feat})",
        "days_before":       DAYS_BEFORE,
        "days_after":        DAYS_AFTER,
        "n_features":        n_feat,
        "train_x_shape":     list(train_x.shape),
        "val_x_shape":       list(val_x.shape),
        "test_x_shape":      list(test_x.shape),
        "nan_ratio_train":   round(nan_pct, 1),
        "unique_patients":   len(set(all_subject_ids)),
    }
    with open(MTM_OUTPUT_V2 / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ── 결과 출력 ────────────────────────────────────────────────────
    print(f"\n  train_x : {train_x.shape}")
    print(f"  val_x   : {val_x.shape}")
    print(f"  test_x  : {test_x.shape}")
    print(f"  NaN ratio (train): {nan_pct:.1f}%")
    print(f"\n  train : {len(train_samples)} samples")
    print(f"  val   : {len(val_samples)} samples")
    print(f"  test  : {len(test_samples)} samples")

    for name, y in [("train", train_y), ("val", val_y), ("test", test_y)]:
        print(f"  {name:5s} {target.upper()}: 0={int(sum(y==0))}, 1={int(sum(y==1))}")

    print(f"\n  {target.upper()} npz : {save_path}")
    print(f"\n  완료! → {MTM_OUTPUT_V2}")


if __name__ == "__main__":
    main()
