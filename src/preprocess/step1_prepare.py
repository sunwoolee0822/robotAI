"""
step1_prepare.py
================
설문 4개 + 센서 CSV 3개 폴더 → 피험자별 DataFrame 캐시 → pickle 저장

이 스크립트가 가장 오래 걸림 (1~2시간).
한번 돌리면 step2, step3에서 pickle만 로드해서 빠르게 처리 가능.

출력:
  CACHE_DIR/
    survey_all.pkl      - 설문 통합 DataFrame
    daily_df_cache.pkl  - {subject_id: daily DataFrame}
    minute_df_cache.pkl - {subject_id: minute DataFrame}
    minute_feat_cache.pkl - {subject_id: [feature_cols]}
    feature_info.json   - canonical_features 등 메타 정보

사용법:
  Run as: `python -m src.preprocess.step1_prepare` from repo root
"""

import json
import os
import pickle
import subprocess
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..paths import DATA_ROOT

warnings.filterwarnings("ignore")


# =====================================================
# 경로 설정
# =====================================================

_RAW_SENSOR_ROOT = DATA_ROOT / "raw" / "sensors"
_LEGACY_SENSOR_ROOT = DATA_ROOT
SENSOR_FOLDERS = [
    (_RAW_SENSOR_ROOT if _RAW_SENSOR_ROOT.exists() else _LEGACY_SENSOR_ROOT) / name
    for name in [
        "data_2025_02_05", "data_2025_06_08", "data_2025_09_10",
        "data_2025_11_01", "data_2026_02_04",
    ]
]

# 새 구조(raw/survey)와 현재 데이터 링크의 기존 구조(설문데이터_수정본)를 모두 지원한다.
_RAW_SURVEY_DIR = DATA_ROOT / "raw" / "survey"
SURVEY_DIR = _RAW_SURVEY_DIR if _RAW_SURVEY_DIR.exists() else DATA_ROOT / "설문데이터_수정본"

# 캐시 통합 위치: step1이 여기에 쓰고, step2/step3 모두 여기서 읽는다
# (원본 저장소의 sw/data/cache_all 과 MTM/dataset/cache_all_new 간 drift를 제거)
CACHE_DIR = DATA_ROOT / "cache"

# ── 이상치 하드컷 (측정 데이터 이상치 리뷰 표 기준) ──────────────
HARD_CUT = {
    'hrv':              (0,    300),
    'step':             (0,    300000),
    'spo2':             (0,    100),
    'hr':               (20,   250),
    'skin_temp':        (15,   45),
    'core_temp':        (30.0, 43.0),
    'glucose':          (0,    None),
    'bp_sys':           (40,   300),
    'bp_dia':           (20,   200),
    'total_sleep_time': (0,    1440),
    'screen_time':      (0,    86400000),
}
NEG_ONLY = {'light_sensor'}
SENTINEL = {'proximity': [-1]}
# 일부 조도 CSV에서 -1이 unsigned 64-bit로 읽혀 2**64-1 근처 값이 된다.
# 음수 검사만으로는 걸러지지 않으므로 같은 sentinel을 명시적으로 차단한다.
WRAPPED_NEGATIVE_SENTINEL_MIN = 1e18
CUT_LOG  = []
BP_CUT   = [0, 0]

def apply_hard_cut(df, eng_name, value_cols):
    if (eng_name not in HARD_CUT and eng_name not in NEG_ONLY
            and eng_name not in SENTINEL):
        return df
    lo = hi = None
    if eng_name in HARD_CUT:
        lo, hi = HARD_CUT[eng_name]
    before = after = 0
    for c in value_cols:
        v = pd.to_numeric(df[c], errors='coerce')
        if v.dtype.kind in 'iub':
            v = v.astype('float64')
        before += int(v.notna().sum())
        if lo is not None:
            v[v <= lo] = np.nan
        if hi is not None:
            v[v > hi] = np.nan
        if eng_name in NEG_ONLY:
            v[(v < 0) | (v >= WRAPPED_NEGATIVE_SENTINEL_MIN)] = np.nan
        if eng_name in SENTINEL:
            v[v.isin(SENTINEL[eng_name])] = np.nan
        after += int(v.notna().sum())
        df[c] = v
        del v
    CUT_LOG.append((eng_name, before, after, before - after))
    return df


# 라벨 기준
PHQ9_THRESHOLD = 10
GAD7_THRESHOLD = 10

MISSING_SENTINEL = 255

AGG_LIST = ["mean", "min", "max"]


# =====================================================
# 매핑
# =====================================================

SENSOR_MAP = {
    "걸음수": ("step", "daily"),
    "일어난시간": ("wake_time", "daily"), "기상시간": ("wake_time", "daily"),
    "깊은수면시간": ("deep_sleep_time", "daily"),
    "램수면시간": ("rem_sleep_time", "daily"), "렘수면시간": ("rem_sleep_time", "daily"),
    "스크린타임": ("screen_time", "daily"),
    "얕은수면시간": ("light_sleep_time", "daily"),
    "이동거리": ("distance", "daily"),
    "잠든시간": ("sleep_time", "daily"),
    "총수면시간": ("total_sleep_time", "daily"),
    "EMA_Anxiety": ("EMA_Anxiety", "daily"),
    "EMA_Depression": ("EMA_Depression", "daily"),
    "EMA_Sleep": ("EMA_Sleep", "daily"),
    "EMA_Stress": ("EMA_Stress", "daily"),
    "근접센서": ("proximity", "minute"),
    "산소포화도": ("spo2", "minute"),
    "심박수": ("hr", "minute"),
    "심부체온": ("core_temp", "minute"),
    "조도센서": ("light_sensor", "minute"),
    "피부체온": ("skin_temp", "minute"),
    "호흡수": ("rr", "minute"),
    "HRV": ("hrv", "minute"),
    "혈압": ("bp", "minute"),
    "혈당": ("glucose", "minute"),
}

EMA_ENCODINGS = {
    "EMA_Anxiety": [
        '전혀 불안 하지 않음', '조금 불안함', '조금 더 불안함',
        '보통 이상 불안함', '꽤 많이 불안함', '굉장히 불안하고 참기 힘듦'
    ],
    "EMA_Depression": [
        '전혀 우울 하지 않음', '조금 우울함', '조금 더 우울함',
        '보통 이상 우울함', '꽤 많이 우울함', '굉장히 우울하고 참기 힘듦'
    ],
    "EMA_Sleep": [
        '전혀 잘 못 잠', '조금 잘 못 잠', '대체로 못 잠',
        '보통임', '대체로 잘 잠', '매우 잘 잠'
    ],
    "EMA_Stress": [
        '전혀 스트레스를 느끼지 않음', '거의 스트레스를 느끼지 않음',
        '약간 스트레스를 느낌', '스트레스를 꽤 느낌',
        '스트레스를 많이 받음', '스트레스가 매우 심함'
    ],
}

DAILY_FEATURES = [
    "step", "wake_time", "deep_sleep_time", "rem_sleep_time", "screen_time",
    "light_sleep_time", "distance", "sleep_time", "total_sleep_time",
    "EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress",
]

MINUTE_FEATURES_RAW = [
    "hrv", "proximity", "spo2", "hr", "core_temp",
    "light_sensor", "skin_temp", "rr", "bp", "glucose",
]


# =====================================================
# 유틸
# =====================================================

def normalize_text(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().replace("​", "").replace("\xa0", " ")


def safe_numeric(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def hhmmss_to_minutes(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return np.nan
    for fmt in ["%H:%M:%S", "%H:%M"]:
        t = pd.to_datetime(x, format=fmt, errors="coerce")
        if pd.notna(t):
            return float(t.hour * 60 + t.minute + t.second / 60.0)
    return np.nan


def to_naive_ts(x):
    if pd.isna(x):
        return pd.NaT
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return ts


def match_sensor_key(filename):
    filename_lower = filename.lower()
    for keyword, (eng_name, scale) in SENSOR_MAP.items():
        if keyword.lower() in filename_lower:
            return eng_name, scale
    return None, None


# =====================================================
# Step 1: 설문 통합
# =====================================================

def _scan_survey_files(survey_dir):
    """SURVEY_DIR 내 모든 CSV/xlsx 경로를 반환 (NFC 정규화)."""
    result = subprocess.run(
        ['find', str(survey_dir), '-maxdepth', '1', r'\(', '-name', '*.csv', '-o', '-name', '*.xlsx', r'\)'],
        capture_output=True, text=True
    )
    # find 명령 대안: os.scandir 로 직접 열거
    paths = []
    for entry in os.scandir(survey_dir):
        if entry.name.endswith('.csv') or entry.name.endswith('.xlsx'):
            paths.append(entry.path)
    return sorted(paths)


def _read_survey_file(filepath):
    """CSV 또는 xlsx 설문 파일을 읽어 DataFrame 반환."""
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath, engine='openpyxl')
    else:
        return pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)


def load_all_surveys():
    print("\n" + "=" * 60)
    print("  Step 1: 설문 데이터 통합")
    print("=" * 60)

    files = _scan_survey_files(SURVEY_DIR)
    print(f"  발견된 설문 파일: {len(files)}개")

    # 컬럼 이름 변형에 대응하는 매핑 (소문자 기준)
    COL_ALIASES = {
        "id":          "ID",
        "phq9_score":  "PHQ9_Score",
        "gad7_score":  "GAD7_Score",
        "sex":         "Sex",
        "age":         "Age",
        "height":      "Height",
        "weight":      "Weight",
    }
    needed = ["ID", "timestamp", "PHQ9_Score", "GAD7_Score", "Sex", "Age", "Height", "Weight"]

    all_rows = []
    for filepath in files:
        fname = os.path.basename(filepath)
        try:
            df = _read_survey_file(filepath)
        except Exception as e:
            print(f"  ⚠ {fname}: {e}")
            continue

        # 컬럼 이름을 소문자로 매핑해서 케이스 무관하게 정규화
        col_lower = {c.lower(): c for c in df.columns}
        rename_map = {}
        # timestamp: "timestamp" 포함하는 컬럼 중 가장 짧은 것
        ts_col = col_lower.get("timestamp") or next(
            (orig for lo, orig in col_lower.items() if "timestamp" in lo), None
        )
        if ts_col and ts_col != "timestamp":
            rename_map[ts_col] = "timestamp"
        # 나머지 필수 컬럼
        for lo, canonical in COL_ALIASES.items():
            if canonical not in df.columns and lo in col_lower:
                rename_map[col_lower[lo]] = canonical
        if rename_map:
            df = df.rename(columns=rename_map)

        df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")

        for col in needed:
            if col not in df.columns:
                df[col] = np.nan

        sub = df[needed].copy()
        sub["source"] = fname
        sub["PHQ9_Score"] = pd.to_numeric(sub["PHQ9_Score"], errors="coerce")
        sub["GAD7_Score"] = pd.to_numeric(sub["GAD7_Score"], errors="coerce")
        sub = sub.dropna(subset=["ID", "timestamp"])
        # 어느 target에도 쓸 수 없는 행만 제외한다. 한 점수만 있는 행은 보존하고,
        # step3_mtm.py가 --target에 맞는 유효 행만 선택한다.
        sub = sub.dropna(subset=["PHQ9_Score", "GAD7_Score"], how="all")
        all_rows.append(sub)
        print(f"  ✓ {fname}: {len(sub)}건")

    survey_all = pd.concat(all_rows, ignore_index=True)
    survey_all["date_only"] = survey_all["timestamp"].dt.date
    survey_all = survey_all.drop_duplicates(subset=["ID", "date_only"], keep="last")
    survey_all = survey_all.drop(columns=["date_only"])
    survey_all = survey_all.sort_values(["ID", "timestamp"]).reset_index(drop=True)

    survey_all["phq9_label"] = (
        (survey_all["PHQ9_Score"] >= PHQ9_THRESHOLD)
        .where(survey_all["PHQ9_Score"].notna())
        .astype("Int64")
    )
    survey_all["gad7_label"] = (
        (survey_all["GAD7_Score"] >= GAD7_THRESHOLD)
        .where(survey_all["GAD7_Score"].notna())
        .astype("Int64")
    )

    print(f"\n  통합: {len(survey_all)}건, 고유 ID: {survey_all['ID'].nunique()}명")
    print(f"  PHQ9: valid={survey_all['phq9_label'].notna().sum()}, "
          f"0={(survey_all['phq9_label']==0).sum()}, 1={(survey_all['phq9_label']==1).sum()}")
    print(f"  GAD7: valid={survey_all['gad7_label'].notna().sum()}, "
          f"0={(survey_all['gad7_label']==0).sum()}, 1={(survey_all['gad7_label']==1).sum()}")

    return survey_all


# =====================================================
# Step 2: 센서 CSV 읽기
# =====================================================

def load_all_sensors():
    print("\n" + "=" * 60)
    print("  Step 2: 센서 CSV 통합 읽기")
    print("=" * 60)

    daily_dict = {}
    minute_dict = {}

    for folder in SENSOR_FOLDERS:
        print(f"\n  폴더: {folder.name}")
        csv_files = sorted([f for f in os.listdir(folder)
                            if f.endswith(".csv") or f.endswith(".xlsx")])
        print(f"    CSV: {len(csv_files)}개")

        for filename in tqdm(csv_files, desc=f"  {folder.name}"):
            filepath = folder / filename
            eng_name, scale = match_sensor_key(filename)
            if eng_name is None:
                continue

            try:
                if filepath.suffix == '.xlsx':
                    df = pd.read_excel(filepath)
                else:
                    df = pd.read_csv(filepath, low_memory=False)
            except Exception as e:
                print(f"    ⚠ {filename}: {e}")
                continue

            date_col = df.columns[0]
            id_list = df.columns[1:]
            df = apply_hard_cut(df, eng_name, list(id_list))

            ema_key = None
            ordinal_map = None
            filename_lower = filename.lower()
            for ema_name in EMA_ENCODINGS:
                if ema_name.lower() in filename_lower:
                    ema_key = ema_name
                    ordinal_map = {normalize_text(label): score
                                for score, label in enumerate(EMA_ENCODINGS[ema_key])}
                    break

            for subject_id in id_list:
                mask = df[subject_id].notna()
                if mask.sum() == 0:
                    continue
                valid_rows = df.loc[mask, [date_col, subject_id]]

                for date_val, value in zip(valid_rows[date_col], valid_rows[subject_id]):
                    if ema_key:
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            value = float(value)
                        else:
                            value = ordinal_map.get(normalize_text(value), np.nan)
                        if pd.isna(value):
                            continue

                    if scale == "daily":
                        date_str = str(pd.to_datetime(date_val, errors="coerce")).split()[0]
                        if date_str == "NaT":
                            continue
                        daily_dict.setdefault(subject_id, {}).setdefault(date_str, {})[eng_name] = value
                    else:
                        minute_dict.setdefault(subject_id, {}).setdefault(str(date_val), {})[eng_name] = value

    print(f"\n  결과: daily {len(daily_dict)}명, minute {len(minute_dict)}명")
    return daily_dict, minute_dict


# =====================================================
# Step 3: DataFrame 캐시 생성
# =====================================================

def build_minute_df(minute_rows):
    if not minute_rows:
        return pd.DataFrame(), []

    records = []
    for dt_str, row in minute_rows.items():
        record = {"datetime": dt_str}
        for feat in MINUTE_FEATURES_RAW:
            if feat == "bp":
                bp_val = row.get("bp", np.nan)
                if pd.notna(bp_val):
                    parts = str(bp_val).split("/")
                    if len(parts) >= 2:
                        record["bp_sys"] = safe_numeric(parts[0])
                        record["bp_dia"] = safe_numeric(parts[1])
                    else:
                        record["bp_sys"] = np.nan
                        record["bp_dia"] = np.nan
                else:
                    record["bp_sys"] = np.nan
                    record["bp_dia"] = np.nan
            else:
                val = row.get(feat, np.nan)
                record[feat] = safe_numeric(val) if pd.notna(val) else np.nan
        _s = record.get("bp_sys", np.nan)
        _d = record.get("bp_dia", np.nan)
        if pd.notna(_s) or pd.notna(_d):
            BP_CUT[0] += int(pd.notna(_s)) + int(pd.notna(_d))
            if pd.notna(_s) and not (40 < _s <= 300):
                _s = np.nan
            if pd.notna(_d) and not (20 < _d <= 200):
                _d = np.nan
            if pd.notna(_s) and pd.notna(_d) and _s <= _d:
                _s = np.nan; _d = np.nan
            BP_CUT[1] += int(pd.notna(_s)) + int(pd.notna(_d))
            record["bp_sys"] = _s; record["bp_dia"] = _d
        records.append(record)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace(MISSING_SENTINEL, np.nan)

    feature_cols = [col for col in df.columns if col != "datetime" and df[col].notna().sum() > 0]
    return df, feature_cols


def build_daily_df(daily_rows):
    if not daily_rows:
        return pd.DataFrame()

    records = []
    for date_str, row in sorted(daily_rows.items()):
        record = {"date": date_str}
        for feat in DAILY_FEATURES:
            val = row.get(feat, np.nan)
            if feat in ("wake_time", "sleep_time"):
                record[feat] = val  # 원본 유지 (나중에 _parse_time_val에서 처리)
            else:
                record[feat] = safe_numeric(val)
        records.append(record)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    if "screen_time" in df.columns:
        df["screen_time"] = df["screen_time"] / 1000 / 3600
    if "distance" in df.columns:
        df["distance"] = df["distance"] / 1000
    for col in ["deep_sleep_time", "rem_sleep_time", "total_sleep_time", "light_sleep_time"]:
        if col in df.columns:
            df[col] = df[col] / 60

    def _parse_time_val(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if not x or x.lower() == "nan":
            return np.nan
        # datetime 형식 ("2026-01-22 09:16") 시도
        dt = pd.to_datetime(x, errors="coerce")
        if pd.notna(dt):
            return float(dt.hour * 60 + dt.minute + dt.second / 60.0)
        return np.nan

    if "wake_time" in df.columns:
        df["wake_time"] = df["wake_time"].apply(lambda x: _parse_time_val(x) if isinstance(x, str) else x)
    if "sleep_time" in df.columns:
        df["sleep_time"] = df["sleep_time"].apply(lambda x: _parse_time_val(x) if isinstance(x, str) else x)


    if "step" in df.columns:
        valid = df["step"].dropna()
        if len(valid) > 0 and valid.max() > valid.min():
            df["step"] = (df["step"] - valid.min()) / (valid.max() - valid.min())

    return df


def build_caches(survey_all, daily_dict, minute_dict):
    print("\n" + "=" * 60)
    print("  Step 3: DataFrame 캐시 생성")
    print("=" * 60)

    survey_ids = set(survey_all["ID"].unique())
    relevant_ids = survey_ids & (set(daily_dict.keys()) | set(minute_dict.keys()))
    print(f"  설문+센서 교집합: {len(relevant_ids)}명")

    minute_df_cache = {}
    minute_feat_cache = {}
    daily_df_cache = {}

    for sid in tqdm(relevant_ids, desc="  Converting"):
        if sid in minute_dict:
            mdf, mfeats = build_minute_df(minute_dict[sid])
            if len(mdf) > 0:
                minute_df_cache[sid] = mdf
                minute_feat_cache[sid] = mfeats

        if sid in daily_dict:
            ddf = build_daily_df(daily_dict[sid])
            if len(ddf) > 0:
                daily_df_cache[sid] = ddf

    print(f"  minute cache: {len(minute_df_cache)}명, daily cache: {len(daily_df_cache)}명")

    # feature union
    minute_union = set()
    for feats in minute_feat_cache.values():
        for c in feats:
            for agg in AGG_LIST:
                minute_union.add(f"{c}_{agg}")
    minute_union = sorted(minute_union)
    canonical_features = minute_union + DAILY_FEATURES

    print(f"  features: minute={len(minute_union)}, daily={len(DAILY_FEATURES)}, total={len(canonical_features)}")

    return minute_df_cache, minute_feat_cache, daily_df_cache, canonical_features, minute_union


# =====================================================
# 메인
# =====================================================

def main():
    print("=" * 60)
    print("  step1_prepare.py: 설문 + 센서 → 캐시 저장")
    print("=" * 60)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1
    survey_all = load_all_surveys()

    # Step 2
    daily_dict, minute_dict = load_all_sensors()

    # Step 3
    minute_df_cache, minute_feat_cache, daily_df_cache, canonical_features, minute_union = \
        build_caches(survey_all, daily_dict, minute_dict)

    # 원본 dict 삭제 (메모리 절약)
    del daily_dict, minute_dict

    # 저장
    print("\n" + "=" * 60)
    print("  캐시 저장 중...")
    print("=" * 60)

    with open(CACHE_DIR / "survey_all.pkl", "wb") as f:
        pickle.dump(survey_all, f)
    print(f"  ✓ survey_all.pkl")

    with open(CACHE_DIR / "daily_df_cache.pkl", "wb") as f:
        pickle.dump(daily_df_cache, f)
    print(f"  ✓ daily_df_cache.pkl ({len(daily_df_cache)}명)")

    with open(CACHE_DIR / "minute_df_cache.pkl", "wb") as f:
        pickle.dump(minute_df_cache, f)
    print(f"  ✓ minute_df_cache.pkl ({len(minute_df_cache)}명)")

    with open(CACHE_DIR / "minute_feat_cache.pkl", "wb") as f:
        pickle.dump(minute_feat_cache, f)
    print(f"  ✓ minute_feat_cache.pkl")

    _agg = {}
    CUT_LOG.append(('bp', BP_CUT[0], BP_CUT[1], BP_CUT[0]-BP_CUT[1]))
    for _n, _b, _a, _d in CUT_LOG:
        _x = _agg.setdefault(_n, [0, 0, 0])
        _x[0] += _b; _x[1] += _a; _x[2] += _d
    with open(CACHE_DIR / "hard_cut_log.json", "w") as f:
        json.dump({k: {"before": v[0], "after": v[1], "removed": v[2],
                       "removed_pct": round(100 * v[2] / v[0], 3) if v[0] else 0}
                   for k, v in _agg.items()}, f, indent=2, ensure_ascii=False)
    print("  ✓ hard_cut_log.json")


    feature_info = {
        "canonical_features": canonical_features,
        "minute_union": minute_union,
        "daily_features": DAILY_FEATURES,
        "agg_list": AGG_LIST,
    }
    with open(CACHE_DIR / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    print(f"  ✓ feature_info.json (total {len(canonical_features)} features)")

    print("\n" + "=" * 60)
    print("  완료!")
    print("=" * 60)
    print(f"  캐시 경로: {CACHE_DIR}")
    print(f"  다음 단계: python -m src.preprocess.step2_arrays 또는 python -m src.preprocess.step3_mtm")

if __name__ == "__main__":
    main()
