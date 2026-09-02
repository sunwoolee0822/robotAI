"""
label_compare.py
=================
환자/대조 및 인구집단별 missing rate 비교

만드는 것
  - 나이x성별 셀별 결측률 불균형 히트맵
  - 블록별 환자 vs 대조 결측률 막대그래프
  - 환자/대조 결측률 CDF
  - subject x 블록 히트맵
  - 결측 패턴만으로 나눈 군집 그림
  - 인구통계 요약 (Table 1)
  - 검정 결과 CSV/JSON

무슨 분석
  - 결측이 질병과 관련 있나
      환자군과 대조군의 결측률이 블록 단위로 다른지 검정.
      다르면 결측이 잡음이 아니라 신호라는 뜻.
  - 결측이 인구집단별로 다른가
      나이/성별 조합마다 결측률이 치우쳐 있는지.
      치우쳐 있으면 증강을 그룹별로 나눠야 할 근거가 됨.
  - 시간대에 따라 다른가
      낮/밤 결측 차이가 라벨에 따라 다른지.
  - 결측만으로 환자를 가를 수 있나
      라벨 없이 결측 패턴만 군집화했을 때 PHQ9 그룹과 겹치는지.

조건
  - 학습 결과 없어도 됨
  - 검정은 Mann-Whitney + BH-FDR 다중검정 보정
  - 표본 부족한 셀은 따로 표시됨
  - 증강 데이터에도 그대로 돌릴 수 있음 -> 실험2 증강 전후 비교 가능
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .. import paths
# BLOCKS/FEATURE_PERIODS_HOURS/compute_sample_missrate의 단일 소스는 grouping.py
# (재정의 금지, 모듈 docstring 참고). DEVICE_GROUPS는 profile.py 소유.
# 예전에는 이 셋+DEVICE_GROUPS+conditional_missing_matrix/plot_cond_heatmap/
# plot_corr_heatmap/plot_dendrogram을 전부 missing_rate에서 import했으나,
# missing_rate.py에는 spearman_matrix만 실제로 있고 나머지는 이관되지 않아
# (BLOCKS 포함) 모듈 import 자체가 항상 실패했다(작업 0 조사 중 발견).
# conditional_missing_matrix/plot_cond_heatmap/plot_corr_heatmap/plot_dendrogram은
# 여전히 미이관 상태 — 이번 작업 범위 밖이라 손대지 않고, 실제로 쓰는
# generate_missing_pattern_by_label() 안으로 import를 옮겨 그 함수를 호출할
# 때만 실패하도록 격리한다(모듈 전체가 죽는 것 방지).
from ..augment.grouping import BLOCKS, FEATURE_PERIODS_HOURS, compute_sample_missrate
from ..augment.profile import DEVICE_GROUPS
from .missing_rate import spearman_matrix

LABEL_NAMES = {0: "Control", 1: "Patient"}
LABEL_COLORS = {0: "#5B8DB8", 1: "#C0504D"}


# =====================================================================
# Shared demographic lookup (imbalance_by_subgroup.py: build_demo_lookup, AGE_LABELS)
# =====================================================================

AGE_BINS = [0, 30, 40, 50, 60, 150]
AGE_LABELS = ["<30s", "30s", "40s", "50s", "60s+"]


def encode_sex(x):
    """Handles inconsistent per-wave sex encodings (matches step3_mtm_v2.py's fix)."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x in ("남", "1"):
        return "Male"
    if x in ("여", "0", "2"):
        return "Female"
    return np.nan


def cliffs_delta_from_u(U, n1, n2):
    return 2.0 * U / (n1 * n2) - 1.0


def build_demo_lookup(cache_dir=None) -> pd.DataFrame:
    """subject_id -> (Sex, Age, AgeGroup) lookup, from survey_all.pkl.

    "age_years" 컬럼을 직접 참조했었는데 survey_all.pkl에 그 컬럼이 없고(작업 3
    스모크테스트 중 발견), 실제 "Age" 컬럼은 정제 전 원본이라 문자열 생년월일/
    Timestamp/datetime이 섞여 있어 pd.cut이 바로 못 씀. step3_mtm.py::get_age가
    이미 이 정확한 fallback 로직(age_years 우선, 없으면 Age를 생년월일로 파싱해
    오늘 기준 나이 계산)을 갖고 있어 그걸 재사용 — age 계산 로직이 preprocess/analysis
    두 곳에 따로 존재하면 안 됨."""
    from ..preprocess.step3_mtm import get_age

    cache_dir = Path(cache_dir) if cache_dir else paths.DATA_ROOT / "cache"
    survey_all = pickle.load(open(cache_dir / "survey_all.pkl", "rb"))
    first = survey_all.drop_duplicates("ID", keep="first").set_index("ID")
    demo = pd.DataFrame({"Sex": first["Sex"].apply(encode_sex),
                         "Age": first.apply(get_age, axis=1)})
    demo["AgeGroup"] = pd.cut(demo["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return demo


# =====================================================================
# Table 1 — basic demographics (analysis/v2_72/analysis0_demo.py)
# =====================================================================

def demographics_summary(data_root=None, sample_dir=None, out_dir=None) -> dict:
    """Subject-level sex/age + PHQ9/GAD7 window-level summary ("Table 1").

    data_root: folder with static.npy (cols: sex, age, height, weight) + subject_ids.json
    sample_dir: folder of per-sample metadata.json (phq9_label/score, gad7_label/score);
                if omitted, the PHQ9/GAD7 section is skipped.
    """
    data_root = Path(data_root) if data_root else paths.DATA_ROOT / "numpy_all_chunk_72_24feat"
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "label_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    static = np.load(data_root / "static.npy")
    sids = [str(x) for x in json.load(open(data_root / "subject_ids.json"))]
    sdf = pd.DataFrame({"subject_id": sids, "sex": static[:, 0], "age": static[:, 1]})
    sdf = sdf.drop_duplicates("subject_id")

    summary = {
        "n_subjects_static": len(sdf),
        "sex_counts": sdf["sex"].value_counts().to_dict(),
        "age_mean": round(float(sdf["age"].mean()), 1),
        "age_median": round(float(sdf["age"].median()), 1),
        "age_sd": round(float(sdf["age"].std()), 1),
        "age_band_counts": pd.cut(sdf["age"], bins=[0, 30, 45, 60, 200],
                                   labels=["<30", "30-45", "45-60", "60+"],
                                   right=False).value_counts().sort_index().to_dict(),
    }

    if sample_dir is not None:
        rows = []
        for sd in sorted(p for p in Path(sample_dir).iterdir() if p.is_dir()):
            mp = sd / "metadata.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text(encoding="utf-8"))
            rows.append({"subject_id": m.get("subject_id"), "sample_id": m.get("sample_id"),
                         "phq9_label": m.get("phq9_label"), "phq9_score": m.get("phq9_score"),
                         "gad7_label": m.get("gad7_label"), "gad7_score": m.get("gad7_score")})
        df = pd.DataFrame(rows)
        n_survey = len(df)
        n_dep = int((df["phq9_label"] == 1).sum())
        n_anx = int((df["gad7_label"] == 1).sum())
        summary.update({
            "n_windows": n_survey, "n_subjects": int(df["subject_id"].nunique()),
            "phq9_depressed": n_dep, "phq9_pct": round(100 * n_dep / max(n_survey, 1), 1),
            "phq9_score_mean": round(float(df["phq9_score"].mean()), 1),
            "gad7_anxious": n_anx, "gad7_pct": round(100 * n_anx / max(n_survey, 1), 1),
        })

    (out_dir / "demographics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"  saved: {out_dir / 'demographics_summary.json'}")
    return summary


# =====================================================================
# CDF + subject x feature-group heatmap (analysis_demographic_missing.py, Step 1-2)
# =====================================================================

_FEATURE_GROUPS_1H_DAILY_WEEKLY = {
    "1h": ["hr", "rr", "core_temp", "skin_temp", "bp_sys", "bp_dia", "glucose"],
    "daily": ["spo2", "hrv", "light_sensor", "proximity", "step", "distance", "screen_time",
              "wake_time", "sleep_time", "deep_sleep_time", "rem_sleep_time",
              "light_sleep_time", "total_sleep_time"],
    "weekly": ["EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress"],
}
_GROUP_COLORS_1H_DAILY_WEEKLY = {"1h": "#4C72B0", "daily": "#DD8452", "weekly": "#55A868"}


def load_and_aggregate(data_dir: Path, split: str = "1"):
    """Load npz -> subject-level records (each subject = 1 observation; avoids
    inflating significance from within-subject correlated windows)."""
    data_dir = Path(data_dir)
    data = np.load(data_dir / "processed_data" / f"{split}.npz", allow_pickle=True)
    feat_cols = json.load(open(data_dir / "feature_columns.json"))
    subject_ids = json.load(open(data_dir / "subject_ids.json"))

    X = np.concatenate([data["train_x"], data["val_x"], data["test_x"]], axis=0)
    Y = np.concatenate([data["train_y"], data["val_y"], data["test_y"]], axis=0)
    sub_ids = np.array(subject_ids)

    subject_records = []
    for sid in np.unique(sub_ids):
        mask = (sub_ids == sid)
        x_sub, y_sub = X[mask], Y[mask]
        label = int(y_sub[0])
        nan_mask = np.isnan(x_sub)
        subject_records.append({
            "subject_id": sid, "label": label,
            "feat_missing": nan_mask.mean(axis=(0, 1)),
            "time_missing": nan_mask.mean(axis=(0, 2)),
            "n_samples": int(mask.sum()),
        })
    return subject_records, feat_cols


def split_by_label(subject_records):
    ctrl = [s for s in subject_records if s["label"] == 0]
    pat = [s for s in subject_records if s["label"] == 1]
    return ctrl, pat


def _get_group_missing(subject_records, feat_cols, group_feats):
    idxs = [feat_cols.index(f) for f in group_feats if f in feat_cols]
    return np.array([s["feat_missing"][idxs].mean() for s in subject_records])


def plot_cdf(ctrl, pat, feat_cols, out_dir: Path):
    """CDF (not boxplot) of missing rate per feature-group: preserves distribution
    shape, robust to missing-rate values piling up at 0/1 or being bimodal."""
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (gname, gfeats) in zip(axes, _FEATURE_GROUPS_1H_DAILY_WEEKLY.items()):
        c_vals = np.sort(_get_group_missing(ctrl, feat_cols, gfeats))
        p_vals = np.sort(_get_group_missing(pat, feat_cols, gfeats))
        ax.plot(c_vals, np.linspace(0, 1, len(c_vals)), label="Control", color=LABEL_COLORS[0])
        ax.plot(p_vals, np.linspace(0, 1, len(p_vals)), label="Patient", color=LABEL_COLORS[1])
        ax.set_title(f"{gname} feature group", fontsize=12, weight="bold")
        ax.set_xlabel("Missing rate"); ax.set_ylabel("CDF")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Missing-rate CDF by feature group: Patient vs Control (subject-level)", weight="bold")
    plt.tight_layout()
    out = out_dir / "cdf_missing_by_group.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")


def plot_subject_heatmap(ctrl, pat, feat_cols, out_dir: Path, max_subjects=150):
    """Subject x feature-group missing-rate heatmap, Control block then Patient block,
    for eyeballing clustered patterns."""
    out_dir = Path(out_dir)
    rows, group_labels = [], []
    for group in (ctrl, pat):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(group), min(max_subjects, len(group)), replace=False) if len(group) > max_subjects \
            else np.arange(len(group))
        for i in idx:
            s = group[i]
            row = {gname: s["feat_missing"][[feat_cols.index(f) for f in gfeats if f in feat_cols]].mean()
                   for gname, gfeats in _FEATURE_GROUPS_1H_DAILY_WEEKLY.items()}
            rows.append(row)
            group_labels.append(s["label"])
    mat = pd.DataFrame(rows).values
    fig, ax = plt.subplots(figsize=(6, max(8, len(rows) * 0.05)))
    sns.heatmap(mat * 100, ax=ax, cmap="YlOrRd_r", vmin=0, vmax=100, cbar_kws={"label": "Missing rate (%)"},
                xticklabels=list(_FEATURE_GROUPS_1H_DAILY_WEEKLY.keys()), yticklabels=False)
    split_at = sum(1 for g in group_labels if g == 0)
    ax.axhline(split_at, color="black", linewidth=2)
    ax.set_ylabel(f"Subjects (Control top {split_at}, Patient bottom {len(group_labels) - split_at})")
    ax.set_title("Subject x feature-group missing rate", weight="bold")
    plt.tight_layout()
    out = out_dir / "subject_feature_group_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")


# =====================================================================
# Layer 2 — block-level Patient vs Control missing rate
#   (MTM/analysis/missing_pattern_groups.py Layer 2)
# =====================================================================

def block_group_test(M, feats, y, subject_ids):
    """Subject-level mean missing rate per block -> Mann-Whitney U + BH-FDR + effect size."""
    fidx = {f: i for i, f in enumerate(feats)}
    block_names, block_sample = [], []
    for bname, bfeats in BLOCKS.items():
        cols = [fidx[f] for f in bfeats if f in fidx]
        if not cols:
            continue
        block_names.append(bname)
        block_sample.append(M[:, cols].mean(axis=1))
    block_sample = np.array(block_sample).T

    df = pd.DataFrame(block_sample, columns=block_names)
    df["subject"] = subject_ids
    df["y"] = y
    subj = df.groupby("subject").agg({**{b: "mean" for b in block_names}, "y": "mean"})
    subj["group"] = (subj["y"] >= 0.5).astype(int)

    rows, pvals = [], []
    for b in block_names:
        g0 = subj.loc[subj.group == 0, b].values
        g1 = subj.loc[subj.group == 1, b].values
        U, p = mannwhitneyu(g1, g0, alternative="two-sided")
        delta = cliffs_delta_from_u(U, len(g1), len(g0))
        rows.append({"block": b, "control_missrate": g0.mean(), "patient_missrate": g1.mean(),
                     "control_sem": g0.std(ddof=1) / np.sqrt(len(g0)),
                     "patient_sem": g1.std(ddof=1) / np.sqrt(len(g1)),
                     "diff(pat-ctrl)": g1.mean() - g0.mean(), "U": U, "p_raw": p,
                     "cliffs_delta(pat-ctrl)": delta, "n_ctrl": len(g0), "n_pat": len(g1)})
        pvals.append(p)

    pvals = np.array(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    p_adj = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        val = min(prev, pvals[i] * m / k)
        p_adj[i] = val
        prev = val
    for r, pa in zip(rows, p_adj):
        r["p_BH"] = pa
        r["sig(BH<0.05)"] = pa < 0.05

    return pd.DataFrame(rows), subj, block_names


def plot_block_bar(res, out_path):
    n = len(res)
    x = np.arange(n)
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, n * 1.3), 6))
    ax.bar(x - w / 2, res["control_missrate"] * 100, w, label="Control (PHQ9<10)", color="#3498db",
           edgecolor="gray", yerr=res["control_sem"] * 100, capsize=4,
           error_kw={"elinewidth": 1.5, "ecolor": "#1a5276"})
    ax.bar(x + w / 2, res["patient_missrate"] * 100, w, label="Patient (PHQ9>=10)", color="#e74c3c",
           edgecolor="gray", yerr=res["patient_sem"] * 100, capsize=4,
           error_kw={"elinewidth": 1.5, "ecolor": "#922b21"})
    ymax = max(res["control_missrate"].max(), res["patient_missrate"].max()) * 100
    for i, row in res.iterrows():
        if row["sig(BH<0.05)"]:
            yy = max(row["control_missrate"], row["patient_missrate"]) * 100
            ax.text(i, yy + 1.5, f"*\nd={row['cliffs_delta(pat-ctrl)']:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(res["block"], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Missing rate (%)  [subject-level mean]", fontsize=12)
    ax.set_ylim(0, min(105, ymax + 12))
    ax.set_title("Block (collection-source) missing rate: Control vs Patient\n"
                 "* = BH-FDR p<0.05 ;  d = Cliff's delta (effect size)", fontsize=13)
    ax.legend(fontsize=11); ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


def plot_missing_vs_label_block(subj_block_missing: pd.DataFrame, out_path,
                                 blocks=("A", "B1", "B2", "C", "D", "E", "F")):
    """Polished variant of the block-bar figure driven off a precomputed per-subject
    per-block missing-rate table (as produced by missing_rate.generate_hourly_missing_rate_figures
    + a block mapping), instead of a raw npz load. Columns required: subject_id, block, y.
    Ported from analysis/v2_72/analysis2_label.py.
    """
    def star(p):
        return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

    rows, fig_data = [], []
    for b in blocks:
        d = subj_block_missing[subj_block_missing["block"] == b]
        a = d.loc[d["y"] == 0, "missing_rate"].values
        c = d.loc[d["y"] == 1, "missing_rate"].values
        if len(a) < 2 or len(c) < 2:
            continue
        u, p = mannwhitneyu(c, a, alternative="two-sided")
        rows.append({"block": b, "n_control": len(a), "n_patient": len(c),
                     "mean_control": a.mean(), "mean_patient": c.mean(), "U": u, "p": p, "sig": star(p)})
        fig_data.append((b, a, c, star(p)))
    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(fig_data)); w = 0.34
    for i, (b, a, c, s) in enumerate(fig_data):
        ax.bar(x[i] - w / 2 - .01, a.mean(), w, yerr=a.std(ddof=1) / np.sqrt(len(a)), capsize=4,
               color="#5dade2", alpha=.5, edgecolor="#5dade2", hatch="//", linewidth=1.2)
        ax.bar(x[i] + w / 2 + .01, c.mean(), w, yerr=c.std(ddof=1) / np.sqrt(len(c)), capsize=4,
               color="#5dade2", alpha=1.0, edgecolor="#5dade2", linewidth=1.2)
        top = max(a.mean() + a.std(ddof=1) / np.sqrt(len(a)), c.mean() + c.std(ddof=1) / np.sqrt(len(c)))
        ax.text(x[i], top + .035, s, ha="center", fontsize=13, weight="bold" if s != "n.s." else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels([b for b, *_ in fig_data])
    ax.set_ylabel("Missing rate (mean ± SEM)")
    ax.set_title("Missing rate by feature block — patient vs control (Mann-Whitney U, two-sided)", weight="bold")
    ax.grid(alpha=.25, axis="y"); ax.set_axisbelow(True)
    handles = [mpatches.Patch(facecolor="#b0b0b0", edgecolor="#555", hatch="//", alpha=.55, label="Control"),
               mpatches.Patch(facecolor="#5dade2", edgecolor="#555", label="Patient")]
    ax.legend(handles=handles, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")
    return tab


# =====================================================================
# Layer 3 — diurnal + day/night pattern by label
#   (MTM/analysis/missing_pattern_groups.py Layer 3)
# =====================================================================

_HIGHFREQ_FEATS = ["hr", "rr", "core_temp", "skin_temp", "bp_sys", "bp_dia", "glucose"]


def _hour_missrate(mask_c, hours, idx):
    h = hours[idx].ravel()
    m = mask_c[idx].ravel().astype(np.float64)
    cnt = np.bincount(h, minlength=24)
    tot = np.bincount(h, weights=m, minlength=24)
    return tot / np.maximum(cnt, 1)


def plot_diurnal_by_label(mask, feats, y, hours, out_path):
    """Hour-of-day missing rate for high-frequency sensors, Control vs Patient."""
    fidx = {f: i for i, f in enumerate(feats)}
    cols = [fidx[f] for f in _HIGHFREQ_FEATS if f in fidx]
    sub = mask[:, :, cols].mean(axis=2)
    r0 = _hour_missrate(sub, hours, y == 0)
    r1 = _hour_missrate(sub, hours, y == 1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(range(24), r0 * 100, "-o", color=LABEL_COLORS[0], label="Control")
    ax.plot(range(24), r1 * 100, "-o", color=LABEL_COLORS[1], label="Patient")
    ax.axvspan(0, 6, color="navy", alpha=0.06); ax.axvspan(22, 23, color="navy", alpha=0.06)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (0-23)"); ax.set_ylabel("Missing rate (%)")
    ax.set_title("Diurnal missingness — high-frequency sensors (hr/rr/temp/bp/glucose)\nshaded = night")
    ax.legend(); ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


def daynight_test(mask, feats, y, hours):
    """Sample-level Mann-Whitney of day (7-21h) vs night (22-6h) missing rate, by label."""
    fidx = {f: i for i, f in enumerate(feats)}
    cols = [fidx[f] for f in _HIGHFREQ_FEATS if f in fidx]
    sub = mask[:, :, cols].mean(axis=2)
    night = (hours >= 22) | (hours <= 6)
    rows = []
    for label, sel in [("night(22-6)", night), ("day(7-21)", ~night)]:
        num = (sub * sel).sum(axis=1)
        den = sel.sum(axis=1)
        rate = num / np.maximum(den, 1)
        g0, g1 = rate[y == 0], rate[y == 1]
        U, p = mannwhitneyu(g1, g0, alternative="two-sided")
        rows.append({"period": label, "control": g0.mean(), "patient": g1.mean(),
                     "diff": g1.mean() - g0.mean(), "cliffs_delta": cliffs_delta_from_u(U, len(g1), len(g0)),
                     "p_raw": p})
    return pd.DataFrame(rows)


def generate_missing_pattern_by_label(data_dir, out_dir=None, split=1, unit_minutes=60):
    """Driver for Layer 1 (co-missing structure, control/patient/diff), Layer 2
    (block bar), and Layer 3 (diurnal + day/night) — the full 3-layer report from
    MTM/analysis/missing_pattern_groups.py."""
    # conditional_missing_matrix/plot_cond_heatmap/plot_corr_heatmap/plot_dendrogram는
    # missing_rate.py에 이관되지 않은 상태(작업 0 조사 결과) — 이 함수를 실제로 부를 때만
    # 실패하도록 로컬 import로 격리.
    from .missing_rate import conditional_missing_matrix, plot_cond_heatmap, plot_corr_heatmap, plot_dendrogram

    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "label_compare" / data_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(data_dir / "processed_data" / f"{split}.npz", allow_pickle=True)
    x = np.concatenate([d["train_x"], d["val_x"], d["test_x"]], axis=0)
    y = np.concatenate([d["train_y"], d["val_y"], d["test_y"]], axis=0)
    feats = json.load(open(data_dir / "feature_columns.json", encoding="utf-8"))
    sample_ids = json.load(open(data_dir / "sample_ids.json", encoding="utf-8"))
    subject_ids = json.load(open(data_dir / "subject_ids.json", encoding="utf-8"))

    mask = np.isnan(x)
    del x
    feat_period = {f: FEATURE_PERIODS_HOURS.get(f, 1) for f in feats}
    ordered = [f for bf in BLOCKS.values() for f in bf if f in feats]
    ordered += [f for f in feats if f not in ordered]
    ord_idx = [feats.index(f) for f in ordered]

    print("[Layer 1] co-missingness structure")
    M = compute_sample_missrate(mask, feats, unit_minutes)
    M_ord = M[:, ord_idx]
    R_all, R_ctrl, R_pat = spearman_matrix(M_ord), spearman_matrix(M_ord[y == 0]), spearman_matrix(M_ord[y == 1])
    plot_corr_heatmap(R_all, ordered, "Missingness Spearman corr — ALL", out_dir / "L1_corr_all.png")
    plot_corr_heatmap(R_ctrl, ordered, "Missingness Spearman corr — CONTROL", out_dir / "L1_corr_control.png")
    plot_corr_heatmap(R_pat, ordered, "Missingness Spearman corr — PATIENT", out_dir / "L1_corr_patient.png")
    plot_corr_heatmap(R_pat - R_ctrl, ordered, "Missingness corr DIFF (Patient - Control)",
                       out_dir / "L1_corr_diff.png")
    P_cond, _ = conditional_missing_matrix(M_ord)
    plot_cond_heatmap(P_cond, ordered, "Conditional missingness P(col|row) — ALL", out_dir / "L1_conditional.png")
    plot_dendrogram(R_all, ordered, feat_period, out_dir / "L1_dendrogram.png")

    print("[Layer 2] block-level group comparison")
    res, subj, block_names = block_group_test(M, feats, y, np.array(subject_ids))
    res.to_csv(out_dir / "L2_block_group_test.csv", index=False)
    plot_block_bar(res, out_dir / "L2_block_bar.png")

    print("[Layer 3] hour-of-day missingness")
    w_start = np.array([int(s.rsplit("_s", 1)[1]) for s in sample_ids], dtype=np.int64)
    T = mask.shape[1]
    hours = (w_start[:, None] + np.arange(T)[None, :]) % 24
    plot_diurnal_by_label(mask, feats, y, hours, out_dir / "L3_diurnal_highfreq.png")
    dn = daynight_test(mask, feats, y, hours)
    dn.to_csv(out_dir / "L3_daynight_test.csv", index=False)

    print(f"Done -> {out_dir}")
    return res, dn


# =====================================================================
# Cell-level (Sex x AgeGroup) imbalance heatmap
#   (MTM/analysis/imbalance_by_subgroup.py + missing_pattern_report.py, merged)
# =====================================================================

MIN_N = 5  # underpowered threshold


def mannwhitney_z_r(p_vals, c_vals, U):
    """Normal-approximation z (tie-corrected) and effect size r = |z|/sqrt(N)."""
    n1, n2 = len(p_vals), len(c_vals)
    N = n1 + n2
    allv = np.concatenate([np.asarray(p_vals), np.asarray(c_vals)])
    _, counts = np.unique(allv, return_counts=True)
    tie_term = (counts ** 3 - counts).sum()
    mu_U = n1 * n2 / 2.0
    var_U = (n1 * n2 / 12.0) * ((N + 1) - tie_term / (N * (N - 1))) if N > 1 else 0.0
    sigma_U = np.sqrt(var_U) if var_U > 0 else 0.0
    z = (U - mu_U) / sigma_U if sigma_U > 0 else 0.0
    r = abs(z) / np.sqrt(N) if N > 0 else 0.0
    return z, r


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if len(idx) == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = p[i] * m / (rank + 1)
        prev = min(prev, val)
        q[i] = prev
    return q


def compute_cell_imbalance(data_dir, cache_dir=None) -> pd.DataFrame:
    """Per (Sex, AgeGroup) cell: Patient vs Control missing-rate Mann-Whitney U +
    Cliff's delta + BH-FDR, at the feature / block / device axes. Cells with either
    group's N < MIN_N are flagged 'underpowered' (NaN stats)."""
    data_dir = Path(data_dir)
    d = np.load(data_dir / "processed_data" / "1.npz", allow_pickle=True)
    feats = json.load(open(data_dir / "feature_columns.json"))
    subject_ids = np.array(json.load(open(data_dir / "subject_ids.json")))
    X = np.concatenate([d["train_x"], d["val_x"], d["test_x"]], axis=0)
    Y = np.concatenate([d["train_y"], d["val_y"], d["test_y"]], axis=0)
    unit_minutes = 60
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        unit_minutes = json.loads(meta_path.read_text()).get("unit_minutes", 60)

    M = compute_sample_missrate(np.isnan(X), feats, unit_minutes)
    df = pd.DataFrame(M, columns=feats)
    for bn, bfeats in BLOCKS.items():
        cols = [f for f in bfeats if f in feats]
        df[f"block_{bn}"] = df[cols].mean(axis=1)
    for dv, dfeats in DEVICE_GROUPS.items():
        cols = [f for f in dfeats if f in feats]
        df[f"device_{dv}"] = df[cols].mean(axis=1)
    df["subject_id"] = subject_ids
    df["phq9_label"] = Y

    subj = df.groupby("subject_id").mean(numeric_only=True)
    subj["phq9_label"] = (subj["phq9_label"] >= 0.5).astype(int)
    demo = build_demo_lookup(cache_dir)
    subj = subj.join(demo, how="left").dropna(subset=["Sex", "AgeGroup"])

    axes = ([("feature", f) for f in feats] + [("block", f"block_{bn}") for bn in BLOCKS] +
            [("device", f"device_{dv}") for dv in DEVICE_GROUPS])

    rows = []
    for sex in ["Male", "Female"]:
        for ageg in AGE_LABELS:
            cell = subj[(subj["Sex"] == sex) & (subj["AgeGroup"] == ageg)]
            pat, ctl = cell[cell["phq9_label"] == 1], cell[cell["phq9_label"] == 0]
            for kind, col in axes:
                p_vals, c_vals = pat[col].dropna(), ctl[col].dropna()
                name = col.replace("block_", "").replace("device_", "")
                base = {"sex": sex, "age_group": ageg, "cell": f"{sex}_{ageg}", "kind": kind, "name": name,
                        "n_patient": len(p_vals), "n_control": len(c_vals)}
                if len(p_vals) < MIN_N or len(c_vals) < MIN_N:
                    rows.append({**base, "patient_mean": np.nan, "control_mean": np.nan, "diff": np.nan,
                                 "cliffs_delta": np.nan, "U": np.nan, "z_value": np.nan, "effect_r": np.nan,
                                 "p_value": np.nan, "q_value": np.nan, "significant": False,
                                 "significant_fdr": False, "underpowered": True})
                    continue
                U, p = mannwhitneyu(p_vals, c_vals, alternative="two-sided")
                z, r = mannwhitney_z_r(p_vals, c_vals, U)
                delta = cliffs_delta_from_u(U, len(p_vals), len(c_vals))
                rows.append({**base, "patient_mean": p_vals.mean(), "control_mean": c_vals.mean(),
                             "diff": p_vals.mean() - c_vals.mean(), "cliffs_delta": delta, "U": float(U),
                             "z_value": z, "effect_r": r, "p_value": p, "q_value": np.nan,
                             "significant": bool(p < 0.05), "significant_fdr": False, "underpowered": False})
    res = pd.DataFrame(rows)
    res["q_value"] = bh_fdr(res["p_value"].values)
    res["significant_fdr"] = res["q_value"] < 0.05
    return res


def plot_cell_imbalance_heatmap(res: pd.DataFrame, col_names: list, title: str, out_path):
    """Sex-split (Male | Female) x AgeGroup heatmap of Cliff's delta, with
    underpowered cells hatched and non-significant cells white-washed."""
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(col_names) * 1.5) * 2, 5.6))
    for ax, sex in zip(axes, ["Male", "Female"]):
        rows = AGE_LABELS
        mat = np.full((len(rows), len(col_names)), np.nan)
        annot = np.full((len(rows), len(col_names)), "", dtype=object)
        under = np.zeros((len(rows), len(col_names)), dtype=bool)
        nonsig = np.zeros((len(rows), len(col_names)), dtype=bool)
        for i, ageg in enumerate(rows):
            cell = f"{sex}_{ageg}"
            for j, cn in enumerate(col_names):
                r = res[(res.cell == cell) & (res.name == cn)]
                if len(r) == 0:
                    continue
                row = r.iloc[0]
                nP, nC = int(row["n_patient"]), int(row["n_control"])
                if row["underpowered"]:
                    under[i, j] = True
                    annot[i, j] = f"n={nP}/{nC}"
                    continue
                mat[i, j] = row["cliffs_delta"]
                star = "**" if row["significant_fdr"] else ("*" if row["significant"] else "")
                annot[i, j] = f"{row['cliffs_delta']:+.2f}{star}\nn={nP}/{nC}"
                nonsig[i, j] = not row["significant"]

        sns.heatmap(mat, ax=ax, xticklabels=col_names, yticklabels=rows, cmap="RdBu_r", vmin=-1, vmax=1,
                    center=0, annot=annot, fmt="", annot_kws={"size": 7.5},
                    cbar_kws={"label": "Cliff's delta (Patient - Control)"}, linewidths=0.5, linecolor="white")
        for i in range(len(rows)):
            for j in range(len(col_names)):
                if under[i, j]:
                    ax.add_patch(mpatches.Rectangle((j, i), 1, 1, fill=True, facecolor="#d0d0d0",
                                                     hatch="//", edgecolor="white", lw=0.5, zorder=3))
                elif nonsig[i, j]:
                    ax.add_patch(mpatches.Rectangle((j, i), 1, 1, fill=True, facecolor="white",
                                                     alpha=0.55, edgecolor="none", zorder=3))
        ax.set_title(sex, fontsize=12, weight="bold")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(title + "\nRED=Patient missing MORE, BLUE=LESS | *p<.05 **q<.05(FDR) | "
                        "white-washed=not significant | hatched=underpowered(n<5)", fontsize=11.5, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


def generate_cell_imbalance_report(data_dir, out_dir=None, label="high", cache_dir=None):
    """Full report: pc_table.csv (all cells x axes x stats), device/block heatmaps,
    and a per-cell summary CSV. Re-runnable on augmented data via data_dir/label to
    verify that group-aware augmentation reproduced the target imbalance structure.
    """
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "label_compare" / "cell_imbalance"
    out_dir = out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    res = compute_cell_imbalance(data_dir, cache_dir=cache_dir)
    res.to_csv(out_dir / "pc_table.csv", index=False)
    n_p_sig, n_q_sig = int(res["significant"].sum()), int(res["significant_fdr"].sum())
    print(f"  p<.05: {n_p_sig} -> BH-FDR q<.05 survive: {n_q_sig}")

    plot_cell_imbalance_heatmap(res[res.kind == "device"], list(DEVICE_GROUPS.keys()),
                                 f"Patient vs Control missing imbalance in {label.upper()}  |  cell x DEVICE",
                                 out_dir / "pc_device_heatmap.png")
    plot_cell_imbalance_heatmap(res[res.kind == "block"], list(BLOCKS.keys()),
                                 f"Patient vs Control missing imbalance in {label.upper()}  |  cell x BLOCK",
                                 out_dir / "pc_block_heatmap.png")

    cells = [f"{s}_{a}" for s in ["Male", "Female"] for a in AGE_LABELS]
    score_rows = []
    for cell in cells:
        r = res[(res.cell == cell) & (res.kind.isin(["device", "block"]))]
        rc = res[res.cell == cell]
        if len(rc) == 0:
            continue
        rc = rc.iloc[0]
        powered = r[~r.underpowered]
        score_rows.append({
            "cell": cell, "n_patient": int(rc["n_patient"]), "n_control": int(rc["n_control"]),
            "powered": bool(len(powered) > 0),
            "n_sig_device": int(r[(r.kind == "device") & r.significant].shape[0]),
            "n_sig_block": int(r[(r.kind == "block") & r.significant].shape[0]),
            "mean_abs_cliffs_delta": float(powered["cliffs_delta"].abs().mean()) if len(powered) else np.nan,
        })
    score = pd.DataFrame(score_rows)
    score.to_csv(out_dir / "pc_cell_score.csv", index=False)
    print(f"  saved: {out_dir / 'pc_cell_score.csv'}")
    return res, score


# =====================================================================
# Unsupervised structure check: does missingness alone separate Patient/Control?
#   (MTM/analysis/missing_unsupervised.py)
# =====================================================================

def load_phq9_continuous(cache_dir=None) -> pd.DataFrame:
    cache_dir = Path(cache_dir) if cache_dir else paths.DATA_ROOT / "cache"
    survey_all = pickle.load(open(cache_dir / "survey_all.pkl", "rb"))
    first = survey_all.drop_duplicates("ID", keep="first").set_index("ID")
    col = "PHQ9" if "PHQ9" in first.columns else next(
        (c for c in first.columns if c.upper().startswith("PHQ9")), None)
    if col is None:
        return pd.DataFrame({"subject": first.index, "phq9_mean": np.nan})
    return pd.DataFrame({"subject": first.index, "phq9_mean": first[col]}).reset_index(drop=True)


def patient_level(M, y, subject_ids, phq_df):
    """Sample-level missing rate matrix M (N,C) -> patient-level mean + PHQ9 join."""
    df = pd.DataFrame(M, columns=[f"f{i}" for i in range(M.shape[1])])
    df["subject"] = subject_ids
    df["y"] = y
    grp = df.groupby("subject").agg({**{f"f{i}": "mean" for i in range(M.shape[1])}, "y": "mean"}).reset_index()
    grp["phq9_binary"] = (grp["y"] >= 0.5).astype(int)
    grp = grp.merge(phq_df, on="subject", how="left")
    feat_cols = [f"f{i}" for i in range(M.shape[1])]
    return grp[feat_cols].values.astype(np.float32), grp


def plot_pca(X_scaled, grp, feats, out_dir):
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    phq_vals = grp["phq9_mean"].fillna(grp["y"] * 27).values
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=phq_vals, cmap="RdYlBu_r", alpha=0.6, s=18, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="PHQ9 score (patient mean)")
    ax.set_title(f"PCA — colored by continuous PHQ9\nPC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")

    ax = axes[1]
    colors = {0: "#3498db", 1: "#e74c3c"}
    for g in [0, 1]:
        idx = grp["phq9_binary"].values == g
        ax.scatter(emb[idx, 0], emb[idx, 1], c=colors[g],
                   label=f"{'Control' if g == 0 else 'Patient'} (n={int(idx.sum())})",
                   alpha=0.6, s=18, edgecolors="none")
    ax.set_title(f"PCA — colored by PHQ9 group (binary)\nPC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.legend()

    loadings = pca.components_.T
    top5 = np.argsort(np.abs(loadings[:, 0]))[-5:]
    for i in top5:
        ax.annotate("", xy=(loadings[i, 0] * 3, loadings[i, 1] * 3), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
        ax.text(loadings[i, 0] * 3.3, loadings[i, 1] * 3.3, feats[i], fontsize=7, color="gray")

    plt.tight_layout()
    out = out_dir / "PCA_missingness.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")
    return emb, pca


def run_kmeans(X_scaled, emb, grp, out_dir, k_range=(2, 3, 4)):
    sil_scores, best_k, best_sil = {}, 2, -1
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        lbl = km.fit_predict(X_scaled)
        if len(np.unique(lbl)) > 1:
            s = silhouette_score(X_scaled, lbl)
            sil_scores[k] = s
            if s > best_sil:
                best_sil, best_k = s, k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    lbl = km.fit_predict(X_scaled)
    grp = grp.copy()
    grp["cluster"] = lbl

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(best_k):
        idx = lbl == c
        ax.scatter(emb[idx, 0], emb[idx, 1], color=cmap(c), alpha=0.65, s=22, edgecolors="none",
                   label=f"Cluster {c}  (n={int(idx.sum())}, patient={int(grp.loc[idx, 'phq9_binary'].sum())})")
    ax.set_title(f"K-means (k={best_k}, silhouette={best_sil:.3f}) on PCA of patient-level missingness")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    plt.tight_layout()
    out = out_dir / f"kmeans_k{best_k}_PCA.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(sil_scores.keys()), list(sil_scores.values()), color="#3498db", alpha=0.8, edgecolor="gray")
    ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Silhouette score")
    ax.set_title("K-means: Silhouette score by k")
    ax.set_xticks(list(sil_scores.keys()))
    plt.tight_layout()
    out = out_dir / "kmeans_silhouette.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")

    grp[["subject", "cluster", "phq9_binary", "phq9_mean"]].to_csv(out_dir / "kmeans_patient_clusters.csv",
                                                                     index=False)
    return grp, best_k, sil_scores


def plot_patient_dendrogram(X_scaled, grp, out_dir, max_patients=200):
    n = len(X_scaled)
    if n > max_patients:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_patients, replace=False)
        X_s, g_s = X_scaled[idx], grp.iloc[idx].reset_index(drop=True)
        title_note = f" (random {max_patients}/{n} patients)"
    else:
        X_s, g_s, title_note = X_scaled, grp.reset_index(drop=True), f" (all {n} patients)"

    Z = linkage(X_s, method="ward")
    fig, ax = plt.subplots(figsize=(max(12, len(X_s) * 0.12), 7))
    dn = dendrogram(Z, ax=ax, color_threshold=0.7 * max(Z[:, 2]), no_labels=True)
    palette = {0: "#3498db", 1: "#e74c3c"}
    for xi, leaf in enumerate(dn["leaves"]):
        g = int(g_s.loc[leaf, "phq9_binary"])
        ax.plot(10 + xi * 10, 0, "o", color=palette[g], markersize=4)
    ax.legend(handles=[mpatches.Patch(facecolor="#3498db", label="Control"),
                        mpatches.Patch(facecolor="#e74c3c", label="Patient")])
    ax.set_title(f"Hierarchical clustering of patients by missingness pattern{title_note}\ndot color = PHQ9 group")
    ax.set_ylabel("Ward distance")
    plt.tight_layout()
    out = out_dir / "hierarchical_patients.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")


def generate_unsupervised_missingness_clustering(data_dir, out_dir=None, split=1, unit_minutes=60, cache_dir=None):
    """Does missingness alone (no labels) cluster subjects consistent with PHQ9?
    PCA scatter + K-means (silhouette-selected k) + hierarchical dendrogram."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "label_compare" / "unsupervised"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(data_dir / "processed_data" / f"{split}.npz", allow_pickle=True)
    x = np.concatenate([d["train_x"], d["val_x"], d["test_x"]], axis=0)
    y = np.concatenate([d["train_y"], d["val_y"], d["test_y"]], axis=0)
    feats = json.load(open(data_dir / "feature_columns.json"))
    subject_ids = json.load(open(data_dir / "subject_ids.json"))

    M = compute_sample_missrate(np.isnan(x), feats, unit_minutes)
    phq_df = load_phq9_continuous(cache_dir)
    X_pat, grp = patient_level(M, y, subject_ids, phq_df)
    X_scaled = StandardScaler().fit_transform(X_pat)

    emb, pca = plot_pca(X_scaled, grp, feats, out_dir)
    grp_clustered, best_k, sil_scores = run_kmeans(X_scaled, emb, grp, out_dir)
    plot_patient_dendrogram(X_scaled, grp, out_dir)
    print(f"Done -> {out_dir}")
    return grp_clustered, sil_scores


# =====================================================================
# Performance ~ demographics GEE partial-effect plot
#   (MTM/analysis/lmm_effect_size.py's build_df/run_inference machinery +
#   MTM/analysis/lmm_partial_effect_patient.py's headline-finding figure)
#
# Brought back in per explicit instruction: this shows "baseline model gets
# worse with more patient-group missingness, the augmented model flips to
# better" — a headline claim, not out of scope. `statsmodels` is imported
# lazily (function-local, not module-level) because this project's other
# conda env (coformer) does not have statsmodels installed and subgroup_cm.py
# imports AGE_LABELS/build_demo_lookup from this module in both envs — a
# module-level `import statsmodels` would break that shared import path.
#
# The other 9 MTM/analysis/lmm_*.py files were NOT migrated — see the
# migration report for a one-line summary of each and the reasoning.
# =====================================================================

def _load_test_arrays(datapath, target):
    """target: "phq9"|"gad7" — processed_data/1_{target}.npz 파일명에 필요
    (예전엔 1.npz로 하드코딩돼 있어서 지금의 target-suffixed 레이아웃에서 항상
    FileNotFoundError였다 — 작업 1/3 진행 중 발견해 같이 고침).

    test_smids(window_id): sample_ids.json이 subject_ids.json과 같은 순서(train+val+test
    concat)로 저장돼 있다는 전제(build._combine_mtm/step3_mtm 참고)로 같은 방식(뒤
    n_test개)으로 슬라이스 — 작업 0821 A-2, CoFormer 쪽 window_id와 동일 개념."""
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    datapath = Path(datapath)
    d = np.load(datapath / "processed_data" / f"1_{target}.npz", allow_pickle=True)
    feats = json.load(open(datapath / "feature_columns.json"))
    all_sids = np.array(json.load(open(datapath / "subject_ids.json")))
    all_smids = json.load(open(datapath / "sample_ids.json"))
    n_train, n_val = d["train_x"].shape[0], d["val_x"].shape[0]
    test_x = d["test_x"]
    test_y = d["test_y"].reshape(-1)
    test_sids = all_sids[n_train + n_val:]
    test_smids = all_smids[n_train + n_val:]
    unit_minutes = json.loads((datapath / "meta.json").read_text())["unit_minutes"]
    return test_x, test_y, test_sids, test_smids, feats, unit_minutes


def _report_test_split_mismatch_mtm(strategy_root, canonical_root):
    """CoFormer 쪽 attention_viz._report_test_split_mismatch와 동일한 목적/근거/한계(그
    docstring의 "subject 집합뿐 아니라 window 집합까지" 설명 참고)의 MTM 버전 — dense
    npz(train_x/val_x/test_x split 경계)로 test subject/window(sample_id) 집합을 비교해
    stdout에 보고만 한다(중단하지 않음). grouping.split_low_high()/build._combine_mtm()이
    CoFormer와 같은 이유로 strategy마다 다른 test population을 만든다(모듈 docstring 참고)."""
    def _test_windows(root):
        npzs = sorted((root / "processed_data").glob("1_*.npz"))
        if not npzs:
            return None
        d = np.load(npzs[0], allow_pickle=True)
        sids = np.array(json.load(open(root / "subject_ids.json")))
        smids = json.load(open(root / "sample_ids.json"))
        n_tr, n_va = d["train_x"].shape[0], d["val_x"].shape[0]
        return (set(sids[n_tr + n_va:].astype(str).tolist()),
                set(smids[n_tr + n_va:]))

    own = _test_windows(strategy_root)
    can = _test_windows(canonical_root)
    if own is None or can is None:
        missing = strategy_root if own is None else canonical_root
        print(f"[!] test split 비교 건너뜀 (processed_data/1_*.npz 없음: {missing})")
        return
    own_test_subjects, own_test_windows = own
    can_test_subjects, can_test_windows = can

    if own_test_windows == can_test_windows:
        print(f"  test split 확인: strategy own test windows == canonical test windows "
              f"({len(own_test_windows)}개, subject {len(own_test_subjects)}명) — 우연히 완전히 "
              f"일치, 그래도 canonical 소스를 계속 사용")
        return
    only_own_w = own_test_windows - can_test_windows
    only_can_w = can_test_windows - own_test_windows
    only_own_s = own_test_subjects - can_test_subjects
    only_can_s = can_test_subjects - own_test_subjects
    print(f"[!] test split 불일치 확인됨 — window 단위: strategy own={len(own_test_windows)}개, "
          f"canonical(baseline)={len(can_test_windows)}개, own에만={len(only_own_w)}개, "
          f"canonical에만={len(only_can_w)}개 | subject 단위: own={len(own_test_subjects)}명, "
          f"canonical={len(can_test_subjects)}명, own에만={len(only_own_s)}명, "
          f"canonical에만={len(only_can_s)}명 "
          f"-> canonical(baseline)만 사용해 strategy 간 평가를 통일함")


def _run_mtm_inference(datapath, ckpt_path, target, test_datapath=None):
    """Run a trained MTM checkpoint, return (preds, labels, prob_true).

    Uses `src.models.mtm.MTMModule`/`RunConfigView`/`load_run_config` — NOT
    pristine `tasks.clsf_module.ClassificationModule`/`config.mtm_clsf_config.
    MTM_Custom_V2_Auto` (the original `lmm_effect_size.py::run_inference` used
    those, but neither exists/works against the pristine `external/mtm`
    submodule the way the old working copy's modified versions did — see
    `src.analysis.attention_viz.resolve_config`'s docstring for the full
    explanation; this is the same fix applied there and in subgroup_cm.py).

    target: "phq9"|"gad7" — RaindropDataModule의 split_idx는 train_mtm.py와 동일하게
    f"1_{target}"(processed_data/1_{target}.npz를 가리키는 문자열 키)이어야 한다.
    이전엔 정수 1을 그대로 넘겨 processed_data/1.npz(존재하지 않음)를 찾다가
    UEA가 아닌 데이터셋 분기(_setup_raindrop, P12/P19/PAM 전용)로 빠져 KeyError가
    났다 — 작업 3 스모크테스트 중 발견해 같이 고침.

    test_datapath: 실제 test 배열을 읽어올 데이터셋(기본값: datapath 자신). 모델 구조/
    하이퍼파라미터(config)는 여전히 datapath(체크포인트가 학습된 own dataset)에서 읽되,
    RaindropDataModule에는 test_datapath를 넘겨 test_dataloader가 그쪽 test split을 쓰게
    한다 — 작업 0821 B: strategy 간 test population을 단일 소스(보통 datasets/baseline)로
    통일하기 위함(build_lmm_performance_df 호출부 docstring 참고). num_chn/window 크기는
    같은 target 안에서 모든 strategy 데이터셋이 동일한 feature 전체집합을 쓰므로 architecture
    mismatch 위험은 없다."""
    import torch

    from .. import paths as _paths
    from ..models.mtm import MTMModule, RunConfigView, load_run_config
    from ..vendor import vendor_ctx

    config = RunConfigView(load_run_config(str(datapath)))
    test_datapath = Path(test_datapath) if test_datapath else Path(datapath)
    with vendor_ctx(_paths.EXTERNAL_MTM):
        from data_modules.raindrop import RaindropDataModule
        # compact=True: train_mtm.py도 config가 아니라 리터럴로 넘김(custom_v2.yaml에
        # compact 키 자체가 없어 config.compact는 항상 AttributeError였다 — 작업 3
        # 스모크테스트 중 발견해 같이 고침).
        rdm = RaindropDataModule(str(test_datapath), f"1_{target}", config.batch_size,
                                 dataset=config.dataset, compact=True)
        test_loader = rdm.test_dataloader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MTMModule.load_from_checkpoint(
        str(ckpt_path), model=config.get_model(), forward_fn=config.forward_fn).to(device)
    model.eval()
    preds, labels, prob_true = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, x_mask, t, y, x_static, _ = batch
            x, x_mask, t = x.to(device), x_mask.to(device), t.to(device)
            x_static = x_static.to(device) if x_static is not None else None
            logits = model.model(x, x_mask, t, x_static)
            probs = torch.softmax(logits, dim=-1)
            preds.append(logits.argmax(-1).cpu().numpy())
            labels.append(y.numpy())
            prob_true.append(probs.gather(1, y.to(device).long().unsqueeze(1)).squeeze(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(prob_true)


def build_lmm_performance_df(datapath, ckpt_path, target, strategy=None, seed=None,
                             test_datapath=None, missrate_band=None, out_dir=None):
    """Window-level dataframe (correct 0/1, prob_true_class, age/sex/patient/missrate
    + block별 missrate_{block} covariates) for a GEE performance~demographics fit.
    Ported from lmm_effect_size.py's `build_df(mode="performance")` — the
    missingness-mode branch (Y=missrate itself, used by lmm_effect_size.py's
    `--mode missingness`) wasn't needed by lmm_partial_effect_patient.py so isn't
    ported here. missrate_{block} 컬럼명은 grouping.BLOCKS 키(A_hr_rr/B1_temp/...)와
    attention_viz.generate_coformer_infer_for_lmm 쪽 컬럼명이 반드시 일치해야 함
    (두 conda env에서 따로 도는 코드라 런타임에 이름 불일치를 못 잡음).
    target: "phq9"|"gad7" — datapath/processed_data/1_{target}.npz를 읽는 데 필요.

    테스트셋 통일 (작업 0821 B — CoFormer 쪽 attention_viz.generate_coformer_infer_for_lmm과
    동일한 문제/근거, 그 함수 docstring 및 `_report_test_split_mismatch_mtm` 참고):
    strategy별 자기 own datapath(DATA_ROOT/mtm/{target}/datasets/{strategy})의 test split은
    strategy마다 population이 다르다(low_missing/high_missing은 window-count-matched
    부분집합, augmented_*는 low.test ∪ high.test). 그래서 실제 test 배열은 항상
    test_datapath(기본값: DATA_ROOT/mtm/{target}/datasets/baseline)에서 읽고, 모델
    구조/체크포인트만 datapath/ckpt_path에서 가져온다(_run_mtm_inference의
    test_datapath 인자로 전달).

    레거시 재현: 이 통일 전 결과(기존 partial_effect_patient/lmm_effect_size 그림·표)를
    그대로 다시 뽑아야 하면 test_datapath에 datapath(strategy 자기 own dataset)를 그대로
    넘기면 된다 — 그러면 이 fix 이전과 동일하게 그 데이터셋 own test split만 사용한다(신규
    컬럼 추가 외에는 수치가 동일). test_datapath를 생략(기본값=baseline)했을 때만 통일된
    평가가 적용된다 — _plot_partial_effect_patient_panel/run_lmm_effect_size는 지금 이
    인자를 안 넘기므로 기본값(baseline 통일)이 자동 적용됨에 유의.

    strategy: 결과 CSV 파일명(mtm_infer_{strategy}.csv)과 strategy 컬럼에 씀. 생략하면
    Path(datapath).name을 씀(DATA_ROOT/mtm/{target}/datasets/{strategy} 관례와 일치할 때만
    유효 — 다른 경로 레이아웃이면 명시적으로 넘길 것).
    seed: 넘기면 seed 컬럼을 추가하고 파일명이 mtm_infer_{strategy}_seed{seed}.csv가 된다
    (안 넘기면 mtm_infer_{strategy}.csv — 여러 seed를 돌리면서 덮어쓰기 방지, ΔNLL을 같은
    seed끼리 짝짓기 위해 build_lmm_input.py가 이 컬럼을 읽음. 교수님 지시, 0821: pseudoreplication
    방지를 위해 3-seed로 학습했으니 ΔNLL도 seed 평균이 아니라 seed-matched로 계산할 것).
    missrate_band: {subject_id: "low"|"high"}. 생략하면 missing_rate.subject_missrate_band()를
    이 target의 MTM baseline(기본적으로 test_datapath와 같은 경로)에 대해 돌려 계산한다 —
    grouping.split_low_high가 group_membership.json을 만든 것과 동일 데이터/계산이라
    diff=0(verify_missrate_band.py로 검증됨). CoFormer 쪽(attention_viz.
    generate_coformer_infer_for_lmm)도 기본값이 같은 MTM baseline 소스라 band가 모델 간
    자동으로 일치한다.

    출력에 신규 스키마 컬럼(subject_id/window_id/strategy/y_true/p_true_class/nll,
    nll = -log(clip(p_true_class, 1e-7, 1-1e-7)))도 추가하고 CSV로 저장한다
    (mtm_infer_{strategy}.csv — attention_viz.generate_coformer_infer_for_lmm의
    coformer_infer_{strategy}.csv와 같은 컬럼 스키마, build_lmm_input.py --model
    coformer|mtm이 둘 다 읽을 수 있게 맞춤). 기존 반환값(DataFrame, 구 컬럼들)은 그대로라
    _plot_partial_effect_patient_panel/run_lmm_effect_size/build_strategy_long_df 등
    기존 호출부는 변경 없이 계속 동작."""
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    datapath = Path(datapath)
    strategy = strategy or datapath.name
    test_datapath = Path(test_datapath) if test_datapath else (
        paths.DATA_ROOT / "mtm" / target / "datasets" / "baseline")
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "lmm_effect_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] strategy={strategy!r} target={target!r} — test windows sourced from single "
          f"canonical data_root (all strategies share this): {test_datapath}")
    if datapath != test_datapath:
        _report_test_split_mismatch_mtm(datapath, test_datapath)

    test_x, test_y, test_sids, test_smids, feats, unit_minutes = _load_test_arrays(test_datapath, target)
    M = compute_sample_missrate(np.isnan(test_x), feats, unit_minutes)
    win_missrate = M.mean(axis=1)

    df = pd.DataFrame({"subject": [str(s) for s in test_sids], "window_id": list(test_smids),
                       "label": test_y.astype(int), "missrate": win_missrate})
    M_df = pd.DataFrame(M, columns=feats)
    for bn, bfeats in BLOCKS.items():
        cols = [f for f in bfeats if f in feats]
        df[f"missrate_{bn}"] = M_df[cols].mean(axis=1).values
    preds, labels, prob_true = _run_mtm_inference(datapath, ckpt_path, target, test_datapath=test_datapath)
    assert len(preds) == len(df), f"{len(preds)} vs {len(df)}"
    assert np.array_equal(labels, test_y[:len(labels)]), "order mismatch"
    df["correct"] = (preds == labels).astype(int)
    df["prob_true_class"] = prob_true

    if missrate_band is None:
        from .missing_rate import subject_missrate_band
        band_source = paths.DATA_ROOT / "mtm" / target / "datasets" / "baseline"
        print(f"[*] computing subject missrate band from MTM baseline {band_source} "
              f"(missing_rate.subject_missrate_band — must match grouping.split_low_high's "
              f"own group_membership.json exactly, diff=0 verified via verify_missrate_band.py) ...")
        missrate_band = subject_missrate_band(band_source, target)["missrate_band"].to_dict()
    df["missrate_band"] = df["subject"].map(missrate_band)

    demo = build_demo_lookup()
    df = df.join(demo, on="subject").dropna(subset=["Sex", "AgeGroup"])
    df["age_ord"] = df["AgeGroup"].map({a: i for i, a in enumerate(AGE_LABELS)}).astype(float)
    df["sex_F"] = (df["Sex"] == "Female").astype(int)
    df["patient"] = df["label"]  # subject phq9 window label(0/1)
    df["missrate_z"] = (df["missrate"] - df["missrate"].mean()) / df["missrate"].std()

    # 신규 스키마 (작업 0821 A-2) — CoFormer 쪽과 동일 컬럼명
    df["subject_id"] = df["subject"]
    df["strategy"] = strategy
    df["y_true"] = df["label"]
    df["p_true_class"] = df["prob_true_class"]
    df["nll"] = -np.log(np.clip(df["prob_true_class"], 1e-7, 1 - 1e-7))
    if seed is not None:
        df["seed"] = seed  # ΔNLL을 같은 seed끼리 짝짓기 위함(교수님 지시, 0821) — build_lmm_input.py가 읽음

    out_path = out_dir / (f"mtm_infer_{strategy}_seed{seed}.csv" if seed is not None
                          else f"mtm_infer_{strategy}.csv")
    df.to_csv(out_path, index=False)
    print(f"[*] n_windows={len(df)} n_subjects={df['subject'].nunique()} "
          f"mean_nll={df['nll'].mean():.4f}")
    print(f"  saved: {out_path}")
    return df


_PATIENT_COLOR = {"Control": "#2980b9", "Patient": "#c0392b"}


def _plot_partial_effect_patient_panel(ax, datapath, ckpt_path, target, title):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = build_lmm_performance_df(datapath, ckpt_path, target)
    gee = smf.gee("correct ~ age_ord + sex_F + patient + missrate", groups="subject", data=df,
                 family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    df["patient_f"] = df["patient"].map({0: "Control", 1: "Patient"})

    binned = df.groupby(["age_ord", "patient_f"], as_index=False)["correct"].mean()
    for grp, color in _PATIENT_COLOR.items():
        sub = binned[binned.patient_f == grp]
        ax.scatter(sub["age_ord"], sub["correct"], s=90, color=color, label=grp,
                  edgecolors="white", linewidths=1, zorder=3)

    x_grid = np.linspace(0, 4, 50)
    sex_mean, mr_mean = df["sex_F"].mean(), df["missrate"].mean()
    for grp, patient_v in [("Control", 0), ("Patient", 1)]:
        logit = (gee.params["Intercept"] + gee.params["age_ord"] * x_grid
                + gee.params["sex_F"] * sex_mean + gee.params["patient"] * patient_v
                + gee.params["missrate"] * mr_mean)
        prob = 1 / (1 + np.exp(-logit))
        ax.plot(x_grid, prob, color=_PATIENT_COLOR[grp], lw=2.5)

    z = gee.tvalues["patient"]
    star = "***" if gee.pvalues["patient"] < .001 else "**" if gee.pvalues["patient"] < .01 else "*" if gee.pvalues["patient"] < .05 else "ns"
    ax.set_xticks(range(5)); ax.set_xticklabels(AGE_LABELS)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Age group"); ax.set_ylabel("P(correct)")
    ax.set_title(f"{title}\npatient coef z={z:+.2f} {star}", fontsize=11, weight="bold")
    ax.legend(title="", fontsize=9)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)


def plot_partial_effect_patient(baseline_datapath, baseline_ckpt,
                                augmented_datapath, augmented_ckpt, target, out_dir=None):
    """The project's headline finding: "baseline model gets worse with more
    patient-group missingness, the augmented model flips to better" — shown as
    two side-by-side GEE-logistic partial-effect panels (age on x-axis,
    P(correct) on y-axis, colored by patient/control). No hardcoded paths —
    all four dataset/checkpoint locations are required arguments (the original
    `lmm_partial_effect_patient.py` hardcoded a specific baseline/augmented
    checkpoint pair under `/home/hail/robot_ai/MTM/logs/...`; callers must now
    pass their own checkpoint paths explicitly). target: "phq9"|"gad7", 두 체크포인트
    공통(같은 target으로 학습된 baseline/augmented 비교)."""
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "lmm_effect_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    print("[*] panel (a) baseline ...")
    _plot_partial_effect_patient_panel(axes[0], baseline_datapath, baseline_ckpt, target,
                                       "(a) baseline — patient WORSE")
    print("[*] panel (b) augmented_plus_high ...")
    _plot_partial_effect_patient_panel(axes[1], augmented_datapath, augmented_ckpt, target,
                                       "(b) augmented_plus_high — patient BETTER")
    fig.suptitle("Performance ~ Age, split by Patient/Control — baseline vs augmented\n"
                "dots=empirical rate per bin, line=GEE logistic fit", fontsize=13, weight="bold")
    plt.tight_layout()
    out = out_dir / "lmm_partial_effect_patient_baseline_vs_augmented.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  saved: {out}")
    return out


# =====================================================================
# LMM effect-size (full CLI + forest plot + GLMM option)
#   (MTM/analysis/lmm_effect_size.py, in full)
# =====================================================================

LMM_FACTOR_LABELS = {
    "age_z": "Age (older→)", "sex_F": "Sex (Female)",
    "patient": "Patient (vs Control)", "missrate_z": "Missing rate (higher→)",
    "patient:missrate_z": "Patient x Missing rate",
}


def build_lmm_missingness_df(datapath, target):
    """Window-level dataframe (missrate continuous + age/sex/patient covariates)
    for a MixedLM missingness~demographics fit — no checkpoint/inference needed
    (Y=missrate itself). Ported from lmm_effect_size.py's
    `build_df(mode="missingness")`; shares `_load_test_arrays`/`build_demo_lookup`
    with `build_lmm_performance_df` above (mode="performance")."""
    test_x, test_y, test_sids, _test_smids, feats, unit_minutes = _load_test_arrays(datapath, target)
    M = compute_sample_missrate(np.isnan(test_x), feats, unit_minutes)
    win_missrate = M.mean(axis=1)

    df = pd.DataFrame({"subject": [str(s) for s in test_sids],
                       "label": test_y.astype(int), "missrate": win_missrate})
    demo = build_demo_lookup()
    df = df.join(demo, on="subject").dropna(subset=["Sex", "AgeGroup"])
    df["age_ord"] = df["AgeGroup"].map({a: i for i, a in enumerate(AGE_LABELS)}).astype(float)
    df["sex_F"] = (df["Sex"] == "Female").astype(int)
    df["patient"] = df["label"]
    df["age_z"] = (df["age_ord"] - df["age_ord"].mean()) / df["age_ord"].std()
    df["missrate_z"] = (df["missrate"] - df["missrate"].mean()) / df["missrate"].std()
    return df


def lmm_forest_plot(coef, title, out_path):
    c = coef[coef.index != "Intercept"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(c) + 1.5))
    y = np.arange(len(c))
    ax.errorbar(c["estimate"], y, xerr=1.96 * c["SE"], fmt="o", color="#2c3e50",
                ecolor="#7f8c8d", elinewidth=2, capsize=4, ms=7)
    ax.axvline(0, color="#e74c3c", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LMM_FACTOR_LABELS.get(i, i) for i in c.index])
    for yi, (_, r) in zip(y, c.iterrows()):
        star = "***" if r["p"] < .001 else "**" if r["p"] < .01 else "*" if r["p"] < .05 else "ns"
        ax.text(r["estimate"], yi + 0.18, f"z={r['z']:+.2f} {star}", fontsize=8, ha="center")
    ax.set_ylim(-0.6, len(c) - 1 + 0.6)  # 맨 위 점의 z값 텍스트가 title과 안 겹치게 여백 확보
    ax.set_xlabel("coefficient (log-odds / std outcome)  ±95% CI")
    ax.set_title(title, fontsize=12, weight="bold", pad=16)
    ax.grid(axis="x", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")
    return out_path


def run_lmm_effect_size(mode, datapath, target, ckpt_path=None, tag="", glmm=False,
                        interaction=False, out_dir=None):
    """"어떤 요인(나이·성별·환자여부·결측률)이 결과를 좌우하는가"를 셀별 쪼개기 대신
    전체 표본을 한 모형에 넣어 통계적으로 추정 (셀별 Mann-Whitney의 BH-FDR 다중비교
    탈락 문제를 full-N mixed model로 대체). Ported from MTM/analysis/lmm_effect_size.py
    in full (CLI options, GLMM, forest plot) — no hardcoded datapath/ckpt defaults
    (the original hardcoded a specific baseline dataset/checkpoint under
    `/home/hail/robot_ai/MTM/...`; both are now required-by-mode arguments).

    mode="performance": Y=window-level correct(0/1) from a trained MTM checkpoint
      (GEE logistic, subject cluster-robust SE; --ckpt required).
    mode="missingness": Y=window-level period-adjusted missing rate (continuous)
      (MixedLM, subject random intercept; no checkpoint needed).
    tag: output filename suffix (e.g. "_augmented") so per-model results don't overwrite.
    glmm: also fit BinomialBayesMixedGLM (random intercept) for mode="performance"
      (slow, off by default — GEE cluster-robust SE already handles repeated measures).
    interaction: adds patient:missrate_z to mode="performance" — distinguishes
      "patient accuracy gain is proportional to missingness" from a pure patient effect.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "lmm_effect_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] mode={mode}  datapath={datapath}  building analysis frame ...")
    if mode == "performance":
        if ckpt_path is None:
            raise ValueError("mode='performance' requires ckpt_path")
        df = build_lmm_performance_df(datapath, ckpt_path, target)
    elif mode == "missingness":
        df = build_lmm_missingness_df(datapath, target)
    else:
        raise ValueError(f"unknown mode: '{mode}'")
    print(f"  n_windows={len(df)}  n_subjects={df['subject'].nunique()}  "
          f"patient_windows={int(df.patient.sum())}")

    factors = "C(AgeGroup) + sex_F + patient + missrate_z"
    if mode == "performance" and interaction:
        factors += " + patient:missrate_z"

    if mode == "performance":
        formula = f"correct ~ {factors}"
        print(f"[*] GEE logistic (cluster=subject): {formula}")
        gee = smf.gee(formula, groups="subject", data=df,
                      family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        coef = pd.DataFrame({"estimate": gee.params, "SE": gee.bse,
                             "z": gee.tvalues, "p": gee.pvalues})
        coef["OR"] = np.exp(coef["estimate"])
        if glmm:
            try:
                from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
                print("[*] BinomialBayesMixedGLM (random intercept=subject) ...")
                glmm_fit = BinomialBayesMixedGLM.from_formula(
                    formula, {"subject": "0 + C(subject)"}, df).fit_vb()
                fe = list(glmm_fit.model.exog_names)
                coef["glmm_estimate"] = [glmm_fit.fe_mean[fe.index(n)] if n in fe else np.nan
                                         for n in coef.index]
            except Exception as e:
                print(f"  (GLMM 생략: {e})")
        outcome_desc = "P(correct)"
    else:
        formula = f"missrate ~ {factors.replace('missrate_z + ', '').replace(' + missrate_z', '')}"
        print(f"[*] MixedLM (random intercept=subject): {formula}")
        md = smf.mixedlm(formula, df, groups=df["subject"]).fit()
        coef = pd.DataFrame({"estimate": md.params, "SE": md.bse,
                             "z": md.tvalues, "p": md.pvalues})
        coef = coef[~coef.index.str.contains("Group")]  # random effect var 제외
        outcome_desc = "period-adjusted missing rate"

    coef.index.name = "term"
    coef.to_csv(out_dir / f"lmm_{mode}{tag}_coef.csv")
    print(f"  saved: {out_dir / f'lmm_{mode}{tag}_coef.csv'}")

    rank = coef[coef.index != "Intercept"].reindex(
        coef[coef.index != "Intercept"]["z"].abs().sort_values(ascending=False).index)
    print(f"\n[effect size 순위 — outcome={outcome_desc}, |z| 내림차순]")
    for term, r in rank.iterrows():
        star = "***" if r["p"] < .001 else "**" if r["p"] < .01 else "*" if r["p"] < .05 else "ns"
        lab = LMM_FACTOR_LABELS.get(term, term)
        extra = f", OR={r['OR']:.2f}" if "OR" in coef.columns else ""
        print(f"  {lab:24s} estimate={r['estimate']:+.3f}{extra} "
              f"z={r['z']:+.2f} p={r['p']:.3g} {star}")

    lmm_forest_plot(coef, f"Effect sizes on {outcome_desc}  ({mode}{tag})",
                    out_dir / f"lmm_{mode}{tag}_forest.png")
    print("\nDone.")
    return coef


# =====================================================================
# LMM robustness checks (VIF / residuals / convergence / ANOVA triangulation /
# omitted-covariate check) — validates lmm_effect_size's conclusions, doesn't
# produce new ones.
#   (MTM/analysis/lmm_robustness_checks.py, in full)
# =====================================================================

def _build_lmm_base_df(datapath, target):
    """Base window-level frame (AgeGroup/sex_F/patient only — no missrate/correct
    outcome column yet, callers add their own Y). Ported from
    MTM/analysis/lmm_effect_size_by_block.py's `build_base_df()` — only this one
    function is migrated from that file (not its own block-heatmap CLI), since
    lmm_robustness_checks.py needs it as a hard dependency and that file itself
    wasn't in scope for this migration.

    age는 연속형 age_z가 아니라 범주형 C(AgeGroup)으로 모형에 들어간다(작업 4) —
    증강 자체가 age band로 정의되므로. age_ord는 남겨둠: _lmm_check_vif의 collinearity
    체크용(VIF는 스케일 불변이라 age_z든 age_ord든 결과 동일)과, 이 df를 안 쓰는
    _plot_partial_effect_patient_panel류의 다른 continuous-age 플롯과는 무관."""
    test_x, test_y, test_sids, _test_smids, feats, unit_minutes = _load_test_arrays(datapath, target)
    M = compute_sample_missrate(np.isnan(test_x), feats, unit_minutes)

    demo = build_demo_lookup()
    base = pd.DataFrame({"subject": [str(s) for s in test_sids]})
    base = base.join(demo, on="subject")
    keep = base["Sex"].notna() & base["AgeGroup"].notna()
    base = base[keep].reset_index(drop=True)
    M = M[keep.values]

    base["age_ord"] = base["AgeGroup"].map({a: i for i, a in enumerate(AGE_LABELS)}).astype(float)
    base["sex_F"] = (base["Sex"] == "Female").astype(int)
    base["patient"] = test_y[keep.values].astype(int)
    return base, M, feats


def _lmm_check_vif(base, cols=("age_ord", "sex_F", "patient")):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    cols = list(cols)
    X = base[cols].copy()
    X["const"] = 1.0
    rows = []
    for i, col in enumerate(cols):
        rows.append({"term": col, "VIF": variance_inflation_factor(X.values, i)})
    df = pd.DataFrame(rows)
    df["flag"] = df["VIF"].apply(lambda v: "OK" if v < 5 else "주의" if v < 10 else "위험")
    return df


def _lmm_check_residuals(base, M, feats, blocks=None):
    """8개 블록 전부(수렴 성공한 것만) 잔차 진단 — A_hr_rr 하나만 보면 8개 중 가장
    연속형에 가까운 블록만 보는 셈이라, Y가 4~6종류뿐인 블록의 위반이 더 클 위험을 놓친다."""
    import statsmodels.formula.api as smf
    from scipy import stats as sps

    if blocks is None:
        blocks = list(BLOCKS)
    rows = []
    for block in blocks:
        cols = [feats.index(f) for f in BLOCKS[block] if f in feats]
        df = base.copy()
        df["missrate"] = M[:, cols].mean(axis=1)
        if df["missrate"].nunique() < 3:
            continue  # 사실상 상수 -> MixedLM 자체가 의미 없음
        try:
            md = smf.mixedlm("missrate ~ C(AgeGroup) + sex_F + patient", df, groups=df["subject"]).fit()
        except Exception as e:
            rows.append({"block": block, "n_resid": len(df), "resid_mean": np.nan,
                         "skew": np.nan, "kurtosis": np.nan, "shapiro_p": np.nan,
                         "shapiro_n_sampled": 0, "verdict": f"fit 실패: {e}"})
            continue
        resid = md.resid
        n_sample = min(5000, len(resid))
        _, shap_p = sps.shapiro(resid.sample(n_sample, random_state=0))
        ok = abs(resid.skew()) < 0.5 and abs(resid.kurtosis()) < 0.5
        rows.append({
            "block": block, "n_resid": len(resid), "resid_mean": resid.mean(),
            "skew": resid.skew(), "kurtosis": resid.kurtosis(),
            "shapiro_p": shap_p, "shapiro_n_sampled": n_sample,
            "verdict": "형식적 위반이나 |skew|,|kurtosis|<0.5로 경미" if ok else "위반 크기 큼 — 재검토 필요",
        })
    return pd.DataFrame(rows)


def _lmm_check_omitted_covariates(base, M, cache_dir=None):
    """build_demo_lookup()이 Sex/Age만 뽑아써서, 여태 모델에 한 번도 안 들어간
    BMI(Height/Weight)/GAD7(불안)/source(설문회차, 잠재적 배치효과)를 통제 변수로
    추가했을 때 age/sex/patient 계수가 버티는지 확인."""
    import statsmodels.formula.api as smf

    cache_dir = Path(cache_dir) if cache_dir else paths.DATA_ROOT / "cache"
    survey = pickle.load(open(cache_dir / "survey_all.pkl", "rb"))
    first = survey.drop_duplicates("ID", keep="first").set_index("ID")
    ext = pd.DataFrame({
        "bmi": first["Weight"] / (first["Height"] / 100) ** 2,
        "gad7": first["GAD7_Score"],
        "source_grp": first["source"].astype(str).str.split("_").str[0],
    })
    df = base.copy()
    df["missrate"] = M.mean(axis=1)
    df = df.join(ext, on="subject").dropna(subset=["bmi", "gad7", "source_grp"])
    df["bmi_z"] = (df.bmi - df.bmi.mean()) / df.bmi.std()
    df["gad7_z"] = (df.gad7 - df.gad7.mean()) / df.gad7.std()

    m0 = smf.mixedlm("missrate ~ C(AgeGroup) + sex_F + patient", df, groups=df["subject"]).fit()
    m1 = smf.mixedlm("missrate ~ C(AgeGroup) + sex_F + patient + bmi_z + gad7_z + C(source_grp)",
                     df, groups=df["subject"]).fit()

    rows = []
    for t in list(m0.params.index):
        if t == "Intercept" or "Group Var" in t:
            continue
        rows.append({"term": t, "model": "baseline(공변량 없음)",
                     "estimate": m0.params[t], "z": m0.tvalues[t], "p": m0.pvalues[t]})
    for t in list(m1.params.index):
        if t == "Intercept" or "Group Var" in t:
            continue
        rows.append({"term": t, "model": "BMI+GAD7+회차 통제 후",
                     "estimate": m1.params[t], "z": m1.tvalues[t], "p": m1.pvalues[t]})
    out = pd.DataFrame(rows)
    out["n_windows"] = len(df)
    out["n_subjects"] = df["subject"].nunique()
    return out


def _lmm_check_convergence(base, M, feats):
    import warnings

    import statsmodels.formula.api as smf

    rows = []
    for bn, bfeats in BLOCKS.items():
        cols = [feats.index(f) for f in bfeats if f in feats]
        df = base.copy()
        df["missrate"] = M[:, cols].mean(axis=1)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            md = smf.mixedlm("missrate ~ C(AgeGroup) + sex_F + patient", df, groups=df["subject"]).fit()
        rows.append({"block": bn, "converged": md.converged, "n_warnings": len(wlist),
                     "n_unique_y": df["missrate"].nunique()})
    return pd.DataFrame(rows)


def _lmm_fema_gee_reanalysis(base, M, feats):
    """F_ema는 window당 값이 5개뿐인 사실상 이산변수라 연속형 MixedLM이 부적합
    (수렴 실패) -> 성능 LMM과 동일하게 GEE 로지스틱으로 대체."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    cols = [feats.index(f) for f in BLOCKS["F_ema"] if f in feats]
    df = base.copy()
    df["missrate"] = M[:, cols].mean(axis=1)
    df["any_ema"] = (df["missrate"] < 1.0).astype(int)
    gee = smf.gee("any_ema ~ C(AgeGroup) + sex_F + patient", groups="subject", data=df,
                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    return (pd.DataFrame({"estimate": gee.params, "SE": gee.bse, "z": gee.tvalues, "p": gee.pvalues})
            .reset_index().rename(columns={"index": "term"}))


def _lmm_anova_subject_level(base, M):
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    base = base.copy()
    base["missrate"] = M.mean(axis=1)
    subj = base.groupby("subject").agg(
        missrate=("missrate", "mean"), AgeGroup=("AgeGroup", "first"),
        Sex=("Sex", "first"), patient=("patient", lambda y: int(y.mean() >= 0.5)),
    ).reset_index()
    subj["patient_f"] = subj["patient"].map({0: "Control", 1: "Patient"})
    model = ols("missrate ~ C(AgeGroup) + C(Sex) + C(patient_f)", data=subj).fit()
    table = anova_lm(model, typ=2).reset_index().rename(columns={"index": "term"})
    table.attrs["n_subjects"] = len(subj)
    return table


def run_lmm_robustness_checks(datapath, target, out_dir=None, cache_dir=None):
    """"돌려보니 되던데요"가 아니라 논문에 실릴 수 있는 수준인지 확인하기 위한 5가지
    체크(VIF/잔차진단/수렴여부/ANOVA삼각검증/누락공변량). 이 함수 자체는 새 결론을
    만들지 않고, lmm_effect_size의 기존 결론이 버티는지 확인한다. Ported from
    MTM/analysis/lmm_robustness_checks.py in full — no hardcoded datapath/cache_dir
    (both are now arguments)."""
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "lmm_effect_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] building base frame (baseline test split) ...")
    base, M, feats = _build_lmm_base_df(datapath, target)
    print(f"  n_windows={len(base)}  n_subjects={base['subject'].nunique()}")

    print("\n[1/5] VIF (다중공선성)")
    vif = _lmm_check_vif(base)
    vif.to_csv(out_dir / "robustness_vif.csv", index=False)
    print(vif.to_string(index=False))

    print("\n[2/5] 8개 블록 MixedLM 수렴 여부")
    conv = _lmm_check_convergence(base, M, feats)
    conv.to_csv(out_dir / "robustness_convergence.csv", index=False)
    print(conv.to_string(index=False))

    print("\n[3/5] 잔차 진단 — 8개 블록 전부 (수렴 성공한 블록만)")
    converged_blocks = conv.loc[conv.converged, "block"].tolist()
    resid = _lmm_check_residuals(base, M, feats, blocks=converged_blocks)
    resid.to_csv(out_dir / "robustness_residual_diagnostics.csv", index=False)
    print(resid.to_string(index=False))

    fema = None
    if not conv.loc[conv.block == "F_ema", "converged"].iloc[0]:
        print("\n  -> F_ema 수렴 실패 확인됨. GEE 로지스틱(Y=any_ema)으로 재분석:")
        fema = _lmm_fema_gee_reanalysis(base, M, feats)
        fema.to_csv(out_dir / "robustness_fema_gee_reanalysis.csv", index=False)
        print(fema.to_string(index=False))

    print("\n[4/5] ANOVA 삼각검증 (subject 단위, 반복측정 없음)")
    anova = _lmm_anova_subject_level(base, M)
    anova.to_csv(out_dir / "robustness_anova_subject_level.csv", index=False)
    print(f"  n_subjects={anova.attrs['n_subjects']}")
    print(anova.to_string(index=False))

    print("\n[5/5] 누락 공변량 체크 (BMI/GAD7/설문회차 통제 후 age/sex/patient 버티는지)")
    omitted = _lmm_check_omitted_covariates(base, M, cache_dir=cache_dir)
    omitted.to_csv(out_dir / "robustness_omitted_covariates.csv", index=False)
    print(f"  n_windows={omitted['n_windows'].iloc[0]}  n_subjects={omitted['n_subjects'].iloc[0]}")
    print(omitted.to_string(index=False))

    lines = ["# LMM Robustness Checks\n"]
    lines.append(f"- 표본: baseline test split, n_windows={len(base)}, n_subjects={base['subject'].nunique()}\n")
    lines.append("\n## 1. VIF (다중공선성)\n")
    lines.append(vif.to_markdown(index=False) + "\n")
    lines.append("\n## 2. 수렴 여부 (8개 블록)\n")
    lines.append(conv.to_markdown(index=False) + "\n")
    lines.append("\n## 3. 잔차 진단 (수렴 성공한 블록 전부)\n")
    lines.append(resid.to_markdown(index=False) + "\n")
    if fema is not None:
        lines.append("\n## 3-1. F_ema GEE 재분석 (수렴 실패로 인한 대체)\n")
        lines.append(fema.to_markdown(index=False) + "\n")
    lines.append("\n## 4. ANOVA 삼각검증 (subject 단위)\n")
    lines.append(anova.to_markdown(index=False) + "\n")
    lines.append("\n## 5. 누락 공변량 체크 (BMI/GAD7/설문회차)\n")
    lines.append(omitted.to_markdown(index=False) + "\n")
    (out_dir / "robustness_summary.md").write_text("".join(lines))
    print(f"\n  saved summary: {out_dir / 'robustness_summary.md'}")
    print("\nDone.")
    return {"vif": vif, "convergence": conv, "residuals": resid, "fema": fema,
            "anova": anova, "omitted_covariates": omitted}


# =====================================================================
# LMM effect-size — CoFormer cross-check
#   (MTM/analysis/lmm_effect_size_coformer.py, in full)
#
# Fits the same GEE logistic as run_lmm_effect_size(mode="performance") but on
# CoFormer's own predictions — reads the CSV that
# attention_viz.generate_coformer_infer_for_lmm() (coformer env) wrote, since
# CoFormer's model/dataloader can only be loaded in that env (dgl etc), while
# fitting GEE needs statsmodels (mtm env). MTM baseline had a negative patient
# coefficient (patients scored worse) that flipped positive under
# augmented_plus_high — this checks whether CoFormer reproduces the same flip.
# =====================================================================

def build_lmm_coformer_df(infer_csv_path):
    """Reads a coformer_infer_{tag}.csv (from
    attention_viz.generate_coformer_infer_for_lmm) and joins demographics —
    ported from lmm_effect_size_coformer.py's build_df()."""
    df = pd.read_csv(infer_csv_path, dtype={"subject": str})
    demo = build_demo_lookup()
    df = df.join(demo, on="subject").dropna(subset=["Sex", "AgeGroup"])
    df["age_ord"] = df["AgeGroup"].map({a: i for i, a in enumerate(AGE_LABELS)}).astype(float)
    df["sex_F"] = (df["Sex"] == "Female").astype(int)
    df["missrate_z"] = (df["missrate"] - df["missrate"].mean()) / df["missrate"].std()
    return df


def run_lmm_effect_size_coformer(tag, infer_csv_path=None, out_dir=None):
    """tag: "baseline" or "augmented" (matches attention_viz.
    generate_coformer_infer_for_lmm's --tag). infer_csv_path defaults to
    OUTPUTS_ROOT/analysis/lmm_effect_size/coformer_infer_{tag}.csv (where that
    function writes it)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "lmm_effect_size"
    out_dir.mkdir(parents=True, exist_ok=True)
    infer_csv_path = Path(infer_csv_path) if infer_csv_path else out_dir / f"coformer_infer_{tag}.csv"

    print(f"[*] mode=performance model=coformer_{tag}  building analysis frame ...")
    df = build_lmm_coformer_df(infer_csv_path)
    print(f"  n_windows={len(df)}  n_subjects={df['subject'].nunique()}  "
          f"patient_windows={int(df.patient.sum())}")

    formula = "correct ~ C(AgeGroup) + sex_F + patient + missrate_z"
    print(f"[*] GEE logistic (cluster=subject): {formula}")
    gee = smf.gee(formula, groups="subject", data=df,
                  family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    coef = pd.DataFrame({"estimate": gee.params, "SE": gee.bse,
                         "z": gee.tvalues, "p": gee.pvalues})
    coef["OR"] = np.exp(coef["estimate"])
    coef.index.name = "term"

    tag_suffix = f"_coformer_{tag}"
    coef.to_csv(out_dir / f"lmm_performance{tag_suffix}_coef.csv")
    print(f"  saved: {out_dir / f'lmm_performance{tag_suffix}_coef.csv'}")

    rank = coef[coef.index != "Intercept"].reindex(
        coef[coef.index != "Intercept"]["z"].abs().sort_values(ascending=False).index)
    print(f"\n[effect size 순위 — outcome=P(correct), model=coformer_{tag}, |z| 내림차순]")
    for term, r in rank.iterrows():
        star = "***" if r["p"] < .001 else "**" if r["p"] < .01 else "*" if r["p"] < .05 else "ns"
        lab = LMM_FACTOR_LABELS.get(term, term)
        print(f"  {lab:24s} estimate={r['estimate']:+.3f}, OR={r['OR']:.2f} "
              f"z={r['z']:+.2f} p={r['p']:.3g} {star}")

    lmm_forest_plot(coef, f"Effect sizes on P(correct)  (CoFormer, {tag})",
                    out_dir / f"lmm_performance{tag_suffix}_forest.png")
    print("\nDone.")
    return coef


# =====================================================================
# Strategy long-format LMM (작업 3) — strategy를 예측변수로 넣은 단일 모형.
# run_lmm_effect_size/run_lmm_effect_size_coformer(전략별 개별 fit, 계수를 눈으로
# 비교)는 부록용으로 그대로 남겨두고, 이게 주분석이 된다.
# =====================================================================

def build_strategy_long_df(runs, target, export_csv=None):
    """runs: [{"strategy":..., "datapath":..., "ckpt":...}, ...] (MTM) 또는
    [{"strategy":..., "infer_csv":...}, ...] (CoFormer, attention_viz.
    generate_coformer_infer_for_lmm의 출력) — 섞어서 줘도 됨(strategy별로 소스가
    다를 수 있음). 정확히 하나는 strategy="baseline"이어야 함(Δ 계산 기준).

    subject 집합은 전략 간 완전히 동일하길 기대하지만, split_low_high의
    window-count-matching 랜덤 서브샘플 때문에 test window가 1개뿐인 subject가
    특정 전략 데이터셋에서만 통째로 빠지는 경우가 정상적으로 생긴다(조사 결과
    특정 인구집단/missrate band에 쏠리지 않음 확인). 그래서 정확히 일치하지 않아도
    죽지 않고 교집합만 써서 진행하되, 무엇이 얼마나 빠졌는지 항상 stdout에 출력하고
    (논문 N 표기용) 손실률이 5%를 넘으면(=순수 표본 변동으로 설명 안 될 정도면) assert.

    window-level -> subject-level 집계(acc/p_true=mean, patient/Sex/AgeGroup=first,
    missrate류=mean) 후 strategy 컬럼을 붙여 세로로 합치고, baseline 대비
    degradation_p_true/degradation_acc = baseline − strategy를 계산한다(기존 R
    분석/미팅로그와 방향 통일 — 양수=baseline 대비 성능 저하, 음수=개선).
    baseline 행 자체는 Δ 계산 후 제거(factor 수준으로 안 넣음 — 독립성 가정 위반 방지).

    missrate_band: 새로 median을 계산하지 않고 grouping.py::split_low_high가 만든
    data/mtm/{target}/group_membership.json(빌드 시점 rank-based median split)을
    그대로 재사용한다. 다만 이 함수의 test-subject 모집단으로 같은 방법(rank-based
    median)을 재계산해 membership 파일과 어긋나지 않는지 assert로 검증한다
    (전체 population median과 test-subject subset median이 다른 모집단이면 안 됨).

    z-score 표준화는 concat 후 한 번만(전략별로 따로 하면 스케일이 섞이는 버그).
    export_csv: 지정하면 long-format을 CSV로 저장(R lme4에서 바로 읽을 수 있게)."""
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    strategies = [r["strategy"] for r in runs]
    if len(strategies) != len(set(strategies)):
        raise ValueError(f"strategy 중복: {strategies}")
    if "baseline" not in strategies:
        raise ValueError("runs에 strategy='baseline' 행이 반드시 있어야 함 (Δ 계산 기준)")

    block_cols = [f"missrate_{bn}" for bn in BLOCKS]
    agg = {"acc": ("correct", "mean"), "p_true": ("prob_true_class", "mean")}
    agg.update({c: (c, "first") for c in ["patient", "Sex", "AgeGroup"]})
    agg.update({c: (c, "mean") for c in ["missrate"] + block_cols})

    subj_dfs = {}
    for run in runs:
        strategy = run["strategy"]
        if "ckpt" in run:
            win_df = build_lmm_performance_df(run["datapath"], run["ckpt"], target, strategy=strategy)
        elif "infer_csv" in run:
            win_df = build_lmm_coformer_df(run["infer_csv"])
        else:
            raise ValueError(f"run에 ckpt 또는 infer_csv가 필요: {run}")
        missing_cols = [c for c in ["correct", "prob_true_class", "patient", "Sex",
                                    "AgeGroup", "missrate"] + block_cols if c not in win_df]
        if missing_cols:
            raise ValueError(f"strategy={strategy!r}: window df에 컬럼 없음 {missing_cols} "
                             f"— attention_viz.generate_coformer_infer_for_lmm / "
                             f"build_lmm_performance_df가 최신 스키마로 만들었는지 확인")
        subj_dfs[strategy] = win_df.groupby("subject").agg(**agg)

    # subject 집합 동일성 검증 — 완전 일치는 요구하지 않되(각 strategy 데이터셋이
    # split_low_high의 window-count-matching 랜덤 서브샘플을 거치면서, window 수가
    # 적은(주로 1개) subject가 통째로 빠지는 게 정상적으로 발생함 — 조사 결과 특정
    # 인구집단/missrate band에 쏠리지 않은 순수 표본 변동으로 확인됨), 교집합만
    # 쓰고 무엇이 얼마나 빠졌는지 항상 stdout에 보고한다(논문에 정확한 N을 적어야 함).
    subject_sets = {s: set(df.index) for s, df in subj_dfs.items()}
    union = set().union(*subject_sets.values())
    ref = set.intersection(*subject_sets.values())
    n_dropped_total = len(union) - len(ref)
    drop_rate = n_dropped_total / len(union) if union else 0.0
    print(f"[*] subject 집합 교집합: {len(ref)}/{len(union)} "
          f"(전략 합집합 대비 {n_dropped_total}명 제외, {drop_rate:.2%})")
    for s, subs in subject_sets.items():
        dropped = sorted(subs - ref)
        if dropped:
            print(f"  strategy={s!r}: {len(subs)}명 중 교집합에서 제외된 subject "
                 f"{len(dropped)}명 -> {dropped}")
    assert drop_rate <= 0.05, (
        f"교집합 손실률이 {drop_rate:.2%}로 너무 큼(5% 초과) — split_low_high 서브샘플로"
        f"설명되는 정상 범위(보통 <1%)를 넘어섬. 전처리/체크포인트 mismatch 의심, "
        f"union={len(union)} vs 교집합={len(ref)}.")
    ref_sorted = sorted(ref)
    base = subj_dfs["baseline"].loc[ref_sorted]

    # missrate_band: group_membership.json 재사용 + "같은 모집단(=baseline 전체
    # train+val+test, split_low_high가 median을 계산했던 바로 그 집합)"에서 같은
    # rank-based median split을 재계산해 일치하는지 assert. test-subject subset만으로
    # 재계산하면 모집단이 달라 median 자체가 어긋나므로(처음엔 이렇게 했다가 무의미한
    # 11% 불일치로 assert가 걸려 발견/수정함) 반드시 baseline 전체를 다시 로드한다.
    membership_path = paths.DATA_ROOT / "mtm" / target / "group_membership.json"
    membership = json.loads(membership_path.read_text())
    low_set, high_set = set(membership["low_subjects"]), set(membership["high_subjects"])
    not_covered = ref - (low_set | high_set)
    if not_covered:
        raise ValueError(f"{membership_path}에 없는 subject {len(not_covered)}명 존재 "
                         f"(예: {sorted(not_covered)[:5]}) — split_low_high 재실행 필요할 수 있음")
    membership_band = {s: ("low" if s in low_set else "high") for s in ref}

    baseline_run = next(r for r in runs if r["strategy"] == "baseline")
    if "datapath" in baseline_run:
        bp = Path(baseline_run["datapath"])
        d_full = np.load(bp / "processed_data" / f"1_{target}.npz", allow_pickle=True)
        full_sids = np.array(json.load(open(bp / "subject_ids.json")))
        full_feats = json.load(open(bp / "feature_columns.json"))
        full_unit_min = json.loads((bp / "meta.json").read_text())["unit_minutes"]
        X_full = np.concatenate([d_full["train_x"], d_full["val_x"], d_full["test_x"]], axis=0)
        M_full = compute_sample_missrate(np.isnan(X_full), full_feats, full_unit_min)
        full_df = pd.DataFrame({"subject_id": full_sids, "score": M_full.mean(axis=1)})
        full_subj = full_df.groupby("subject_id")["score"].mean().sort_values(kind="stable")
        half = len(full_subj) // 2
        recomputed_low = set(full_subj.iloc[:half].index)
        n_mismatch = sum(1 for s in ref if (s in recomputed_low) != (membership_band[s] == "low"))
        mismatch_rate = n_mismatch / len(ref)
        assert mismatch_rate <= 0.02, (
            f"missrate_band 재계산(baseline 전체 population, split_low_high와 동일 방법)이 "
            f"group_membership.json과 {n_mismatch}/{len(ref)}명({mismatch_rate:.1%}) 불일치 — "
            f"group_membership.json의 source={membership['source']!r}와 baseline_run['datapath']="
            f"{bp!s}가 같은 baseline 데이터인지 확인할 것(다른 빌드/버전이면 재실행 필요).")
        print(f"[*] missrate_band: baseline 전체 population({len(full_subj)}명) 재계산이 "
              f"group_membership.json과 {len(ref) - n_mismatch}/{len(ref)}명 일치 "
              f"({mismatch_rate:.1%} 불일치, threshold=2%)")
    else:
        print("[!] missrate_band: baseline run에 datapath가 없어(CoFormer infer_csv만 줌) "
              "재계산 검증 생략 — group_membership.json을 그대로 신뢰함")

    long_rows = []
    for strategy, subj in subj_dfs.items():
        if strategy == "baseline":
            continue
        d = subj.loc[ref_sorted].copy()
        # degradation = baseline - strategy (양수=성능 저하, 음수=baseline 대비 개선)
        d["degradation_p_true"] = base["p_true"].to_numpy() - d["p_true"].to_numpy()
        d["degradation_acc"] = base["acc"].to_numpy() - d["acc"].to_numpy()
        d["strategy"] = strategy
        d["missrate_band"] = [membership_band[s] for s in d.index]
        d["subject"] = d.index
        long_rows.append(d.reset_index(drop=True))
    long_df = pd.concat(long_rows, ignore_index=True)

    # z-score는 concat 후 한 번만
    long_df["missrate_z"] = (long_df["missrate"] - long_df["missrate"].mean()) / long_df["missrate"].std()
    for c in block_cols:
        long_df[f"{c}_z"] = (long_df[c] - long_df[c].mean()) / long_df[c].std()
    long_df["age_ord"] = long_df["AgeGroup"].map({a: i for i, a in enumerate(AGE_LABELS)}).astype(float)
    long_df["sex_F"] = (long_df["Sex"] == "Female").astype(int)

    print(f"[*] long_df: n_rows={len(long_df)} (strategies={sorted(set(long_df.strategy))} "
          f"x n_subjects={len(ref)}), missrate_band=[{(long_df.missrate_band=='low').sum()} low / "
          f"{(long_df.missrate_band=='high').sum()} high] (전략 수만큼 중복 카운트)")

    if export_csv:
        export_csv = Path(export_csv)
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(export_csv, index=False)
        print(f"  long-format CSV(R lme4용) 저장 -> {export_csv}")
    return long_df


def run_strategy_lmm(long_df, sig_blocks, y="degradation_p_true", out_dir=None, tag=""):
    """strategy를 예측변수로 넣은 단일 MixedLM(주모형). build_strategy_long_df()의
    출력을 그대로 받는다.

    sig_blocks: Section 4 결측 분석에서 유의했던 블록 이름 리스트(예: ["B2_ppg"],
    BLOCKS 키 중에서 고름) — 그 missrate_{block}_z만 covariate로 추가한다. 전부
    넣으면 다중공선성 위험이 크므로 일부러 필수 인자로 뒀다: 이 함수 안에서
    "무엇이 유의한 블록인지" 추측하지 않음 — Section 4 결과를 보고 호출자가 정할 것.
    빈 리스트([])를 넘기면 블록 covariate 없이 적합.

    y: "degradation_p_true"(주분석, softmax 확률 기반) 또는 "degradation_acc"
    (민감도 분석, 0/1 정확도). 둘 다 baseline − strategy 방향(양수=성능 저하,
    음수=baseline 대비 개선 — 기존 R 분석/미팅로그와 방향 통일).

    회귀식: y ~ C(strategy) * missrate_band + C(AgeGroup) + sex_F + (block covariates)
    + (1|subject). 적합 후 _lmm_check_vif로 공선성을 확인해 참고용으로 출력한다."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    if y not in ("degradation_p_true", "degradation_acc"):
        raise ValueError(f"y는 'degradation_p_true' 또는 'degradation_acc'만 지원: {y!r}")
    unknown = [b for b in sig_blocks if b not in BLOCKS]
    if unknown:
        raise ValueError(f"BLOCKS에 없는 블록: {unknown} (가능한 값: {list(BLOCKS)})")

    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "lmm_effect_size"
    out_dir.mkdir(parents=True, exist_ok=True)

    block_terms = " + ".join(f"missrate_{bn}_z" for bn in sig_blocks)
    formula = f"{y} ~ C(strategy) * missrate_band + C(AgeGroup) + sex_F"
    if block_terms:
        formula += f" + {block_terms}"
    print(f"[*] MixedLM (random intercept=subject): {formula}")
    md = smf.mixedlm(formula, long_df, groups=long_df["subject"]).fit()
    print(md.summary())

    fe_names = list(md.fe_params.index)
    coef = pd.DataFrame({"estimate": md.fe_params, "SE": md.bse_fe,
                         "z": md.tvalues[fe_names], "p": md.pvalues[fe_names]})
    coef.index.name = "term"
    out_csv = out_dir / f"strategy_lmm_{y}{tag}_coef.csv"
    coef.to_csv(out_csv)
    print(f"  saved: {out_csv}")

    if sig_blocks:
        vif_cols = ["missrate_band_bin"] + [f"missrate_{bn}_z" for bn in sig_blocks]
        vif_df = long_df.assign(missrate_band_bin=(long_df["missrate_band"] == "high").astype(int))
        vif = _lmm_check_vif(vif_df, cols=vif_cols)
        print(f"[*] VIF (block covariates 공선성 확인용):\n{vif}")

    lmm_forest_plot(coef, f"Strategy effect on {y}", out_dir / f"strategy_lmm_{y}{tag}_forest.png")
    return md, coef


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["pattern_by_label", "cell_imbalance", "unsupervised",
                                       "demographics", "partial_effect_patient",
                                       "lmm_effect_size", "lmm_robustness_checks",
                                       "lmm_effect_size_coformer"],
                    default="pattern_by_label")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--label", type=str, default="high")
    ap.add_argument("--baseline_datapath", type=str, default=None)
    ap.add_argument("--baseline_ckpt", type=str, default=None)
    ap.add_argument("--augmented_datapath", type=str, default=None)
    ap.add_argument("--augmented_ckpt", type=str, default=None)
    ap.add_argument("--lmm_mode", choices=["performance", "missingness"], default="performance",
                    help="--mode lmm_effect_size only")
    ap.add_argument("--datapath", type=str, default=None, help="--mode lmm_effect_size/lmm_robustness_checks")
    ap.add_argument("--ckpt", type=str, default=None, help="--mode lmm_effect_size (mode=performance only)")
    ap.add_argument("--tag", type=str, default="",
                    help="--mode lmm_effect_size output filename suffix; "
                         "--mode lmm_effect_size_coformer 'baseline'/'augmented'")
    ap.add_argument("--glmm", action="store_true", help="--mode lmm_effect_size: also fit BinomialBayesMixedGLM")
    ap.add_argument("--interaction", action="store_true",
                    help="--mode lmm_effect_size: add patient:missrate_z interaction term")
    ap.add_argument("--target", choices=["phq9", "gad7"], default=None,
                    help="--mode partial_effect_patient/lmm_effect_size/lmm_robustness_checks: "
                         "processed_data/1_{target}.npz 파일명에 필요")
    args = ap.parse_args()

    if args.mode == "pattern_by_label":
        generate_missing_pattern_by_label(args.data_dir, args.out_dir)
    elif args.mode == "cell_imbalance":
        generate_cell_imbalance_report(args.data_dir, args.out_dir, label=args.label)
    elif args.mode == "unsupervised":
        generate_unsupervised_missingness_clustering(args.data_dir, args.out_dir)
    elif args.mode == "partial_effect_patient":
        for req in ("baseline_datapath", "baseline_ckpt", "augmented_datapath", "augmented_ckpt", "target"):
            if getattr(args, req) is None:
                raise SystemExit(f"--{req} is required for --mode partial_effect_patient")
        plot_partial_effect_patient(args.baseline_datapath, args.baseline_ckpt,
                                    args.augmented_datapath, args.augmented_ckpt, args.target, args.out_dir)
    elif args.mode == "lmm_effect_size":
        if args.datapath is None:
            raise SystemExit("--datapath is required for --mode lmm_effect_size")
        if args.target is None:
            raise SystemExit("--target is required for --mode lmm_effect_size")
        if args.lmm_mode == "performance" and args.ckpt is None:
            raise SystemExit("--ckpt is required for --mode lmm_effect_size --lmm_mode performance")
        run_lmm_effect_size(args.lmm_mode, args.datapath, args.target, args.ckpt, tag=args.tag,
                            glmm=args.glmm, interaction=args.interaction, out_dir=args.out_dir)
    elif args.mode == "lmm_robustness_checks":
        if args.datapath is None:
            raise SystemExit("--datapath is required for --mode lmm_robustness_checks")
        if args.target is None:
            raise SystemExit("--target is required for --mode lmm_robustness_checks")
        run_lmm_robustness_checks(args.datapath, args.target, out_dir=args.out_dir)
    elif args.mode == "lmm_effect_size_coformer":
        if not args.tag:
            raise SystemExit("--tag ('baseline' or 'augmented') is required for "
                            "--mode lmm_effect_size_coformer")
        run_lmm_effect_size_coformer(args.tag, out_dir=args.out_dir)
    else:
        demographics_summary(out_dir=args.out_dir)
