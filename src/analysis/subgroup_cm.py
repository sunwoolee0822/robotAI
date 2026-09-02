"""
subgroup_cm.py
===============
인구집단별로 나눠 본 분류 성능.

만드는 것
  - 성별 x 나이 셀별 confusion matrix
  - 셀별 F1 히트맵
  - 결측률 불균형 vs F1 산점도  (셀의 결측 불균형이 성능을 예측하나)
  - 나이를 통제한 결측 효과
  - 블록별 F1
  - 결측률 4분위별 F1
  - 셀별 블록 결측률 격차 표 (CSV)
  - 나이대별 accuracy vs F1  (다수 클래스 때문에 성능이 좋아 보이는지 확인)

무슨 분석인가
  - 특정 집단에서만 성능이 나쁜가
      성별/나이 셀마다 F1이 다르면 그 집단에 증강이 더 필요하다는 뜻.
  - 성능 차이가 결측 때문인가
      셀의 결측률 불균형과 F1의 관계를 봄. 관계가 있으면 결측이 원인.
  - 결측이 많으면 실제로 성능이 떨어지나
      결측률 4분위별 F1. 실험1의 핵심 주장이 여기서 확인됨.
  - accuracy가 높은 게 착시인가
      대조군이 훨씬 많아서 다 정상이라 찍어도 accuracy가 높게 나옴.
      나이대별로 accuracy와 F1을 같이 봐야 함.

실행
  - MTM  : run_mtm_confusion_by_subgroup(ckpt_path, data_path, ...)
  - CoFormer : load_predictions(pred_npz, data_root, ...) 후 각 plot 함수

조건
  - 학습 결과 필요 (ckpt 또는 예측 npz)
  - 인구통계는 cache_all의 설문 데이터에서 가져옴
  - 표본이 얇은 셀은 결과가 불안정하니 n을 같이 볼 것
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .. import paths
from ..vendor import vendor_ctx
from .label_compare import AGE_LABELS, build_demo_lookup

DEVICE = "cuda"


# =====================================================================
# MTM-side: confusion matrix by (Sex x AgeGroup) cell
#   (MTM/analysis/confusion_by_subgroup.py)
# =====================================================================

def run_mtm_confusion_by_subgroup(ckpt_path, data_path, out_dir=None, cache_dir=None):
    """Run inference with a trained MTM checkpoint on its test split, join
    Sex/AgeGroup, and compute per-cell confusion matrix / F1.

    Loads the checkpoint via `src.models.mtm.MTMModule`/`RunConfigView`/
    `load_run_config` (this project's own Lightning wrapper + auto-config —
    NOT pristine `tasks.clsf_module.ClassificationModule`/`config.mtm_clsf_config.
    MTM_Custom_V2_Auto`, neither of which exist/work against the pristine
    `external/mtm` submodule the way the old working copy's modified versions
    did — see `src.analysis.attention_viz.resolve_config`'s docstring for the
    full explanation). `RaindropDataModule` is genuinely unmodified upstream,
    so it's still loaded from vendor via vendor_ctx.
    """
    import torch

    from ..models.mtm import MTMModule, RunConfigView, load_run_config

    data_path = Path(data_path)
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "subgroup_cm" / "mtm"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = RunConfigView(load_run_config(str(data_path)))

    with vendor_ctx(paths.EXTERNAL_MTM):
        from data_modules.raindrop import RaindropDataModule
        rdm = RaindropDataModule(config.datapath, 1, config.batch_size,
                                  dataset=config.dataset, compact=config.compact)
        test_loader = rdm.test_dataloader()

    model = MTMModule.load_from_checkpoint(
        str(ckpt_path), model=config.get_model(), forward_fn=config.forward_fn,
    ).to(DEVICE)
    model.eval()

    d = np.load(data_path / "processed_data" / "1.npz")
    n_train, n_val = d["train_x"].shape[0], d["val_x"].shape[0]
    all_sids = np.array(json.load(open(data_path / "subject_ids.json")))
    test_sids = all_sids[n_train + n_val:]
    test_y_ref = d["test_y"]

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, x_mask, t, y, x_static, uids = batch
            x, x_mask, t = x.to(DEVICE), x_mask.to(DEVICE), t.to(DEVICE)
            x_static = x_static.to(DEVICE) if x_static is not None else None
            logits = model.model(x, x_mask, t, x_static)
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    assert len(preds) == len(test_sids), f"len mismatch {len(preds)} vs {len(test_sids)}"
    assert np.array_equal(labels, test_y_ref.reshape(-1)[:len(labels)]), "collected labels != npz test_y order"

    df = pd.DataFrame({"subject_id": [str(s) for s in test_sids], "pred": preds.astype(int),
                       "label": labels.astype(int)})
    demo = build_demo_lookup(cache_dir)
    df = df.join(demo, on="subject_id")
    df = df.dropna(subset=["Sex", "AgeGroup"])

    cell_rows = []
    for sex in ["Male", "Female"]:
        for ageg in AGE_LABELS:
            cell = df[(df["Sex"] == sex) & (df["AgeGroup"] == ageg)]
            if len(cell) == 0:
                continue
            y_true, y_pred = cell["label"].values, cell["pred"].values
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            cell_rows.append({
                "cell": f"{sex}_{ageg}", "sex": sex, "age_group": ageg,
                "n_samples": len(cell), "n_subjects": cell["subject_id"].nunique(),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            })
    res = pd.DataFrame(cell_rows)
    res.to_csv(out_dir / "confusion_table.csv", index=False)
    plot_confusion_f1_heatmap(res, out_dir / "confusion_f1_heatmap.png")
    print(f"Done -> {out_dir}")
    return res


def plot_confusion_f1_heatmap(res: pd.DataFrame, out_path):
    mat = np.full((2, len(AGE_LABELS)), np.nan)
    annot = np.full((2, len(AGE_LABELS)), "", dtype=object)
    for i, sex in enumerate(["Male", "Female"]):
        for j, ageg in enumerate(AGE_LABELS):
            r = res[(res.sex == sex) & (res.age_group == ageg)]
            if len(r) == 0:
                continue
            mat[i, j] = r.iloc[0]["f1"]
            annot[i, j] = (f"{r.iloc[0]['f1']:.2f}\nwin={int(r.iloc[0]['n_samples'])}\n"
                           f"sub={int(r.iloc[0]['n_subjects'])}")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    sns.heatmap(mat, ax=ax, xticklabels=AGE_LABELS, yticklabels=["Male", "Female"], cmap="viridis",
                vmin=0, vmax=1, annot=annot, fmt="", annot_kws={"size": 8}, cbar_kws={"label": "F1"})
    ax.set_title("Baseline: per-(Sex x Age) cell F1 (test set)\nwin=test windows, sub=unique subjects",
                 fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


def plot_imbalance_vs_performance(imbalance_csv, confusion_csv, out_dir):
    """Cell-level missing-rate imbalance (imbalance_by_subgroup/label_compare's
    cell score) vs cell-level classification performance (confusion_by_subgroup) —
    Spearman scatter. Only 10 cells so treat p-values as indicative, not conclusive.
    """
    out_dir = Path(out_dir)
    imbalance = pd.read_csv(imbalance_csv)
    confusion = pd.read_csv(confusion_csv)
    df = imbalance.merge(confusion[["cell", "n_samples", "accuracy", "precision", "recall", "f1"]],
                          on="cell", how="inner")
    df.to_csv(out_dir / "imbalance_vs_performance.csv", index=False)

    score_cols = [c for c in ["n_significant_blocks", "n_sig_block", "mean_abs_cliffs_delta"]
                  if c in df.columns]
    fig, axes = plt.subplots(1, len(score_cols), figsize=(6 * len(score_cols), 5), squeeze=False)
    for ax, score_col in zip(axes[0], score_cols):
        rho, p = spearmanr(df[score_col], df["f1"])
        ax.scatter(df[score_col], df["f1"], s=60, c="#3498db", edgecolor="black", zorder=3)
        for _, row in df.iterrows():
            ax.annotate(row["cell"], (row[score_col], row["f1"]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel(score_col); ax.set_ylabel("F1 (test set)")
        ax.set_title(f"Spearman rho={rho:.2f}, p={p:.3f} (n={len(df)})", fontsize=11)
        ax.grid(alpha=0.3)
    plt.suptitle("Cell-level missing-rate imbalance vs classification performance", fontsize=13, weight="bold")
    plt.tight_layout()
    out = out_dir / "imbalance_vs_performance_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")
    return df


# =====================================================================
# CoFormer/v2_72-side: subgroup performance figures from a saved prediction file
#   (analysis/v2_72/make_figures.py + group_cm_missing.py + analysis3b_age.py +
#    analysis3c_sex_age.py)
# =====================================================================

def _band(a):
    return "<30" if a < 30 else "30s" if a < 40 else "40s" if a < 50 else "50s" if a < 60 else "60+"


def _f1s(y, p):
    return f1_score(y, p, zero_division=0) if len(y) > 5 else np.nan


def load_predictions(pred_npz, data_root, block_prefixes=None):
    """Load a saved (pred, gt) file plus the matching static/time/feature_columns/
    split for the CoFormer-style preprocessed dataset. Returns a dict with pred,
    gt, sex, age, bands, miss (overall missing rate), block_miss (dict of
    block-name -> per-sample missing rate), cols.
    """
    data_root = Path(data_root)
    P = np.load(pred_npz)
    pred, gt = P["pred"], P["gt"].reshape(-1)

    static = np.load(data_root / "static.npy")
    time = np.load(data_root / "time.npy", mmap_mode="r")
    cols = json.load(open(data_root / "feature_columns.json"))
    split = np.load(data_root / "split.npy", allow_pickle=True)
    test_idx = split[2]
    st = static[test_idx]
    sex, age = st[:, 0], st[:, 1]
    obs = (time[test_idx] >= 0)
    miss = 1.0 - obs.mean(axis=(1, 2))
    assert len(pred) == len(st), f"length mismatch {len(pred)} vs {len(st)}"

    if block_prefixes is None:
        block_prefixes = {
            "A hr/rr": ["hr_", "rr_"], "A spo2": ["spo2"],
            "B temp/bp/glu": ["core_temp", "skin_temp", "bp_", "glucose"], "C hrv": ["hrv"],
            "D phone": ["light_sensor", "proximity"],
            "E daily": ["step", "distance", "screen_time", "wake_time", "sleep_time",
                        "deep_sleep", "rem_sleep", "light_sleep", "total_sleep"],
            "F ema": ["EMA_"],
        }
    block_cols = {k: [i for i, c in enumerate(cols) if any(c.startswith(p) for p in v)]
                  for k, v in block_prefixes.items()}
    block_cols = {k: v for k, v in block_cols.items() if len(v) > 0}
    block_miss = {k: 1.0 - obs[:, ci, :].mean(axis=(1, 2)) for k, ci in block_cols.items()}

    valid = age >= 18  # drop age=0 data-quality contamination
    bands = np.array([_band(a) for a in age[valid]])
    return {
        "pred": pred[valid], "gt": gt[valid], "sex": sex[valid], "age": age[valid], "bands": bands,
        "miss": miss[valid], "block_miss": {k: m[valid] for k, m in block_miss.items()}, "cols": cols,
    }


def plot_f1_by_sex_age(data, out_path):
    """fig1: F1 by Sex x AgeGroup."""
    ages = ["<30", "30s", "40s", "50s", "60+"]
    x, w = np.arange(5), 0.38
    pred, gt, sex, bands = data["pred"], data["gt"], data["sex"], data["bands"]
    F_f1 = [_f1s(gt[(sex == 1.0) & (bands == b)], pred[(sex == 1.0) & (bands == b)]) for b in ages]
    M_f1 = [_f1s(gt[(sex == 0.0) & (bands == b)], pred[(sex == 0.0) & (bands == b)]) for b in ages]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, F_f1, w, label="Female", color="#e87ba4")
    ax.bar(x + w / 2, M_f1, w, label="Male", color="#2a78d6")
    ax.set_xticks(x); ax.set_xticklabels(ages); ax.set_xlabel("Age group"); ax.set_ylabel("F1")
    ax.set_ylim(0, 0.7); ax.set_title("F1 by Sex x Age"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, val in enumerate(F_f1):
        if not np.isnan(val):
            ax.text(i - w / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    for i, val in enumerate(M_f1):
        if not np.isnan(val):
            ax.text(i + w / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"  saved: {out_path}")


def plot_imbalance_vs_f1(data, out_path):
    """fig2: age-band Patient-Control missing-rate gap vs F1 (imbalance effect, no matching)."""
    ages = ["<30", "30s", "40s", "50s", "60+"]
    x = np.arange(5)
    gt, pred, bands, miss = data["gt"], data["pred"], data["bands"], data["miss"]
    f1_age = [_f1s(gt[bands == b], pred[bands == b]) for b in ages]
    gap = []
    for b in ages:
        m = bands == b
        y = gt[m]
        gap.append((miss[m][y == 1].mean() - miss[m][y == 0].mean()) * 100 if (y == 1).any() and (y == 0).any()
                   else np.nan)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x, f1_age, 0.5, color="#2a78d6"); ax1.set_ylabel("F1", color="#2a78d6"); ax1.set_ylim(0, 0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(ages); ax1.set_xlabel("Age group")
    ax2 = ax1.twinx()
    ax2.plot(x, gap, "o-", color="#eda100", lw=2)
    ax2.set_ylabel("Patient - Control missing gap (%p)", color="#eda100"); ax2.axhline(0, color="#ccc", lw=0.8)
    ax1.set_title("Imbalance (P-C gap) vs F1  (no matching)"); ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"  saved: {out_path}")


def plot_missing_effect_age_controlled(data, out_path):
    """fig3: within each age band, F1 for low- vs high-missing-rate half (median split)."""
    ages = ["<30", "30s", "40s", "50s", "60+"]
    x, w = np.arange(5), 0.38
    gt, pred, bands, miss = data["gt"], data["pred"], data["bands"], data["miss"]
    lo, hi = [], []
    for b in ages:
        m = bands == b
        y, p, r = gt[m], pred[m], miss[m]
        med = np.median(r)
        lo.append(_f1s(y[r <= med], p[r <= med])); hi.append(_f1s(y[r > med], p[r > med]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, lo, w, label="Low missing", color="#1baf7a")
    ax.bar(x + w / 2, hi, w, label="High missing", color="#e34948")
    ax.set_xticks(x); ax.set_xticklabels(ages); ax.set_xlabel("Age group"); ax.set_ylabel("F1")
    ax.set_ylim(0, 0.75); ax.set_title("Missing effect (age-controlled)"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"  saved: {out_path}")


def plot_f1_by_block(data, out_path):
    """fig4: within each collection-source block, F1 for low- vs high-missing half."""
    gt, pred, block_miss = data["gt"], data["pred"], data["block_miss"]
    bl_names = list(block_miss.keys())
    bl_lo, bl_hi = [], []
    for k in bl_names:
        r = block_miss[k]
        med = np.median(r)
        l, h = r <= med, r > med
        bl_lo.append(_f1s(gt[l], pred[l])); bl_hi.append(_f1s(gt[h], pred[h]))
    xb, w = np.arange(len(bl_names)), 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(xb - w / 2, bl_lo, w, label="Low missing (this block)", color="#1baf7a")
    ax.bar(xb + w / 2, bl_hi, w, label="High missing (this block)", color="#e34948")
    ax.set_xticks(xb); ax.set_xticklabels(bl_names, rotation=20, ha="right"); ax.set_ylabel("F1")
    ax.set_ylim(0, 0.6); ax.set_title("Missing effect by device/block"); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"  saved: {out_path}")


def plot_f1_by_missing_quartile(data, out_path):
    """fig5: F1 by overall-missing-rate quartile."""
    gt, pred, miss = data["gt"], data["pred"], data["miss"]
    q = np.quantile(miss, [0, 0.25, 0.5, 0.75, 1.0])
    qlabel = ["Q1\n(least missing)", "Q2", "Q3", "Q4\n(most missing)"]
    qf1 = []
    for i in range(4):
        lo_, hi_ = q[i], q[i + 1]
        m = (miss >= lo_) & (miss <= hi_) if i == 3 else (miss >= lo_) & (miss < hi_)
        qf1.append(_f1s(gt[m], pred[m]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(4), qf1, 0.55, color=["#1baf7a", "#9acd6e", "#eda100", "#e34948"])
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(qlabel); ax.set_ylabel("F1")
    ax.set_ylim(0, 0.6); ax.set_title("F1 by overall missing-rate quartile")
    ax.grid(axis="y", alpha=0.3)
    for i, val in enumerate(qf1):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"  saved: {out_path}")


def subgroup_cell_table(data, out_csv, show_blocks=("E daily", "F ema", "A hr/rr")):
    """Per (Sex, AgeGroup) cell: N/F1/Recall + per-block Patient-Control missing-rate
    gap (%p). Table companion to plot_f1_by_sex_age (analysis/v2_72/group_cm_missing.py)."""
    pred, gt, sex, bands = data["pred"], data["gt"], data["sex"], data["bands"]
    block_miss = data["block_miss"]
    show_blocks = [b for b in show_blocks if b in block_miss]
    rows = []
    for s in [1.0, 0.0]:
        for b in ["<30", "30s", "40s", "50s", "60+"]:
            m = (sex == s) & (bands == b)
            if m.sum() == 0:
                continue
            y, p = gt[m], pred[m]
            cm = confusion_matrix(y, p, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            row = {"sex": "F" if s == 1.0 else "M", "age_group": b, "n": int(m.sum()),
                   "f1": f1_score(y, p, zero_division=0),
                   "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0}
            for blk in show_blocks:
                mr = block_miss[blk][m]
                pat = mr[y == 1].mean() if (y == 1).any() else np.nan
                ctl = mr[y == 0].mean() if (y == 0).any() else np.nan
                row[f"delta_pc_{blk}"] = (pat - ctl) * 100
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv}")
    return df


def age_band_accuracy_vs_f1_table(pred, gt, age, out_csv,
                                   bins=(0, 30, 45, 60, 200), labels=("<30", "30-45", "45-60", "60+")):
    """Age-band accuracy vs F1: a large acc-F1 gap flags majority-class bias in that
    band (accuracy looks fine but the model isn't actually catching positives).
    Ported from analysis/v2_72/analysis3b_age.py part (a)/(b); its part (c)
    (statsmodels categorical-age logit) was dropped, see migration report.
    """
    age_band = pd.cut(pd.Series(age), bins=list(bins), labels=list(labels), right=False)
    rows = []
    for band in labels:
        sel = (age_band == band).values
        if sel.sum() == 0:
            continue
        y, p = gt[sel], pred[sel]
        rows.append({
            "age_band": band, "n": int(sel.sum()), "accuracy": accuracy_score(y, p),
            "f1": f1_score(y, p, zero_division=0), "true_pos_rate": float((y == 1).mean()),
            "pred_pos_rate": float((p == 1).mean()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv}")
    return df


def sexage_cell_table(pred, gt, sex, age, missing_rate, out_csv,
                       bins=(0, 30, 45, 60, 200), labels=("<30", "30-45", "45-60", "60+")):
    """8-cell (2 sex x 4 age band) performance/missing-rate table.
    Ported from analysis/v2_72/analysis3c_sex_age.py part (a); its part (b)
    (statsmodels interaction logit) was dropped, see migration report.
    """
    age_band = pd.cut(pd.Series(age), bins=list(bins), labels=list(labels), right=False)
    rows = []
    for s in sorted(pd.unique(sex)):
        for band in labels:
            sel = ((sex == s) & (age_band == band).values)
            if sel.sum() == 0:
                continue
            y, p = gt[sel], pred[sel]
            rows.append({
                "sex": s, "age_band": band, "n": int(sel.sum()), "accuracy": accuracy_score(y, p),
                "f1": f1_score(y, p, zero_division=0), "missing_rate": float(np.mean(missing_rate[sel])),
                "true_pos": float((y == 1).mean()), "pred_pos": float((p == 1).mean()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv}")
    return df


def generate_subgroup_performance_figures(pred_npz, data_root, out_dir=None):
    """Driver: figs 1-5 + the two companion tables, from a saved prediction npz."""
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "subgroup_cm" / "coformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_predictions(pred_npz, data_root)

    plot_f1_by_sex_age(data, out_dir / "fig1_f1_by_age_sex.png")
    plot_imbalance_vs_f1(data, out_dir / "fig2_imbalance_vs_f1.png")
    plot_missing_effect_age_controlled(data, out_dir / "fig3_missing_effect.png")
    plot_f1_by_block(data, out_dir / "fig4_missing_by_block.png")
    plot_f1_by_missing_quartile(data, out_dir / "fig5_f1_by_quartile.png")
    subgroup_cell_table(data, out_dir / "subgroup_cell_table.csv")
    age_band_accuracy_vs_f1_table(data["pred"], data["gt"], data["age"], out_dir / "age_band_acc_f1.csv")
    sexage_cell_table(data["pred"], data["gt"], data["sex"], data["age"], data["miss"],
                       out_dir / "sexage_cells.csv")
    print(f"Done -> {out_dir}")
    return data


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["mtm", "coformer"], default="coformer")
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--data_path", type=str, default=None)
    ap.add_argument("--pred_npz", type=str, default=None)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    if args.mode == "mtm":
        run_mtm_confusion_by_subgroup(args.ckpt_path, args.data_path, args.out_dir)
    else:
        generate_subgroup_performance_figures(args.pred_npz, args.data_root, args.out_dir)
