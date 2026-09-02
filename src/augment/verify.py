"""
verify.py
=========
group-aware augment(build.augment_mtm / build.augment_coformer)가 "제대로" low를 high
쪽으로 옮겼는지 검증한다. matplotlib를 사용한다(이 파일은 검증/리포트 체커라 HARD
CONSTRAINT의 torch 금지와 달리 matplotlib는 허용됨 — src/augment 패키지에서 matplotlib을
쓰는 유일한 파일).

두 가지 확인:
  (a) 분포 수렴(check_occurrence_convergence): block별(+셀별) 결측 occurrence가
      low -> low+aug로 가면서 high에 가까워졌는가.
  (b) 분포 일치(check_distribution_match): augmented_low vs high의 subject-level
      결측률 분포를 block/feature별 KS test + Mann-Whitney U로 비교, 시간대(diurnal)
      매치도 함께 확인.

원본: MTM/analysis/verify_augment.py + MTM/analysis/augment_distribution_test.py.

**드롭한 부분**: verify_augment.py의 `run_structure_report`/`compare_structure`는
`subprocess.run([sys.executable, "analysis/missing_pattern_report.py", ...])`로 원본
저장소의 missing_pattern_report.py를 서브프로세스로 재실행해 그 출력(pc_table.csv)을
비교하는 구조였다. missing_pattern_report.py는 이번 마이그레이션 대상이 아니므로(다른
스크립트가 참조하지 않는 report-only 스크립트), 그 스크립트에 의존하는 구조-재현 검증은
통째로 제외했다. occurrence-convergence + distribution-match만 남겼다(둘 다 self-contained,
외부 스크립트 shell-out 없음).

또한 원본 augment_coformer_group_aware.py가 만들던 "dense view snapshot"
(coformer_low_augmented_dense_view / coformer_high_dense_view, missing_pattern_report.py를
그대로 돌리기 위한 검증 전용 산출물)도 만들지 않는다 — 이 파일의 occurrence/distribution
체크는 애초에 그 스냅샷을 쓴 적이 없고(run_structure_report만 썼음), packing.unpack_dense로
그때그때 dense 변환하면 충분하기 때문이다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu

from ..paths import DATA_ROOT, OUTPUTS_ROOT
from .grouping import BLOCKS, FEATURE_PERIODS_HOURS, compute_sample_missrate
from .profile import block_slot_missing, block_bin_missing, parse_wstart
from .packing import load_packed, parse_wstart_min, unpack_dense
from .inject import build_subject_cell_label, assign_window_cell_label

LOW_COLOR, AUG_COLOR, HIGH_COLOR = "#3498db", "#9b59b6", "#e74c3c"
HOURLY_BLOCKS = ["A_hr_rr", "B1_temp", "B2_ppg"]
UNIT_MINUTES = 60


# =====================================================
# 로더 (dataset_base로 mtm/coformer 둘 다 지원)
# =====================================================

def load_mtm_dense(name, dataset_base=None):
    base = Path(dataset_base) if dataset_base else DATA_ROOT / "mtm"
    d = np.load(base / name / "processed_data" / "1.npz", allow_pickle=True)
    feats = json.load(open(base / name / "feature_columns.json"))
    sids = np.array(json.load(open(base / name / "subject_ids.json")))
    Y = np.concatenate([d["train_y"], d["val_y"], d["test_y"]], axis=0)
    X = np.concatenate([d["train_x"], d["val_x"], d["test_x"]], axis=0)
    smids = None
    p = base / name / "sample_ids.json"
    if p.exists():
        smids = json.load(open(p))
    return X, Y, feats, sids, smids


def load_coformer_dense(name, dataset_base=None, unit_minutes=UNIT_MINUTES):
    """packed CoFormer 데이터를 dense NaN-grid로 즉석 복원."""
    base = Path(dataset_base) if dataset_base else DATA_ROOT / "coformer"
    d = load_packed(base / name)
    wstart = parse_wstart_min(d["smids"])
    T = d["array"].shape[2]
    X = unpack_dense(d["array"], d["time"], d["mask"], wstart, T, unit_minutes)
    Y = d["gt"].reshape(-1)
    return X, Y, d["feats"], d["sids"], d["smids"]


# =====================================================
# (a) occurrence convergence
# =====================================================

def occurrence_by_cell_block(X, Y, feats, sids, subj_lookup=None):
    """block별, (cell 있으면 cell별) window-occurrence 평균. subj_lookup 없으면 전체 평균만."""
    mask = np.isnan(X)
    T = X.shape[1]
    rows = []
    if subj_lookup is not None:
        cell_arr, _ = assign_window_cell_label(sids, subj_lookup)
    for bn, bfeats in BLOCKS.items():
        p = FEATURE_PERIODS_HOURS[bfeats[0]]
        sl = block_slot_missing(mask, feats, bfeats)
        bin_m = block_bin_missing(sl, p, T)
        occ = bin_m.mean(axis=1)
        rows.append({"block": bn, "cell": "ALL", "occurrence": occ.mean(), "n": len(occ)})
        if subj_lookup is not None:
            for cell in sorted(set(c for c in cell_arr if c is not None)):
                m = cell_arr == cell
                if m.sum() == 0:
                    continue
                rows.append({"block": bn, "cell": cell, "occurrence": occ[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def check_occurrence_convergence(model_name, load_fn, low_name, aug_name, high_name, out_dir):
    """low / low+aug / high 3자의 block별 결측 occurrence를 비교하고, low+aug가 low보다
    high에 더 가까워졌는지 확인한다. CSV + PNG(occurrence_convergence.{csv,png})를 저장하고
    merged DataFrame을 반환한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{model_name}] occurrence convergence (low / low+aug / high) ...")
    Xl, Yl, feats, sidl, _ = load_fn(low_name)
    Xa, Ya, _, sida, _ = load_fn(aug_name)
    Xh, Yh, _, sidh, _ = load_fn(high_name)

    subj_l = build_subject_cell_label(sidl, Yl)
    subj_a = build_subject_cell_label(sida, Ya)
    subj_h = build_subject_cell_label(sidh, Yh)

    occ_l = occurrence_by_cell_block(Xl, Yl, feats, sidl, subj_l)
    occ_a = occurrence_by_cell_block(Xa, Ya, feats, sida, subj_a)
    occ_h = occurrence_by_cell_block(Xh, Yh, feats, sidh, subj_h)

    merged = (occ_l.rename(columns={"occurrence": "low"})
             .merge(occ_a.rename(columns={"occurrence": "low_aug"})[["block", "cell", "low_aug"]],
                    on=["block", "cell"])
             .merge(occ_h.rename(columns={"occurrence": "high"})[["block", "cell", "high"]],
                    on=["block", "cell"]))
    merged["gap_low"] = (merged["low"] - merged["high"]).abs()
    merged["gap_aug"] = (merged["low_aug"] - merged["high"]).abs()
    merged["improved"] = merged["gap_aug"] < merged["gap_low"]
    merged.to_csv(out_dir / "occurrence_convergence.csv", index=False)

    overall = merged[merged.cell == "ALL"]
    n_improved = int(overall["improved"].sum())
    print(f"  block(전체) 개선: {n_improved}/{len(overall)}")
    print(overall[["block", "low", "low_aug", "high", "gap_low", "gap_aug", "improved"]]
         .to_string(index=False))

    cell_rows = merged[merged.cell != "ALL"]
    n_cell_improved = int(cell_rows["improved"].sum())
    print(f"  block x cell 개선: {n_cell_improved}/{len(cell_rows)}")

    block_names = list(BLOCKS.keys())
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(block_names)); w = 0.27
    o = overall.set_index("block").loc[block_names]
    ax.bar(x - w, o["low"] * 100, w, color=LOW_COLOR, label="low (source)")
    ax.bar(x, o["low_aug"] * 100, w, color=AUG_COLOR, label="low+aug (group-aware)")
    ax.bar(x + w, o["high"] * 100, w, color=HIGH_COLOR, label="high (target)")
    ax.set_xticks(x); ax.set_xticklabels(block_names, rotation=25, ha="right")
    ax.set_ylabel("Missing occurrence (%)"); ax.set_ylim(0, 108)
    ax.set_title(f"[{model_name}] group-aware augment validation: low + injection -> high",
                fontsize=13, weight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "occurrence_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_dir / 'occurrence_convergence.png'}")
    return merged


# =====================================================
# (b) distribution match (KS/MWU + diurnal overlay)
# =====================================================

def subject_missrate(X, feats, sids):
    M = compute_sample_missrate(np.isnan(X), feats, 60)
    df = pd.DataFrame(M, columns=feats); df["sid"] = sids
    return df.groupby("sid").mean(numeric_only=True)


def diurnal_profile(X, feats, smids, block_feats):
    """hour-of-day(0~23)별 block 결측 발생률. smids로 각 slot의 실제 시각 계산.
    smids는 parse_wstart()가 가정하는 "_s{hour}" 포맷이어야 한다(CoFormer의 "_s{minutes}"는
    호출 전에 시간 단위로 변환)."""
    if smids is None:
        return None
    T = X.shape[1]
    wstart = parse_wstart(smids)
    hours = (wstart[:, None] + np.arange(T)[None, :]) % 24   # (N,T)
    sl = block_slot_missing(np.isnan(X), feats, block_feats)  # (N,T) bool
    prof = np.zeros(24)
    for h in range(24):
        m = hours == h
        prof[h] = sl[m].mean() if m.any() else np.nan
    return prof


def _minute_smids_to_hour_smids(smids):
    """CoFormer의 "_s{minutes}" sample_id를 parse_wstart()가 가정하는 "_s{hour}"로 변환."""
    out = []
    for s in smids:
        prefix, m = s.rsplit("_s", 1)
        out.append(f"{prefix}_s{int(m) // UNIT_MINUTES}")
    return out


def check_distribution_match(model="mtm", high_name="high_missing",
                             aug_name="low_missing_augmented", out_dir=None):
    """augmented_low vs high의 subject-level 결측률 분포를 block/feature별 KS test +
    Mann-Whitney U로 비교하고, 시간대(diurnal) 매치를 overlay figure로 저장한다.
    산출: ks_mwu_table.csv, distribution_overlay.png, diurnal_match.png"""
    if out_dir is None:
        out_dir = OUTPUTS_ROOT / "augment_verification" / "distribution" / model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] loading {high_name} / {aug_name} ({model}) ...")
    if model == "mtm":
        Xh, Yh, feats, sidh, smidh = load_mtm_dense(high_name)
        Xa, Ya, _, sida, smida = load_mtm_dense(aug_name)
    elif model == "coformer":
        Xh, Yh, feats, sidh, smidh = load_coformer_dense(high_name)
        Xa, Ya, _, sida, smida = load_coformer_dense(aug_name)
        # CoFormer sample_id는 "_s{minutes}"; diurnal_profile()이 쓰는 parse_wstart()는
        # "_s{hour}"를 가정하므로 시간 단위로 변환한다(원본 dense-view snapshot이 하던 변환과 동일).
        if smidh is not None:
            smidh = _minute_smids_to_hour_smids(smidh)
        if smida is not None:
            smida = _minute_smids_to_hour_smids(smida)
    else:
        raise ValueError(f"unknown model: {model!r} (expected 'mtm' or 'coformer')")

    subj_h = subject_missrate(Xh, feats, sidh)
    subj_a = subject_missrate(Xa, feats, sida)

    # ── (1) 분포 검정 (block + feature) ──────────────────────────
    rows = []

    def add(name, hv, av):
        hv, av = hv.dropna().values, av.dropna().values
        ks, ksp = ks_2samp(av, hv)
        U, mwp = mannwhitneyu(av, hv, alternative="two-sided")
        md = av.mean() - hv.mean()
        rows.append({"name": name, "high_mean": hv.mean(), "aug_mean": av.mean(),
                     "mean_diff": md, "ks_stat": ks, "ks_p": ksp, "mwu_p": mwp,
                     "close(|Δ|<.05 & KS<.3)": (abs(md) < 0.05) and (ks < 0.30)})

    for bn, bfeats in BLOCKS.items():
        cols = [f for f in bfeats if f in feats]
        add(f"block_{bn}", subj_h[cols].mean(axis=1), subj_a[cols].mean(axis=1))
    for f in feats:
        add(f"feat_{f}", subj_h[f], subj_a[f])
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "ks_mwu_table.csv", index=False)
    blk = res[res.name.str.startswith("block_")]
    n_close = int(blk["close(|Δ|<.05 & KS<.3)"].sum())
    print(f"  saved: {out_dir/'ks_mwu_table.csv'}")
    print(f"  block 근접(|Δmean|<5%p & KS<0.30): {n_close}/{len(BLOCKS)}   "
          f"(참고: n이 크면 KS p값은 항상 유의 -> 효과크기로 판정)")
    print(f"  block 평균 |Δmean| = {blk['mean_diff'].abs().mean()*100:.1f}%p, "
          f"평균 KS통계량 = {blk['ks_stat'].mean():.3f}")
    print(blk[["name", "high_mean", "aug_mean", "mean_diff", "ks_stat",
               "close(|Δ|<.05 & KS<.3)"]].to_string(index=False))

    # ── (1-fig) block별 subject 결측률 분포 overlay ───────────────
    block_names = list(BLOCKS.keys())
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, bn in zip(axes.flat, block_names):
        cols = [f for f in BLOCKS[bn] if f in feats]
        hv = subj_h[cols].mean(axis=1).dropna().values
        av = subj_a[cols].mean(axis=1).dropna().values
        bins = np.linspace(0, 1, 30)
        ax.hist(hv, bins=bins, density=True, alpha=0.5, color=HIGH_COLOR, label="missing_high (origin)")
        ax.hist(av, bins=bins, density=True, alpha=0.5, color=AUG_COLOR, label="augmented_low")
        ksp = res[res.name == f"block_{bn}"]["ks_p"].values[0]
        ax.set_title(f"{bn}  (KS p={ksp:.3f})", fontsize=11, weight="bold")
        ax.set_xlabel("subject missing rate"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"[{model.upper()}] subject-level missing-rate distribution: "
                f"augmented_low vs missing_high (origin)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "distribution_overlay.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_dir/'distribution_overlay.png'}")

    # ── (2) 시간대(diurnal) 일치 ─────────────────────────────────
    if smidh is not None and smida is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        corrs = []
        for ax, bn in zip(axes, HOURLY_BLOCKS):
            ph = diurnal_profile(Xh, feats, smidh, BLOCKS[bn])
            pa = diurnal_profile(Xa, feats, smida, BLOCKS[bn])
            ax.plot(range(24), ph*100, "-o", color=HIGH_COLOR, label="missing_high (origin)", ms=4)
            ax.plot(range(24), pa*100, "-s", color=AUG_COLOR, label="augmented_low", ms=4)
            c = np.corrcoef(ph, pa)[0, 1]
            corrs.append(c)
            ax.set_title(f"{bn}  (hour-profile corr={c:.2f})", fontsize=12, weight="bold")
            ax.set_xlabel("hour of day"); ax.set_ylabel("missing occurrence (%)")
            ax.set_xticks(range(0, 24, 3)); ax.grid(alpha=0.3); ax.legend(fontsize=9)
        fig.suptitle(f"[{model.upper()}] time-of-day (diurnal) missing profile match — hourly features",
                     fontsize=14, weight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "diurnal_match.png", dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_dir/'diurnal_match.png'}  (hour-profile corr: "
              f"{', '.join(f'{b}={c:.2f}' for b,c in zip(HOURLY_BLOCKS,corrs))})")
    else:
        print("  (sample_ids 없음 -> diurnal 생략)")

    print("Done.")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mtm", choices=["mtm", "coformer"])
    args = ap.parse_args()

    out_root = OUTPUTS_ROOT / "augment_verification"
    if args.model == "mtm":
        check_occurrence_convergence("MTM", load_mtm_dense, "low_missing",
                                     "low_missing_augmented", "high_missing", out_root / "mtm")
    else:
        check_occurrence_convergence("CoFormer", load_coformer_dense, "low_missing",
                                     "low_missing_augmented", "high_missing", out_root / "coformer")
    check_distribution_match(model=args.model)
