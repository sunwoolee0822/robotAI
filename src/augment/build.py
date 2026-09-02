"""
build.py
========
group-aware augmentation 드라이버: low_missing에 (성별x나이x환자/대조) 셀별 차등 결측
패턴을 주입해 high_missing 분포를 재현하고(`augment_mtm`/`augment_coformer`), augment된
low + 원본 high를 합쳐 "전체가 high 수준 결측"인 학습 데이터셋을 만든다
(`combine_augmented_plus_high`). 주입 자체는 `inject.inject_group_aware()`가 하고,
이 모듈은 포맷별(dense npz / packed array) 로드·저장·train-only 적용만 담당한다.

원본: MTM/analysis/augment_mtm_group_aware.py + augment_coformer_group_aware.py +
build_augmented_plus_high.py + build_coformer_augmented_plus_high.py.

──────────────────────────────────────────────────────────────────────────
CRITICAL FIX (원본 버그 수정): 원본 스크립트들은 train+val+test를 전부 concat한 배열에
`inject_group_aware()`를 통째로 돌린 뒤에야 train/val/test로 다시 slice해서 저장했다
(augment_mtm_group_aware.py의 `load_full()` -> `X = np.concatenate([train_x, val_x,
test_x])` -> `inject_group_aware(Xl_aug, ...)`는 전체 N에 대해 실행 -> 그 후에야
`train_x_aug/val_x_aug/test_x_aug`로 slice). 즉 val/test에도 synthetic 결측이 섞여
들어가 평가셋이 실제 분포를 반영하지 못하게 된다(evaluation validity 훼손).

이 파일은 **train split에만** `inject_group_aware()`를 적용하도록 고쳤다:
  - `augment_mtm()`: low의 train_x/val_x/test_x를 애초에 분리해서 로드하고, train_x만
    (그리고 train_x에 대응하는 cell/label/hours slice만) 주입 함수에 넘긴다. val_x/test_x는
    원본 low의 값을 그대로 복사해서 저장한다(주입 함수를 아예 거치지 않음).
  - `augment_coformer()`: packed 포맷은 물리적으로 분리된 train/test 배열이 없고
    `split.npy`의 인덱스로만 구분되므로, dense로 unpack한 뒤 `idx_train` 행만 잘라내
    별도 배열로 주입하고, 그 결과만 원본 dense 배열의 `idx_train` 위치에 되써서 나머지
    (`idx_val`/`idx_test`) 행은 repack 이전 dense 단계에서 원본과 byte-identical하게
    유지한다.

`combine_augmented_plus_high()`도 방어적으로 짰다: 위 fix가 있으면 이미
`low_missing_augmented`의 val/test == 원본 `low_missing`의 val/test라서 결과는 같지만,
그래도 val/test는 명시적으로 **원본 low_missing에서** 가져오고 train만
`low_missing_augmented`에서 가져온다(과제 지시사항 — optional이 아님).
──────────────────────────────────────────────────────────────────────────

사용법:
  from src.augment.build import augment_mtm, augment_coformer, combine_augmented_plus_high
  augment_mtm(ablation="full")
  augment_coformer()
  combine_augmented_plus_high("mtm", ablation="full")
  combine_augmented_plus_high("coformer")
"""
import json
import shutil
from pathlib import Path

import numpy as np

from ..paths import DATA_ROOT
from .grouping import BLOCKS
from .inject import build_subject_cell_label, assign_window_cell_label, inject_group_aware
from .packing import load_packed, parse_wstart_min, unpack_dense, repack_from_dense
from .profile import parse_wstart, build_group_profile

SEED = 42
UNIT_MINUTES = 60

# W6 ablation: cell 정의를 바꿔 lmm_effect_size.py가 밝힌 effect size 순서
# (나이 ≫ 성별 > 환자여부)대로 순차로 켜가며 비교한다.
#   uniform  : cell 없음(전역 1개), label 없음 -> 원본 block-uniform 주입과 동일
#   age      : cell=AgeGroup만, label 없음
#   age_sex  : cell=AgeGroup x Sex, label 없음
#   full     : cell=AgeGroup x Sex, label=환자/대조 (기본, 기존 low_missing_augmented와 동일)
#   rate_only: cell=AgeGroup x Sex, label=환자/대조 — full과 셀 구성은 동일하되
#              inject_mode="bernoulli_only"로 burst(패턴 재현)를 꺼서 "결측률은 같고
#              모양(burst 시간대 응집)만 다른" rate-only 베이스라인을 만든다.
#              (기존 rate_only.yaml이 "uniform"과 동일 메커니즘을 쓰던 버그의 수정판 —
#              작업 0 조사 참고)
ABLATIONS = {
    "uniform": dict(cell_axes=(), use_label=False, suffix="_uniform", inject_mode="auto",
                    note="cell 없음(전역 1개), label 없음 -> block-uniform 주입"),
    "age":     dict(cell_axes=("AgeGroup",), use_label=False, suffix="_age", inject_mode="auto",
                    note="cell=AgeGroup만, label 없음"),
    "age_sex": dict(cell_axes=("Sex", "AgeGroup"), use_label=False, suffix="_age_sex", inject_mode="auto",
                    note="cell=AgeGroup x Sex, label 없음"),
    "full":    dict(cell_axes=("Sex", "AgeGroup"), use_label=True, suffix="", inject_mode="auto",
                    note="cell=AgeGroup x Sex, label=환자/대조 (기본)"),
    "rate_only": dict(cell_axes=("Sex", "AgeGroup"), use_label=True, suffix="_rate_only",
                      inject_mode="bernoulli_only",
                      note="cell=AgeGroup x Sex, label=환자/대조 (full과 동일) + burst 끔 "
                           "(bernoulli_only) -> rate-only 베이스라인"),
}


def get_group_profile(target="phq9", dataset_base=None, cache=True):
    """target별 low/high missing profile을 계산하고 재사용한다."""
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    cache_path = DATA_ROOT / "mtm" / target / "work" / "group_profile.json"
    if cache and cache_path.exists():
        return json.loads(cache_path.read_text())
    prof = build_group_profile(target=target, dataset_base=dataset_base)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(prof, indent=2, ensure_ascii=False))
    return prof


# =====================================================
# MTM (dense npz: train_x/val_x/test_x 분리 배열)
# =====================================================

def augment_mtm(ablation="full", target="phq9", low_dir=None, high_dir=None, prof=None, seed=SEED):
    """low_missing(MTM, dense grid)에 (성별x나이x환자/대조) 셀별 차등 결측 패턴을 주입해
    high_missing 분포를 재현. train split에만 주입하고 val/test는 원본 low를 그대로 저장.
    출력: DATA_ROOT/mtm/low_missing_augmented{suffix}/ (low_missing과 동일 폴더 구조)."""
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    cfg = ABLATIONS[ablation]
    target_root = DATA_ROOT / "mtm" / target
    dataset_base = target_root / "datasets"
    low_dir = Path(low_dir) if low_dir else dataset_base / "low_missing"
    high_dir = Path(high_dir) if high_dir else dataset_base / "high_missing"
    out_dir = target_root / "work" / f"low_augmented_{ablation}"
    prof = prof if prof is not None else get_group_profile(target=target, dataset_base=dataset_base)
    rng = np.random.default_rng(seed)

    print(f"[*] ablation={ablation} ({cfg['note']}) -> {out_dir}")
    print("[*] loading low_missing / high_missing ...")
    dl = np.load(low_dir / "processed_data" / f"1_{target}.npz", allow_pickle=True)
    feats = json.load(open(low_dir / "feature_columns.json"))
    sidl = np.array(json.load(open(low_dir / "subject_ids.json")))          # full N (train+val+test)
    smidl = json.load(open(low_dir / "sample_ids.json"))                    # full N
    n_train, n_val = dl["train_x"].shape[0], dl["val_x"].shape[0]

    dh = np.load(high_dir / "processed_data" / f"1_{target}.npz", allow_pickle=True)
    sidh = np.array(json.load(open(high_dir / "subject_ids.json")))
    smidh = json.load(open(high_dir / "sample_ids.json"))
    Xh = np.concatenate([dh["train_x"], dh["val_x"], dh["test_x"]], axis=0)
    Yh = np.concatenate([dh["train_y"], dh["val_y"], dh["test_y"]], axis=0)

    T = dl["train_x"].shape[1]

    print("[*] building subject cell x label lookup ...")
    Yl_full = np.concatenate([dl["train_y"], dl["val_y"], dl["test_y"]], axis=0)
    subj_low = build_subject_cell_label(sidl, Yl_full, cell_axes=cfg["cell_axes"], use_label=cfg["use_label"])
    subj_high = build_subject_cell_label(sidh, Yh, cell_axes=cfg["cell_axes"], use_label=cfg["use_label"])
    cell_arr_low_full, label_arr_low_full = assign_window_cell_label(sidl, subj_low)
    cell_arr_high, label_arr_high = assign_window_cell_label(sidh, subj_high)
    print(f"  low: {len(subj_low)} subjects with demo, high: {len(subj_high)} subjects with demo, "
          f"n_cells={subj_high['cell'].nunique()}")

    hoursl_full = (parse_wstart(smidl)[:, None] + np.arange(T)[None, :]) % 24
    hoursh = (parse_wstart(smidh)[:, None] + np.arange(T)[None, :]) % 24
    maskh = np.isnan(Xh)

    # ── train-only injection: val/test는 애초에 주입 함수에 넘기지 않는다 ──────────
    Xl_train = dl["train_x"].copy()
    maskl_train = np.isnan(Xl_train)
    cell_arr_low = cell_arr_low_full[:n_train]
    label_arr_low = label_arr_low_full[:n_train]
    hoursl = hoursl_full[:n_train]

    print(f"[*] injecting group-aware missingness (train split only, inject_mode={cfg['inject_mode']}) ...")
    inject_group_aware(Xl_train, maskl_train, hoursl, cell_arr_low, label_arr_low,
                       maskh, hoursh, cell_arr_high, label_arr_high,
                       feats, prof, subj_high, rng, mode=cfg["inject_mode"])

    added = np.isnan(Xl_train).mean() - maskl_train.mean()
    print(f"  train 결측률 low={maskl_train.mean()*100:.1f}% -> low+aug={np.isnan(Xl_train).mean()*100:.1f}% "
          f"(high={maskh.mean()*100:.1f}%, +{added*100:.1f}%p)")

    print(f"[*] saving to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "processed_data").mkdir(exist_ok=True)

    # val_x/test_x: 원본 low_missing 그대로 (inject_group_aware를 절대 거치지 않음)
    np.savez(out_dir / "processed_data" / f"1_{target}.npz",
             train_x=Xl_train, train_y=dl["train_y"], train_stat=dl["train_stat"],
             val_x=dl["val_x"], val_y=dl["val_y"], val_stat=dl["val_stat"],
             test_x=dl["test_x"], test_y=dl["test_y"], test_stat=dl["test_stat"])

    for fname in ["feature_columns.json", "subject_ids.json", "sample_ids.json"]:
        shutil.copy(low_dir / fname, out_dir / fname)
    meta = json.loads((low_dir / "meta.json").read_text())
    meta["_augment_note"] = (f"group-aware injection from low_missing toward high_missing "
                             f"(ablation={ablation}: {cfg['note']}; "
                             f"fallback to block-global target when subgroup n<10; "
                             f"injection applied to TRAIN split only, val/test copied verbatim "
                             f"from low_missing). See src/augment/build.py:augment_mtm")
    meta["ablation"] = ablation
    meta["target"] = target
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    manifest = {
        "target": target,
        "method": ablation,
        "artifact_role": "intermediate",
        "source_low": str(low_dir),
        "source_high": str(high_dir),
        "grouping_axes": list(cfg["cell_axes"]),
        "uses_target_label": bool(cfg["use_label"]),
        "inject_mode": cfg["inject_mode"],
        "target_threshold": 10,
        "injection_scope": "train_only",
        "validation_test": "copied_from_original_low_missing",
        "seed": seed,
        "implementation": "src/augment/build.py:augment_mtm",
    }
    (out_dir / "augmentation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("Done.")
    return out_dir


# =====================================================
# CoFormer (packed: array/time/mask + split.npy)
# =====================================================

def augment_coformer(ablation="full", target="phq9", low_dir=None, high_dir=None,
                       prof=None, seed=SEED):
    """target별 CoFormer low_missing의 train split에 group-aware 결측을 주입한다."""
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    cfg = ABLATIONS[ablation]
    target_root = DATA_ROOT / "coformer" / target
    dataset_base = target_root / "datasets"
    low_dir = Path(low_dir) if low_dir else dataset_base / "low_missing"
    high_dir = Path(high_dir) if high_dir else dataset_base / "high_missing"
    out_dir = target_root / "work" / f"low_augmented_{ablation}"
    prof = prof if prof is not None else get_group_profile(
        target=target, dataset_base=DATA_ROOT / "mtm" / target / "datasets")
    rng = np.random.default_rng(seed)

    print(f"[*] target={target} ablation={ablation} ({cfg['note']}) -> {out_dir}")
    low = load_packed(low_dir)
    high = load_packed(high_dir)
    feats = low["feats"]
    if feats != high["feats"]:
        raise ValueError("low/high feature columns differ")
    meta = json.loads((low_dir / "meta.json").read_text())
    unit_minutes = int(meta.get("unit_minutes", UNIT_MINUTES))
    T = int(meta.get("window_slots", low["array"].shape[2]))

    wstart_l = parse_wstart_min(low["smids"])
    wstart_h = parse_wstart_min(high["smids"])
    Xl_full = unpack_dense(low["array"], low["time"], low["mask"], wstart_l, T, unit_minutes)
    Xh = unpack_dense(high["array"], high["time"], high["mask"], wstart_h, T, unit_minutes)
    maskl_full = np.isnan(Xl_full)
    maskh = np.isnan(Xh)

    Yl = low["gt"].reshape(-1)
    Yh = high["gt"].reshape(-1)
    subj_low = build_subject_cell_label(
        low["sids"], Yl, cell_axes=cfg["cell_axes"], use_label=cfg["use_label"])
    subj_high = build_subject_cell_label(
        high["sids"], Yh, cell_axes=cfg["cell_axes"], use_label=cfg["use_label"])
    cell_l_full, label_l_full = assign_window_cell_label(low["sids"], subj_low)
    cell_h, label_h = assign_window_cell_label(high["sids"], subj_high)

    hours_l_full = (wstart_l[:, None] // unit_minutes + np.arange(T)[None, :]) % 24
    hours_h = (wstart_h[:, None] // unit_minutes + np.arange(T)[None, :]) % 24
    idx_train = np.asarray(low["split"][0], dtype=np.int64)
    Xl_train = Xl_full[idx_train].copy()
    print(f"[*] injecting group-aware missingness (train split only, inject_mode={cfg['inject_mode']}) ...")
    inject_group_aware(
        Xl_train, maskl_full[idx_train], hours_l_full[idx_train],
        cell_l_full[idx_train], label_l_full[idx_train],
        maskh, hours_h, cell_h, label_h, feats, prof, subj_high, rng, mode=cfg["inject_mode"])

    Xl_aug = Xl_full.copy()
    Xl_aug[idx_train] = Xl_train
    added = np.isnan(Xl_train).mean() - maskl_full[idx_train].mean()
    print(f"  train missing low={maskl_full[idx_train].mean()*100:.1f}% -> "
          f"aug={np.isnan(Xl_train).mean()*100:.1f}% (+{added*100:.1f}%p)")
    array_aug, time_aug, mask_aug = repack_from_dense(
        Xl_aug, wstart_l, unit_minutes, low["array"].shape[2])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "array.npy", array_aug)
    np.save(out_dir / "time.npy", time_aug)
    np.save(out_dir / "mask.npy", mask_aug)
    for filename in ("static.npy", "gt.npy", "split.npy", "subject_ids.json",
                     "sample_ids.json", "feature_columns.json", "origins.json",
                     "w_start_days.json"):
        source = low_dir / filename
        if source.exists():
            shutil.copy(source, out_dir / filename)

    meta.update({
        "target": target,
        "ablation": ablation,
        "missing_group": f"low_augmented_{ablation}",
        "_augment_note": (f"group-aware injection toward high_missing; ablation={ablation}; "
                          "train split only; validation/test copied from original low_missing"),
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    manifest = {
        "target": target,
        "method": ablation,
        "artifact_role": "intermediate",
        "source_low": str(low_dir),
        "source_high": str(high_dir),
        "grouping_axes": list(cfg["cell_axes"]),
        "uses_target_label": bool(cfg["use_label"]),
        "inject_mode": cfg["inject_mode"],
        "target_threshold": 10,
        "injection_scope": "train_only",
        "validation_test": "copied_from_original_low_missing",
        "seed": seed,
        "implementation": "src/augment/build.py:augment_coformer",
    }
    (out_dir / "augmentation.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  saved -> {out_dir}")
    return out_dir


# =====================================================
# augmented_plus_high = low_missing_augmented(train) + low_missing(val/test, 원본) + high_missing
# =====================================================

def _combine_mtm(target, ablation):
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    cfg = ABLATIONS[ablation]
    target_root = DATA_ROOT / "mtm" / target
    dataset_base = target_root / "datasets"
    aug_dir = target_root / "work" / f"low_augmented_{ablation}"
    low_dir = dataset_base / "low_missing"
    high_dir = dataset_base / "high_missing"
    out_dir = dataset_base / f"augmented_{ablation}"

    da = np.load(aug_dir / "processed_data" / f"1_{target}.npz", allow_pickle=True)
    dl = np.load(low_dir / "processed_data" / f"1_{target}.npz", allow_pickle=True)
    dh = np.load(high_dir / "processed_data" / f"1_{target}.npz", allow_pickle=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "processed_data").mkdir(exist_ok=True)

    # 방어적 소싱: train은 low_missing_augmented(주입됨)에서, val/test는 명시적으로
    # 원본 low_missing에서 가져온다(=da가 아니라 dl). augment_mtm()의 train-only injection
    # fix가 있으면 da의 val/test == dl의 val/test라 결과는 같지만, "val/test에 주입 결과가
    # 섞여 들어가면 안 된다"는 요구사항을 이 함수 스스로도 명시적으로 강제한다.
    src = {"train": da, "val": dl, "test": dl}
    out = {}
    for split in ["train", "val", "test"]:
        for suf in ["x", "y", "stat"]:
            k = f"{split}_{suf}"
            out[k] = np.concatenate([src[split][k], dh[k]], axis=0)
        print(f"  {split}: low(+aug if train)={src[split][f'{split}_x'].shape[0]} + "
              f"high={dh[f'{split}_x'].shape[0]} -> {out[f'{split}_x'].shape[0]}")
    np.savez(out_dir / "processed_data" / f"1_{target}.npz", **out)

    # subject_ids/sample_ids도 x/y/stat과 동일한 소스([aug_train|low_val|low_test] + high)
    # 순서로 재조립 (원래 build_augmented_plus_high.py가 안 만들어서, LMM 등에서 subject
    # demo를 못 붙이던 문제 수정).
    aug_sids = json.load(open(aug_dir / "subject_ids.json"))
    low_sids = json.load(open(low_dir / "subject_ids.json"))
    high_sids = json.load(open(high_dir / "subject_ids.json"))
    aug_smids = json.load(open(aug_dir / "sample_ids.json"))
    low_smids = json.load(open(low_dir / "sample_ids.json"))
    high_smids = json.load(open(high_dir / "sample_ids.json"))

    a_ntr, a_nva = da["train_x"].shape[0], da["val_x"].shape[0]
    l_ntr, l_nva = dl["train_x"].shape[0], dl["val_x"].shape[0]
    h_ntr, h_nva = dh["train_x"].shape[0], dh["val_x"].shape[0]

    low_id_src = {
        "train": {"sids": aug_sids[:a_ntr], "smids": aug_smids[:a_ntr]},
        "val": {"sids": low_sids[l_ntr:l_ntr + l_nva], "smids": low_smids[l_ntr:l_ntr + l_nva]},
        "test": {"sids": low_sids[l_ntr + l_nva:], "smids": low_smids[l_ntr + l_nva:]},
    }
    high_id_src = {
        "train": {"sids": high_sids[:h_ntr], "smids": high_smids[:h_ntr]},
        "val": {"sids": high_sids[h_ntr:h_ntr + h_nva], "smids": high_smids[h_ntr:h_ntr + h_nva]},
        "test": {"sids": high_sids[h_ntr + h_nva:], "smids": high_smids[h_ntr + h_nva:]},
    }
    out_sids, out_smids = [], []
    for split in ["train", "val", "test"]:
        out_sids += low_id_src[split]["sids"] + high_id_src[split]["sids"]
        out_smids += low_id_src[split]["smids"] + high_id_src[split]["smids"]
    assert len(out_sids) == sum(out[f"{s}_x"].shape[0] for s in ["train", "val", "test"])
    json.dump(out_sids, open(out_dir / "subject_ids.json", "w"))
    json.dump(out_smids, open(out_dir / "sample_ids.json", "w"))
    print(f"  subject_ids/sample_ids 재조립 완료 ({len(out_sids)}개)")

    shutil.copy(high_dir / "feature_columns.json", out_dir / "feature_columns.json")
    meta = json.loads((high_dir / "meta.json").read_text())
    meta["_note"] = ("augmented_low(group-aware injection, train split only) + "
                     "low_missing(원본 val/test) + origin high_missing 결합. "
                     "전체가 high 수준 결측(train)/원본 low(val/test 물리적으로는 low 그대로). "
                     f"ablation={ablation}. "
                     "See src/augment/build.py:combine_augmented_plus_high")
    meta["missing_group"] = f"augmented_{ablation}"
    meta["target"] = target
    meta["data_version"] = "v3"
    meta["train_x_shape"] = list(out["train_x"].shape)
    meta["val_x_shape"] = list(out["val_x"].shape)
    meta["test_x_shape"] = list(out["test_x"].shape)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    manifest = {
        "target": target,
        "method": ablation,
        "artifact_role": "training_dataset",
        "source_augmented_low": str(aug_dir),
        "source_original_low": str(low_dir),
        "source_original_high": str(high_dir),
        "grouping_axes": list(cfg["cell_axes"]),
        "uses_target_label": bool(cfg["use_label"]),
        "inject_mode": cfg["inject_mode"],
        "target_threshold": 10,
        "composition": {
            "train": "augmented_low + original_high",
            "validation": "original_low + original_high",
            "test": "original_low + original_high",
        },
        "seed": SEED,
        "implementation": "src/augment/build.py:_combine_mtm",
    }
    (out_dir / "augmentation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    total_nan = np.mean([np.isnan(out[f"{s}_x"]).mean() for s in ["train", "val", "test"]]) * 100
    print(f"  전체 결측률 ~{total_nan:.1f}%")
    print(f"  saved -> {out_dir}")
    print("Done.")
    return out_dir


def _combine_coformer(target, ablation):
    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target}")
    cfg = ABLATIONS[ablation]
    target_root = DATA_ROOT / "coformer" / target
    dataset_base = target_root / "datasets"
    aug_dir = target_root / "work" / f"low_augmented_{ablation}"
    low_dir = dataset_base / "low_missing"
    high_dir = dataset_base / "high_missing"
    out_dir = dataset_base / f"augmented_{ablation}"

    aug = load_packed(aug_dir)
    low = load_packed(low_dir)
    high = load_packed(high_dir)
    if not (aug["feats"] == low["feats"] == high["feats"]):
        raise ValueError("augmented-low/low/high feature columns differ")

    idx_val = np.asarray(aug["split"][1], dtype=np.int64)
    idx_test = np.asarray(aug["split"][2], dtype=np.int64)
    array_l = aug["array"].copy()
    time_l = aug["time"].copy()
    mask_l = aug["mask"].copy()
    array_l[idx_val] = low["array"][idx_val]
    time_l[idx_val] = low["time"][idx_val]
    mask_l[idx_val] = low["mask"][idx_val]
    array_l[idx_test] = low["array"][idx_test]
    time_l[idx_test] = low["time"][idx_test]
    mask_l[idx_test] = low["mask"][idx_test]

    n_low = len(array_l)
    array = np.concatenate([array_l, high["array"]], axis=0)
    time_arr = np.concatenate([time_l, high["time"]], axis=0)
    mask_arr = np.concatenate([mask_l, high["mask"]], axis=0)
    static = np.concatenate([aug["static"], high["static"]], axis=0)
    gt = np.concatenate([aug["gt"], high["gt"]], axis=0)
    split = np.empty(3, dtype=object)
    for i, name in enumerate(("train", "val", "test")):
        idx_l = np.asarray(aug["split"][i], dtype=np.int64)
        idx_h = np.asarray(high["split"][i], dtype=np.int64) + n_low
        split[i] = np.concatenate([idx_l, idx_h])
        print(f"  {name}: augmented_low={len(idx_l)} + high={len(idx_h)} -> {len(split[i])}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "array.npy", array)
    np.save(out_dir / "time.npy", time_arr)
    np.save(out_dir / "mask.npy", mask_arr)
    np.save(out_dir / "static.npy", static)
    np.save(out_dir / "gt.npy", gt)
    np.save(out_dir / "split.npy", split, allow_pickle=True)
    sids = aug["sids"].tolist() + high["sids"].tolist()
    smids = aug["smids"] + high["smids"]
    (out_dir / "subject_ids.json").write_text(json.dumps(sids, ensure_ascii=False))
    (out_dir / "sample_ids.json").write_text(json.dumps(smids, ensure_ascii=False))
    (out_dir / "feature_columns.json").write_text(
        json.dumps(aug["feats"], ensure_ascii=False, indent=2))

    meta = json.loads((high_dir / "meta.json").read_text())
    meta.update({
        "target": target,
        "ablation": ablation,
        "missing_group": f"augmented_{ablation}",
        "array_shape": list(array.shape),
        "train": len(split[0]), "val": len(split[1]), "test": len(split[2]),
        "_note": "augmented low(train only) + original low(val/test) + original high",
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    manifest = {
        "target": target,
        "method": ablation,
        "artifact_role": "training_dataset",
        "source_augmented_low": str(aug_dir),
        "source_original_low": str(low_dir),
        "source_original_high": str(high_dir),
        "grouping_axes": list(cfg["cell_axes"]),
        "uses_target_label": bool(cfg["use_label"]),
        "inject_mode": cfg["inject_mode"],
        "target_threshold": 10,
        "composition": {
            "train": "augmented_low + original_high",
            "validation": "original_low + original_high",
            "test": "original_low + original_high",
        },
        "seed": SEED,
        "implementation": "src/augment/build.py:_combine_coformer",
    }
    (out_dir / "augmentation.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  saved -> {out_dir}")
    return out_dir


def combine_augmented_plus_high(format="mtm", ablation="full", target="phq9"):
    """augment된 low + 원본 high를 split별로 결합한 최종 학습 데이터셋을 만든다."""
    if format == "mtm":
        return _combine_mtm(target=target, ablation=ablation)
    if format == "coformer":
        return _combine_coformer(target=target, ablation=ablation)
    raise ValueError(f"unknown format: {format!r} (expected 'mtm' or 'coformer')")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["mtm", "coformer"], default="mtm")
    ap.add_argument("--target", choices=["phq9", "gad7"], default="phq9")
    ap.add_argument("--ablation", choices=list(ABLATIONS.keys()), default="full")
    ap.add_argument("--step", choices=["augment", "combine", "all"], default="all")
    args = ap.parse_args()

    if args.format == "mtm":
        if args.step in ("augment", "all"):
            augment_mtm(ablation=args.ablation, target=args.target)
        if args.step in ("combine", "all"):
            combine_augmented_plus_high("mtm", ablation=args.ablation, target=args.target)
    else:
        if args.step in ("augment", "all"):
            augment_coformer(ablation=args.ablation, target=args.target)
        if args.step in ("combine", "all"):
            combine_augmented_plus_high("coformer", ablation=args.ablation, target=args.target)
