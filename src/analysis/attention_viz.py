"""
attention_viz.py
=================
학습된 모델이 무엇을 보고 판단했는지 시각화.

만드는 것
  CoFormer (GATv2Conv edge attention)
    - attention strip  샘플별 시간축 attention 띠
    - 개별 샘플 attention 상세
    - 정답 / 오답 샘플 attention 비교
    - 우울 심각도(severity)별 attention
    - attention entropy  한 곳에 쏠렸는지 퍼졌는지
    - encoder temporal self-attention + agent aggregation attention
  MTM (Transformer attention 3종)
    - Temporal      시간 슬롯 간
    - TokenMixing   토큰 간
    - Channel       feature 간
    각각 클래스별 히트맵 + 클래스 간 차이 히트맵
    - low결측 모델 vs high결측 모델 비교 (같은 3종, 축만 바꿈)
    - feature importance  CLS 토큰 norm 기반

무슨 분석인가
  - 모델이 어느 feature를 보고 판단하나
      결측이 많은 feature를 무시하는지, 아니면 오히려 결측 여부 자체를 신호로 쓰는지.
  - 결측이 많은 샘플에서 attention이 달라지나
      low/high 모델 비교가 이걸 봄. 다르면 증강이 필요한 근거가 됨.
  - 틀린 샘플에서 무엇을 보고 있었나
      오답의 attention이 정답과 다르면 실패 원인을 짚을 수 있음.

실행
  --mode coformer              CoFormer GAT attention
  --mode mtm                   MTM attention 3종
  --mode mtm_missing_compare   low결측 모델 vs high결측 모델
  --mode mtm_importance        MTM feature importance

  주요 인자: --model_path(ckpt) --data_root --split_path --out_dir
             --model --split --layer

조건
  - 학습 결과(ckpt) 필요
  - mode가 서로 독립이라 한쪽 모델만 학습돼도 그쪽만 돌리면 됨
  - conda 환경이 달라 따로 실행해야 함 (coformer: py3.10/dgl, mtm: py3.11)
  - external/ 원본 코드는 건드리지 않고 src/models 경유로만 접근

미완성
  - plot_temporal_and_aggregation_attention()은 그림 함수만 있음.
    attention 추출부가 옛 module.py/dataset.py 의존이라 이관에서 빠졌고,
    쓰려면 추출 헬퍼를 다시 만들어야 함.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scstats

from .. import paths
from ..vendor import vendor_ctx

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})


# =====================================================================
# MTM side — shared config/checkpoint resolution
#   (MTM/analysis/attention_viz.py + feature_importance.py, near-identical helpers unified)
# =====================================================================

import re

_V2_AUTO_RE = re.compile(r"^custom_v([23])_u(\d+)_w(\d+)_s(\d+)$")

PAM_FEATURE_NAMES = [
    "acc_hand_x", "acc_hand_y", "acc_hand_z", "gyro_hand_x", "gyro_hand_y", "gyro_hand_z",
    "mag_hand_x", "mag_hand_y", "mag_hand_z", "acc_chest_x", "acc_chest_y", "acc_chest_z",
    "acc_ankle_x", "acc_ankle_y", "acc_ankle_z", "temp_hand", "temp_ankle",
]
PAM_LABEL_NAMES = {0: "lying", 1: "sitting", 2: "standing", 3: "walking",
                   4: "running", 5: "cycling", 6: "nordic_walk", 7: "rope_jump"}
_STATIC_LABEL_MAP = {
    "p12": {0: "Survived", 1: "Deceased"}, "p19": {0: "Normal", 1: "Sepsis"}, "pam": PAM_LABEL_NAMES,
}
_RAINDROP_DATAPATH = {
    "p12": "P12data", "p19": "P19data", "pam": "PAMAP2data",
}


def resolve_config(model_name, datapath=None):
    """모델 이름 -> 설정 객체.
    p12/p19/pam        external/mtm 원본 config (데이터 경로만 덮어씀)
    custom_v3_* 등      src.models.mtm.load_run_config()로 생성
    """
    if model_name in _RAINDROP_DATAPATH:
        with vendor_ctx(paths.EXTERNAL_MTM):
            from config.mtm_clsf_config import MTM_P12, MTM_P19, MTM_PAM
            cfg = {"p12": MTM_P12, "p19": MTM_P19, "pam": MTM_PAM}[model_name]()
        cfg.datapath = str(paths.DATA_ROOT / "raindrop" / _RAINDROP_DATAPATH[model_name])
        return cfg
    m = _V2_AUTO_RE.match(model_name)
    if m:
        version, unit, window, stride = m.groups()
        if datapath is None:
            datapath = str(paths.DATA_ROOT / "mtm" / f"mtm_v{version}_unit{unit}_w{window}_s{stride}")
        from ..models.mtm import RunConfigView, load_run_config
        return RunConfigView(load_run_config(datapath))
    raise ValueError(f"Unknown model: '{model_name}'")


def resolve_label_map(model_name):
    if model_name in _STATIC_LABEL_MAP:
        return _STATIC_LABEL_MAP[model_name]
    if _V2_AUTO_RE.match(model_name):
        return {0: "PHQ9 < 10", 1: "PHQ9 >= 10"}
    raise ValueError(f"No label map for: '{model_name}'")


def load_feature_names(model_name, config):
    if model_name == "p12":
        path = paths.DATA_ROOT / "raindrop" / "P12data" / "processed_data" / "ts_params.npy"
        if path.exists():
            return np.load(path, allow_pickle=True).tolist()
    elif model_name == "p19":
        path = paths.DATA_ROOT / "raindrop" / "P19data" / "processed_data" / "labels_ts.npy"
        if path.exists():
            names = np.load(path, allow_pickle=True).tolist()
            return [n for n in names if n != "SepsisLabel"]
    elif model_name == "pam":
        return PAM_FEATURE_NAMES
    else:
        feat_path = Path(config.datapath) / "feature_columns.json"
        if feat_path.exists():
            with open(feat_path, encoding="utf-8") as f:
                return json.load(f)
    return [f"ch_{i}" for i in range(config.num_chn)]


def find_ckpt(dataset_name, split_idx, version=None, logs_dir=None):
    logs_dir = Path(logs_dir) if logs_dir else paths.OUTPUTS_ROOT / "logs"
    ckpt_dir = logs_dir / dataset_name / str(split_idx)
    if version is not None:
        ckpts = sorted((ckpt_dir / f"version_{version}").glob("checkpoints/*.ckpt"))
    else:
        ckpts = sorted(ckpt_dir.glob("version_*/checkpoints/*.ckpt"),
                       key=lambda p: int(p.parts[-3].split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found: {ckpt_dir}")
    return ckpts[-1]


# =====================================================================
# MTM side — 3-attention-type extraction and plotting
#   (MTM/analysis/attention_viz.py)
# =====================================================================

def extract_attention(model, dataloader, device, layer_idx=-1):
    """지정한 mixer layer(0=inp_layer, -1=마지막 mixer)에서 Temporal/TokenMixing/Channel
    attention을 뽑아 클래스별로 평균. temporal은 forward hook으로, mixing/channel은
    layer.mixer/layer.channel의 attn_cache에서 읽는다 (src.models.mtm import 시 적용되는
    패치로 채워짐 — 자세한 이유는 그쪽 _load_mtm_base 참고)."""
    import torch

    from ..models.mtm import attach_attention_hooks

    mtm = model.model
    if layer_idx == 0:
        layer_name = "inp_layer"
        layer = mtm.inp_layer
    elif layer_idx > 0:
        layer_name = f"mixer_{layer_idx - 1}"
        layer = mtm.mixers[layer_idx - 1]
    else:
        layer_name = f"mixer_{len(mtm.mixers) + layer_idx}"
        layer = mtm.mixers[layer_idx]
    captured = attach_attention_hooks(mtm)

    all_temporal, all_mixing, all_channel = [], [], []
    all_labels, all_preds = [], []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, x_mask, t, y, x_static, uids = batch
            x, x_mask, t = x.to(device), x_mask.to(device), t.to(device)
            x_static = x_static.to(device) if x_static is not None else None
            logits = model.model(x, x_mask, t, x_static)
            preds = logits.argmax(-1).cpu()

            t_attn = captured[f"{layer_name}.temporal"].cpu()
            all_temporal.append(t_attn[:, :, 0, 1:])
            m_attn = layer.mixer.attn_cache.cpu()
            all_mixing.append(m_attn[:, :, 0, 1:])
            c_attn = layer.channel.attn_cache.cpu()
            all_channel.append(c_attn[:, 1:, :, :].mean(dim=1))

            all_labels.append(y.numpy())
            all_preds.append(preds.numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)

    def _pad_cat(tensors):
        max_T = max(t.shape[-1] for t in tensors)
        padded = [torch.nn.functional.pad(t, (0, max_T - t.shape[-1]), value=0.0) for t in tensors]
        return torch.cat(padded, dim=0)

    temporal_all = _pad_cat(all_temporal).numpy()
    mixing_all = _pad_cat(all_mixing).numpy()
    channel_all = torch.cat(all_channel, dim=0).numpy()

    classes = sorted(np.unique(labels))
    temporal_by_cls = {c: temporal_all[labels == c].mean(0) for c in classes}
    mixing_by_cls = {c: mixing_all[labels == c].mean(0) for c in classes}
    channel_by_cls = {c: channel_all[labels == c].mean(0) for c in classes}
    return temporal_by_cls, mixing_by_cls, channel_by_cls, labels, preds


def _add_reading_guide(fig, text, y=0.0):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=13, color="#444444",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc", lw=0.8))


def plot_ct_attn(attn_by_cls, feature_names, label_map, save_path, title_prefix, attn_type, vmax=None):
    """Temporal/TokenMixing 공용 플롯: 클래스별 패널 + (이진 분류면) Diff 패널.
    vmax를 넘기면 여러 그림이 같은 컬러 스케일을 공유한다."""
    classes = sorted(attn_by_cls.keys())
    n_cls = len(classes)
    C, T = attn_by_cls[classes[0]].shape
    t_labels = [f"t{i + 1}" for i in range(T)]
    binary = (n_cls == 2)
    n_plots = n_cls + (1 if binary else 0)
    if binary:
        l0, l1 = label_map.get(classes[0], str(classes[0])), label_map.get(classes[1], str(classes[1]))

    fig, axes = plt.subplots(1, n_plots, figsize=(max(10, T * 0.9) * n_plots, max(8, C * 0.4) + 1.2),
                             squeeze=False)
    axes = axes[0]
    if vmax is None:
        vmax = max(v.max() for v in attn_by_cls.values())

    for i, c in enumerate(classes):
        label = label_map.get(c, str(c))
        sns.heatmap(attn_by_cls[c], ax=axes[i], xticklabels=t_labels, yticklabels=feature_names,
                    cmap="YlOrRd", vmin=0, vmax=vmax, linewidths=0.3, linecolor="lightgray",
                    cbar_kws={"label": "Attention weight (0 = ignore, 1 = full focus)"})
        axes[i].set_title(f"Class: {label}", fontsize=13)
        axes[i].set_xlabel("Downsampled Time Step"); axes[i].set_ylabel("Feature (channel)")
        axes[i].tick_params(axis="x", rotation=45)

    if binary:
        diff = attn_by_cls[classes[1]] - attn_by_cls[classes[0]]
        vd = max(np.abs(diff).max(), 1e-8)
        sns.heatmap(diff, ax=axes[-1], xticklabels=t_labels, yticklabels=feature_names, cmap="RdBu_r",
                    vmin=-vd, vmax=vd, center=0, linewidths=0.3, linecolor="lightgray",
                    cbar_kws={"label": f"Attention diff  ({l1}) - ({l0})"})
        axes[-1].set_title(f"Diff: {l1} vs {l0}\nRED = {l1}  |  BLUE = {l0}", fontsize=13)
        axes[-1].set_xlabel("Downsampled Time Step"); axes[-1].set_ylabel("Feature (channel)")
        axes[-1].tick_params(axis="x", rotation=45)

    type_title, type_note = {
        "temporal": ("Temporal Attention  (cls_tok -> time step, per feature)",
                     "How to read: each row = one feature. Darker cell = more attention to that time step."),
        "mixing": ("Token Mixing Attention  (cls_tok -> time step, per channel, after guided imputation)",
                   "Columns = same downsampled time steps as Temporal. Missing positions are replaced with "
                   "the most-attended channel's features — compare with Temporal to see imputation's effect."),
    }[attn_type]
    fig.suptitle(f"{title_prefix}  |  {type_title}", fontsize=13, y=1.01)
    _add_reading_guide(fig, type_note, y=-0.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved : {save_path}")


def plot_channel_attn(attn_by_cls, feature_names, label_map, save_path, title_prefix):
    """ChannelAttn (C,C) 히트맵, 시간/샘플 평균. 행=query feature, 열=key feature."""
    classes = sorted(attn_by_cls.keys())
    n_cls = len(classes)
    C = len(feature_names)
    binary = (n_cls == 2)
    n_plots = n_cls + (1 if binary else 0)
    do_annot = (C <= 20)
    if binary:
        l0, l1 = label_map.get(classes[0], str(classes[0])), label_map.get(classes[1], str(classes[1]))

    fig_sz = max(10, C * 0.45)
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_sz * n_plots, fig_sz + 1.2), squeeze=False)
    axes = axes[0]
    vmax = max(v.max() for v in attn_by_cls.values())

    for i, c in enumerate(classes):
        label = label_map.get(c, str(c))
        sns.heatmap(attn_by_cls[c], ax=axes[i], xticklabels=feature_names, yticklabels=feature_names,
                    cmap="YlOrRd", vmin=0, vmax=vmax, annot=do_annot, fmt=".2f", linewidths=0.3,
                    linecolor="lightgray", cbar_kws={"label": "Attention weight (avg over time & samples)"})
        axes[i].set_title(f"Class: {label}", fontsize=13)
        axes[i].set_xlabel("Key feature  (being attended to)"); axes[i].set_ylabel("Query feature  (attending)")
        axes[i].tick_params(axis="x", rotation=45)

    if binary:
        diff = attn_by_cls[classes[1]] - attn_by_cls[classes[0]]
        vd = max(np.abs(diff).max(), 1e-8)
        sns.heatmap(diff, ax=axes[-1], xticklabels=feature_names, yticklabels=feature_names, cmap="RdBu_r",
                    vmin=-vd, vmax=vd, center=0, annot=do_annot, fmt=".2f", linewidths=0.3,
                    linecolor="lightgray", cbar_kws={"label": f"Attention diff  ({l1}) - ({l0})"})
        axes[-1].set_title(f"Diff: {l1} vs {l0}\nRED = {l1}  |  BLUE = {l0}", fontsize=13)
        axes[-1].set_xlabel("Key feature  (being attended to)"); axes[-1].set_ylabel("Query feature  (attending)")
        axes[-1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"{title_prefix}  |  Channel Attention  (feature x feature, averaged over time)",
                fontsize=13, y=1.01)
    _add_reading_guide(fig, "Cell (A, B) = how much feature A attends to feature B (avg over time & samples).",
                       y=-0.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved : {save_path}")


def generate_mtm_attention_figures(model_name, split=1, version=None, datapath=None, layer=-1, out_dir=None):
    """MTM attention 3종 그림 생성 메인 드라이버.
    체크포인트는 꼭 `src.models.mtm.MTMModule`로 로드해야 함 — pristine
    `tasks.clsf_module.ClassificationModule`은 학습 때 쓴 metric 구조가 없어서
    state_dict가 안 맞음. `RaindropDataModule`은 원본 그대로라 vendor_ctx로 로드."""
    from ..models.mtm import MTMModule

    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "attention_viz" / f"{model_name}_split{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = resolve_config(model_name, datapath)
    label_map = resolve_label_map(model_name)
    ckpt_path = find_ckpt(config.dataset, split, version)
    feature_names = load_feature_names(model_name, config)
    device = "cuda" if _cuda_available() else "cpu"

    with vendor_ctx(paths.EXTERNAL_MTM):
        from data_modules.raindrop import RaindropDataModule
        rdm = RaindropDataModule(config.datapath, split, config.batch_size,
                                  dataset=config.dataset, compact=config.compact)
        test_loader = rdm.test_dataloader()

    model = MTMModule.load_from_checkpoint(
        ckpt_path, model=config.get_model(), forward_fn=config.forward_fn).to(device)
    temporal_by_cls, mixing_by_cls, channel_by_cls, labels, preds = extract_attention(
        model, test_loader, device, layer)

    title_prefix = f"{model_name.upper()}  split{split}"
    prefix = out_dir / f"layer{layer}"
    plot_ct_attn(temporal_by_cls, feature_names, label_map, f"{prefix}_temporal.png", title_prefix, "temporal")
    plot_ct_attn(mixing_by_cls, feature_names, label_map, f"{prefix}_mixing.png", title_prefix, "mixing")
    plot_channel_attn(channel_by_cls, feature_names, label_map, f"{prefix}_channel.png", title_prefix)
    print(f"Done -> {out_dir}")
    return temporal_by_cls, mixing_by_cls, channel_by_cls


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def generate_mtm_attention_missing_compare(runs=None, layer=-1, out_dir=None):
    """같은 3종 attention을 클래스 대신 low결측 모델 vs high결측 모델로 비교.
    `runs`는 {0: {"name","datapath","ckpt","test_f1"}, 1: {...}} 형식이고
    기본값은 DATA_ROOT/OUTPUTS_ROOT 밑 low/high_missing 쌍."""
    import torch

    if runs is None:
        runs = {
            0: {"name": "missing_low", "datapath": str(paths.DATA_ROOT / "mtm" / "low_missing"),
                "ckpt": None, "test_f1": None},
            1: {"name": "missing_high", "datapath": str(paths.DATA_ROOT / "mtm" / "high_missing"),
                "ckpt": None, "test_f1": None},
        }
    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "attention_viz" / "missing_low_vs_high"
    out_dir.mkdir(parents=True, exist_ok=True)

    def weighted_avg(by_cls, labels):
        classes = sorted(by_cls.keys())
        counts = np.array([(labels == c).sum() for c in classes], dtype=np.float64)
        stacked = np.stack([by_cls[c] for c in classes], axis=0)
        return np.average(stacked, axis=0, weights=counts)

    temporal_by_grp, mixing_by_grp, channel_by_grp = {}, {}, {}
    feature_names = None
    device = "cuda" if _cuda_available() else "cpu"

    from ..models.mtm import MTMModule, RunConfigView, load_run_config

    for key, info in runs.items():
        config = RunConfigView(load_run_config(info["datapath"]))
        if feature_names is None:
            feature_names = load_feature_names("custom_v3", config)
        ckpt_path = info["ckpt"] or find_ckpt(config.dataset, 1)

        with vendor_ctx(paths.EXTERNAL_MTM):
            from data_modules.raindrop import RaindropDataModule
            rdm = RaindropDataModule(config.datapath, 1, config.batch_size,
                                      dataset=config.dataset, compact=config.compact)
            test_loader = rdm.test_dataloader()

        model = MTMModule.load_from_checkpoint(
            ckpt_path, model=config.get_model(), forward_fn=config.forward_fn).to(device)
        temporal_by_cls, mixing_by_cls, channel_by_cls, labels, preds = extract_attention(
            model, test_loader, device, layer)
        temporal_by_grp[key] = weighted_avg(temporal_by_cls, labels)
        mixing_by_grp[key] = weighted_avg(mixing_by_cls, labels)
        channel_by_grp[key] = weighted_avg(channel_by_cls, labels)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    label_map = {k: f"{v['name']}" + (f" (test F1={v['test_f1']:.2f})" if v.get("test_f1") else "")
                for k, v in runs.items()}
    title_prefix = "MISSING_LOW vs MISSING_HIGH  split1"
    prefix = out_dir / f"layer{layer}"

    T = next(iter(temporal_by_grp.values())).shape[1]
    if T > 24:
        n_days = (T + 23) // 24
        temporal_vmax = max(v.max() for v in temporal_by_grp.values())
        mixing_vmax = max(v.max() for v in mixing_by_grp.values())
        for d in range(n_days):
            s, e = d * 24, min((d + 1) * 24, T)
            day_title = f"{title_prefix}  (Day {d + 1}: hour {s}-{e})"
            plot_ct_attn({k: v[:, s:e] for k, v in temporal_by_grp.items()}, feature_names, label_map,
                        f"{prefix}_temporal_day{d + 1}.png", day_title, "temporal", vmax=temporal_vmax)
            plot_ct_attn({k: v[:, s:e] for k, v in mixing_by_grp.items()}, feature_names, label_map,
                        f"{prefix}_mixing_day{d + 1}.png", day_title, "mixing", vmax=mixing_vmax)
        plot_channel_attn(channel_by_grp, feature_names, label_map, f"{prefix}_channel.png", title_prefix)
    else:
        plot_ct_attn(temporal_by_grp, feature_names, label_map, f"{prefix}_temporal.png", title_prefix, "temporal")
        plot_ct_attn(mixing_by_grp, feature_names, label_map, f"{prefix}_mixing.png", title_prefix, "mixing")
        plot_channel_attn(channel_by_grp, feature_names, label_map, f"{prefix}_channel.png", title_prefix)
    print(f"Done -> {out_dir}")


# =====================================================================
# MTM side — CLS-token-norm feature importance (feature_importance.py)
# =====================================================================

def extract_importance(model, dataloader, device):
    """feature별 중요도 = 마지막 layer CLS 토큰 임베딩의 norm, 클래스별 평균."""
    import torch

    all_imp, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, x_mask, t, y, x_static, uids = batch
            x, x_mask, t = x.to(device), x_mask.to(device), t.to(device)
            x_static = x_static.to(device) if x_static is not None else None
            _ = model.model(x, x_mask, t, x_static)
            cls = model.model.mixers[-1].cls_tok  # (B, C, D) — per-feature CLS embedding
            imp = cls.norm(dim=-1).cpu().numpy()  # (B, C)
            all_imp.append(imp)
            all_labels.append(y.numpy())
    importance = np.concatenate(all_imp, axis=0)
    labels = np.concatenate(all_labels)
    return importance, labels


def plot_importance_heatmap(importance, labels, feature_names, label_map, save_path, title_prefix):
    classes = sorted(np.unique(labels))
    fig, axes = plt.subplots(1, len(classes), figsize=(max(10, len(feature_names) * 0.3) * len(classes), 6),
                             squeeze=False)
    vmax = importance.max()
    for i, c in enumerate(classes):
        m = importance[labels == c].mean(axis=0, keepdims=True)
        sns.heatmap(m, ax=axes[0][i], xticklabels=feature_names, yticklabels=[label_map.get(c, str(c))],
                    cmap="YlOrRd", vmin=0, vmax=vmax, cbar_kws={"label": "CLS-token norm"})
        axes[0][i].tick_params(axis="x", rotation=90)
    fig.suptitle(f"{title_prefix}  |  Feature importance (CLS-token norm)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved : {save_path}")


def plot_importance_bar(importance, labels, feature_names, label_map, save_path, title_prefix):
    classes = sorted(np.unique(labels))
    x = np.arange(len(feature_names))
    w = 0.8 / len(classes)
    fig, ax = plt.subplots(figsize=(max(12, len(feature_names) * 0.35), 6))
    for i, c in enumerate(classes):
        m = importance[labels == c].mean(axis=0)
        ax.bar(x + i * w - 0.4 + w / 2, m, w, label=label_map.get(c, str(c)))
    ax.set_xticks(x); ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax.set_ylabel("CLS-token norm"); ax.set_title(f"{title_prefix}  |  Feature importance", fontsize=13)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved : {save_path}")


def generate_mtm_feature_importance(model_name, split=1, version=None, datapath=None, out_dir=None):
    """체크포인트는 `src.models.mtm.MTMModule`로 로드 — 이유는
    `generate_mtm_attention_figures` docstring 참고."""
    from ..models.mtm import MTMModule

    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "attention_viz" / f"{model_name}_split{split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = resolve_config(model_name, datapath)
    label_map = resolve_label_map(model_name)
    ckpt_path = find_ckpt(config.dataset, split, version)
    feature_names = load_feature_names(model_name, config)
    device = "cuda" if _cuda_available() else "cpu"

    with vendor_ctx(paths.EXTERNAL_MTM):
        from data_modules.raindrop import RaindropDataModule
        rdm = RaindropDataModule(config.datapath, split, config.batch_size,
                                  dataset=config.dataset, compact=config.compact)
        test_loader = rdm.test_dataloader()

    model = MTMModule.load_from_checkpoint(
        ckpt_path, model=config.get_model(), forward_fn=config.forward_fn).to(device)
    importance, labels = extract_importance(model, test_loader, device)

    title_prefix = f"{model_name.upper()}  split{split}"
    plot_importance_heatmap(importance, labels, feature_names, label_map,
                            out_dir / "feature_importance_heatmap.png", title_prefix)
    plot_importance_bar(importance, labels, feature_names, label_map,
                        out_dir / "feature_importance_bar.png", title_prefix)
    print(f"Done -> {out_dir}")
    return importance, labels


# =====================================================================
# CoFormer side — GATv2Conv edge-attention capture
#   (sw/CoFormer/analysis/attention_utils.py: only AttentionCapturer +
#    extract_all_attention + analyze_and_plot actually exist, see module docstring)
# =====================================================================

class AttentionCapturer:
    """GATv2Conv 레이어마다 attn_drop(softmax 이후 attention weight)과
    forward hook(graph src/dst)을 건다. DGL이 local_scope를 써서 edata['a']를
    직접 못 읽기 때문에 attn_drop 출력을 대신 쓰는 게 공식 우회법."""

    def __init__(self):
        self.graph_info = {}
        self.attn_weights = {}
        self.handles = []

    def register_hooks(self, model):
        count = 0
        for name, module in model.named_modules():
            if type(module).__name__ == "GATv2Conv":
                layer_name = name

                def graph_hook(lname):
                    def fn(mod, inp, out):
                        graph = inp[0]
                        src, dst = graph.edges()
                        self.graph_info[lname] = {"src": src.detach().cpu(), "dst": dst.detach().cpu(),
                                                   "num_nodes": graph.num_nodes()}
                    return fn

                h1 = module.register_forward_hook(graph_hook(layer_name))
                self.handles.append(h1)

                def attn_hook(lname):
                    def fn(mod, inp, out):
                        self.attn_weights[lname] = out.detach().cpu()
                    return fn

                h2 = module.attn_drop.register_forward_hook(attn_hook(layer_name))
                self.handles.append(h2)
                count += 1
                print(f"  Hook: {layer_name} (GATv2Conv + attn_drop)")
        print(f"  Total {count} layers hooked")
        return count

    def get_feature_attention(self, num_agents, num_timesteps, layer_idx=-1):
        """edge attention -> A x A feature-attention 행렬 변환. node=A x T, edge=KNN 기준."""
        import torch

        keys = list(self.attn_weights.keys())
        if not keys:
            return None
        key = keys[layer_idx] if abs(layer_idx) <= len(keys) else keys[-1]
        if key not in self.attn_weights or key not in self.graph_info:
            return None

        attn = self.attn_weights[key]
        info = self.graph_info[key]
        src, dst, total_nodes = info["src"], info["dst"], info["num_nodes"]
        if attn.dim() == 3:
            attn = attn.squeeze(-1)
        if attn.dim() == 2:
            attn = attn.mean(dim=-1)

        A, T = num_agents, num_timesteps
        B = total_nodes // (A * T)
        all_matrices = []
        edges_per_batch = len(src) // B
        for b in range(B):
            start, end = b * edges_per_batch, (b + 1) * edges_per_batch
            s = src[start:end] - b * A * T
            d = dst[start:end] - b * A * T
            s_agent, d_agent = s // T, d // T
            a = attn[start:end]
            mat_flat = torch.zeros(A * A)
            idx = d_agent * A + s_agent
            mat_flat.scatter_add_(0, idx.long(), a.float())
            mat = mat_flat.reshape(A, A)
            row_sum = mat.sum(dim=1, keepdim=True)
            row_sum[row_sum == 0] = 1
            all_matrices.append(mat / row_sum)
        return torch.stack(all_matrices, dim=0)

    def clear(self):
        self.graph_info = {}
        self.attn_weights = {}

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.clear()


def extract_all_attention(model, test_loader, device, num_agents, num_timesteps=336):
    import torch
    from tqdm import tqdm

    capturer = AttentionCapturer()
    n_hooks = capturer.register_hooks(model)
    if n_hooks == 0:
        print("  No GATv2Conv found!")
        return None, None

    all_attn, all_gt = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting attention"):
            array = batch["data"].to(device)
            time_t = batch["time"].to(device)
            static = batch["static"].to(device)
            mask = batch["mask"].to(device)
            gt = batch["gt"].squeeze(1).long()
            _ = model(array, time_t, mask, static)
            attn_matrix = capturer.get_feature_attention(num_agents, num_timesteps, layer_idx=-1)
            if attn_matrix is not None:
                all_attn.append(attn_matrix)
                all_gt.append(gt)
            capturer.clear()
    capturer.remove_hooks()

    if not all_attn:
        return None, None
    return torch.cat(all_attn, dim=0).numpy(), torch.cat(all_gt, dim=0).numpy()


def analyze_and_plot(attn, labels, features, output_dir):
    """control/patient/diff 3패널 히트맵 + top-20 feature쌍 bar + feature 중요도 bar
    + 통계 JSON 저장."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normal = attn[labels == 0].mean(axis=0)
    patient = attn[labels == 1].mean(axis=0)
    diff = patient - normal

    n = len(features)
    tick_fs = max(4, min(9, int(180 / n)))
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    vmin, vmax = min(normal.min(), patient.min()), max(normal.max(), patient.max())
    for ax, data, title in [(axes[0], normal, "Control (Normal)"), (axes[1], patient, "Patient (Depressed)")]:
        sns.heatmap(data, xticklabels=features, yticklabels=features, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                    ax=ax, cbar_kws={"label": "Attention"})
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fs)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs)
    abs_max = max(np.abs(diff).max(), 1e-8)
    sns.heatmap(diff, xticklabels=features, yticklabels=features, cmap="RdBu_r", center=0, vmin=-abs_max,
                vmax=abs_max, ax=axes[2], cbar_kws={"label": "Difference"})
    axes[2].set_title("Difference (Patient - Control)", fontsize=14, weight="bold")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=90, fontsize=tick_fs)
    axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0, fontsize=tick_fs)
    plt.tight_layout()
    plt.savefig(output_dir / "attention_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    diff_flat = diff.flatten()
    top_k = 20
    top_idx = np.argsort(np.abs(diff_flat))[-top_k:][::-1]
    pairs = np.unravel_index(top_idx, diff.shape)
    vals = diff_flat[top_idx]
    labels_text = [f"{features[i]} <- {features[j]}" for i, j in zip(pairs[0], pairs[1])]
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    ax.barh(range(len(labels_text)), vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(labels_text))); ax.set_yticklabels(labels_text, fontsize=10)
    ax.set_xlabel("Attention Difference (Patient - Control)")
    ax.set_title(f"Top {top_k} Feature Pairs\nRed = higher in Patient | Blue = higher in Normal",
                 fontsize=13, weight="bold")
    ax.axvline(0, color="black", linewidth=0.8); ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "top_differences.png", dpi=150, bbox_inches="tight")
    plt.close()

    normal_imp, patient_imp = normal.mean(axis=0), patient.mean(axis=0)
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(features)); w = 0.35
    ax.bar(x - w / 2, normal_imp, w, label="Normal", color="#3498db", edgecolor="black", linewidth=0.3)
    ax.bar(x + w / 2, patient_imp, w, label="Patient", color="#e74c3c", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(features, rotation=90, fontsize=8)
    ax.set_ylabel("Mean Attention Received"); ax.set_title("Feature Importance: Normal vs Patient", weight="bold")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    stats = {
        "num_features": len(features), "features": features,
        "normal_samples": int((labels == 0).sum()), "patient_samples": int((labels == 1).sum()),
        "control": {"mean": float(normal.mean()), "std": float(normal.std())},
        "patient": {"mean": float(patient.mean()), "std": float(patient.std())},
        "top_pairs": [{"pair": labels_text[i], "diff": float(vals[i])} for i in range(len(vals))],
    }
    with open(output_dir / "attention_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  saved: {output_dir}")


def plot_attention_strips(attn, gt, features, output_dir):
    """같은 feature x feature attention을 다른 방식 3개로: 전체 strip, feature별
    1행씩 쌓은 strip, diff strip. (원본: plot_attention_strips.py)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normal, patient = attn[gt == 0].mean(axis=0), attn[gt == 1].mean(axis=0)
    diff = patient - normal
    vmin, vmax = min(normal.min(), patient.min()), max(normal.max(), patient.max())
    abs_max = max(np.abs(diff).max(), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(28, 16))
    for ax, data, title, cmap, vm in [
        (axes[0], normal, "Control (Normal)", "YlOrRd", (vmin, vmax)),
        (axes[1], patient, "Patient (Depressed)", "YlOrRd", (vmin, vmax)),
        (axes[2], diff, "Difference (Patient - Control)", "RdBu_r", (-abs_max, abs_max)),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vm[0], vmax=vm[1], interpolation="nearest")
        ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=7)
        ax.set_xticks(range(len(features))); ax.set_xticklabels(features, rotation=90, fontsize=7)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_ylabel("Query Feature (attends to ->)"); ax.set_xlabel("Key Feature (attended)")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle(f"Feature-wise Attention Score Map ({len(features)} features)", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_strip_all.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {output_dir / 'attention_strip_all.png'}")

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(20, max(10, n * 0.75)), gridspec_kw={"hspace": 0.05})
    for i, (ax, feat) in enumerate(zip(axes, features)):
        combined = np.vstack([normal[i:i + 1, :], patient[i:i + 1, :]])
        im = ax.imshow(combined, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["N", "P"], fontsize=6)
        ax.set_ylabel(feat, fontsize=7, rotation=0, ha="right", va="center")
        if i == n - 1:
            ax.set_xticks(range(n)); ax.set_xticklabels(features, rotation=90, fontsize=6)
        else:
            ax.set_xticks([])
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Attention")
    fig.suptitle("Per-Feature Attention Strip\nTop=Normal, Bottom=Patient", fontsize=14, weight="bold", y=0.98)
    plt.savefig(output_dir / "attention_strip_per_feature.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {output_dir / 'attention_strip_per_feature.png'}")

    fig, ax = plt.subplots(figsize=(20, 14))
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, interpolation="nearest")
    ax.set_yticks(range(n)); ax.set_yticklabels(features, fontsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels(features, rotation=90, fontsize=8)
    ax.set_title("Attention Difference Map (Patient - Control)\nRed = Patient attends more | Blue = Normal attends more",
                 fontsize=14, weight="bold")
    ax.set_ylabel("Query Feature"); ax.set_xlabel("Key Feature")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Difference")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_diff_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {output_dir / 'attention_diff_map.png'}")


def plot_attention_individual(attn, gt, features, output_dir, nrows=8, ncols=6):
    """feature 개수만큼 개별 attention 그리드 타일 — 각 타일에 top-3 attended
    feature 표시. (원본: plot_attention_individual.py)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normal, patient = attn[gt == 0].mean(axis=0), attn[gt == 1].mean(axis=0)
    diff = patient - normal
    n = len(features)
    side = int(np.ceil(np.sqrt(n)))
    vmin_all, vmax_all = min(normal.min(), patient.min()), max(normal.max(), patient.max())
    abs_max = max(np.abs(diff).max(), 1e-8)

    def _grid_panel(mat, cmap, vmin, vmax, title, fname, is_diff=False):
        fig, axes = plt.subplots(nrows, ncols, figsize=(24, 28))
        for idx in range(nrows * ncols):
            ax = axes[idx // ncols][idx % ncols]
            if idx >= n:
                ax.set_visible(False)
                continue
            row = mat[idx]
            padded = np.full(side * side, 0.0 if is_diff else np.nan)
            padded[:n] = row
            grid = padded.reshape(side, side)
            ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
            ax.set_title(features[idx], fontsize=8, weight="bold", pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            top3 = np.argsort(np.abs(row) if is_diff else row)[-3:][::-1]
            fmt = "{:+.3f}" if is_diff else "{:.2f}"
            txt = "\n".join(f"{features[j][:8]}=" + fmt.format(row[j]) for j in top3)
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=5, va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        fig.suptitle(title, fontsize=14, weight="bold")
        fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.15)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                     cax=cbar_ax, label="Attention" if not is_diff else "Difference")
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved: {output_dir / fname}")

    _grid_panel(normal, "YlOrRd", vmin_all, vmax_all, "Per-Feature Attention Map — Control (Normal)",
               "attention_individual_normal.png")
    _grid_panel(patient, "YlOrRd", vmin_all, vmax_all, "Per-Feature Attention Map — Patient (Depressed)",
               "attention_individual_patient.png")
    _grid_panel(diff, "RdBu_r", -abs_max, abs_max,
               "Per-Feature Attention Difference (Patient - Control)",
               "attention_individual_diff.png", is_diff=True)


def plot_attention_correct_vs_wrong(attn, gt, preds, features, output_dir):
    """정답/오답 샘플 간 attention 차이 비교. (원본: analyze_attention_deep.py part 1)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tick_fs = 7
    correct_mask, wrong_mask = (preds == gt), (preds != gt)
    panels = []
    for label_val, name in [(0, "Normal"), (1, "Patient")]:
        for mask, mname in [(correct_mask, "Correct"), (wrong_mask, "Wrong")]:
            sel = (gt == label_val) & mask
            data = attn[sel].mean(0) if sel.sum() > 0 else np.zeros(attn.shape[1:])
            panels.append((data, f"{name} - {mname} (n={int(sel.sum())})"))

    fig, axes = plt.subplots(2, 2, figsize=(24, 22))
    vmin = min(d.min() for d, _ in panels)
    vmax = max(d.max() for d, _ in panels)
    for ax, (data, title) in zip(axes.ravel(), panels):
        sns.heatmap(data, xticklabels=features, yticklabels=features, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                    ax=ax, cbar_kws={"label": "Attention"})
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fs)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs)
    fig.suptitle("Attention Score Map: Correct vs Wrong Classification", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_correct_vs_wrong.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {output_dir / 'attention_correct_vs_wrong.png'}")

    normal_correct, normal_wrong, patient_correct, patient_wrong = (p[0] for p in panels)
    fig, axes = plt.subplots(1, 2, figsize=(24, 11))
    for ax, diff_data, title in [(axes[0], normal_correct - normal_wrong, "Normal: Correct - Wrong"),
                                  (axes[1], patient_correct - patient_wrong, "Patient: Correct - Wrong")]:
        am = max(np.abs(diff_data).max(), 1e-8)
        sns.heatmap(diff_data, xticklabels=features, yticklabels=features, cmap="RdBu_r", center=0, vmin=-am,
                    vmax=am, ax=ax, cbar_kws={"label": "Difference"})
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fs)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs)
    fig.suptitle("Attention Difference: RIGHT vs WRONG", fontsize=15, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_correct_minus_wrong.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {output_dir / 'attention_correct_minus_wrong.png'}")


def plot_attention_entropy(attn, gt, preds, features, output_dir):
    """행(query feature)별 attention entropy — 낮으면 집중, 높으면 분산.
    (원본: analyze_attention_deep.py part 3)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    correct_mask, wrong_mask = (preds == gt), (preds != gt)

    def _entropy(attn_matrix):
        eps = 1e-10
        row_sum = attn_matrix.sum(axis=2, keepdims=True)
        row_sum[row_sum == 0] = 1
        probs = attn_matrix / row_sum
        return -np.sum(probs * np.log(probs + eps), axis=2)

    per_row = _entropy(attn)
    entropy_all = per_row.mean(axis=1)
    entropy_normal, entropy_patient = entropy_all[gt == 0], entropy_all[gt == 1]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    ax = axes[0]
    bins = np.linspace(entropy_all.min(), entropy_all.max(), 30)
    ax.hist(entropy_normal, bins=bins, alpha=0.6, label=f"Normal (n={len(entropy_normal)})", color="#3498db")
    ax.hist(entropy_patient, bins=bins, alpha=0.6, label=f"Patient (n={len(entropy_patient)})", color="#e74c3c")
    t_stat, p_val = scstats.ttest_ind(entropy_normal, entropy_patient, equal_var=False)
    ax.set_title(f"Attention Entropy Distribution\nWelch t={t_stat:.2f}, p={p_val:.2e}", weight="bold")
    ax.set_xlabel("Entropy (higher = more spread)"); ax.set_ylabel("# of samples"); ax.legend()

    ax = axes[1]
    groups = {"Normal\nCorrect": entropy_all[(gt == 0) & correct_mask], "Normal\nWrong": entropy_all[(gt == 0) & wrong_mask],
              "Patient\nCorrect": entropy_all[(gt == 1) & correct_mask], "Patient\nWrong": entropy_all[(gt == 1) & wrong_mask]}
    bp = ax.boxplot(list(groups.values()), tick_labels=list(groups.keys()), patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["#3498db", "#85c1e9", "#e74c3c", "#f1948a"]):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel("Entropy"); ax.set_title("Entropy by Group & Correctness", weight="bold")

    ax = axes[2]
    normal_fe, patient_fe = per_row[gt == 0].mean(0), per_row[gt == 1].mean(0)
    x = np.arange(len(features)); w = 0.35
    ax.bar(x - w / 2, normal_fe, w, label="Normal", color="#3498db")
    ax.bar(x + w / 2, patient_fe, w, label="Patient", color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(features, rotation=90, fontsize=6)
    ax.set_ylabel("Mean Entropy"); ax.set_title("Per-Feature Attention Entropy", weight="bold"); ax.legend()

    fig.suptitle("Attention Entropy Analysis", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_entropy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {output_dir / 'attention_entropy.png'}")


def plot_attention_by_severity(attn, gt, features, output_dir, phq9_scores=None):
    """PHQ9 중증도 그룹별 attention 평균. 연속 점수가 없으면 정상/우울 이진
    분류로 대체. (원본: analyze_attention_deep.py part 2)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if phq9_scores is not None and len(phq9_scores) == len(gt):
        severity_groups = {"Minimal (0-4)": (phq9_scores >= 0) & (phq9_scores < 5),
                            "Mild (5-9)": (phq9_scores >= 5) & (phq9_scores < 10),
                            "Moderate (10-14)": (phq9_scores >= 10) & (phq9_scores < 15),
                            "Severe (15+)": phq9_scores >= 15}
    else:
        severity_groups = {"Normal (PHQ9<10)": gt == 0, "Depressed (PHQ9>=10)": gt == 1}

    n_groups = len(severity_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(10 * n_groups, 10))
    axes = [axes] if n_groups == 1 else axes
    group_means, vmax_s = {}, 0
    for name, mask in severity_groups.items():
        if mask.sum() > 0:
            group_means[name] = attn[mask].mean(0)
            vmax_s = max(vmax_s, group_means[name].max())
    for ax, (name, mask) in zip(axes, severity_groups.items()):
        if mask.sum() == 0:
            ax.set_visible(False)
            continue
        sns.heatmap(group_means[name], xticklabels=features, yticklabels=features, cmap="YlOrRd", vmin=0,
                    vmax=vmax_s, ax=ax, cbar_kws={"label": "Attention"})
        ax.set_title(f"{name} (n={mask.sum()})", fontsize=13, weight="bold")
    fig.suptitle("Attention Score Map by PHQ9 Severity", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_by_severity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {output_dir / 'attention_by_severity.png'}")


# =====================================================================
# CoFormer side — driver: build model + test loader, run everything
#   (run_attention.py + attention_dataloader.py)
# =====================================================================

def build_coformer_test_loader(data_root, split_path, batch_size=2, num_workers=4):
    """attention 분석용 test-only DataLoader. 학습 때와 같은
    medical_dataloader/seq_collate_irregular(src.models.coformer)를 쓴다 —
    vendor dataloader를 직접 import하지 않음."""
    from torch.utils.data import DataLoader

    from ..models.coformer import medical_dataloader, seq_collate_irregular

    test_dset = medical_dataloader(root=str(data_root), split_path=str(split_path), training=False)
    loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=seq_collate_irregular, pin_memory=True)
    features = json.load(open(Path(data_root) / "feature_columns.json"))
    return loader, features


def _load_coformer_backbone(model_path, num_agents, device, num_neighbors=30,
                             N=8, d_model=256, d_ff=256, h=8, dropout=0.1,
                             agent_encoding_dim=32, static_dim=4):
    """src.models.coformer.build_backbone으로 CoFormer 백본을 만들고 체크포인트
    weight를 로드. static_dim=4는 실제 학습 값(성별/나이/키/몸무게) —
    wandb config.yaml로 확인함."""
    import torch

    from ..models.coformer import build_backbone

    model = build_backbone(src_vocab=1, tgt_vocab=2, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout,
                           num_agents=num_agents, num_neighbors=num_neighbors,
                           agent_encoding_dim=agent_encoding_dim, static_dim=static_dim)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_dict"] if "model_dict" in ckpt else ckpt)
    return model.to(device).eval()


def generate_coformer_attention_figures(model_path, data_root, split_path, out_dir=None,
                                        batch_size=2, num_timesteps=336, gt_for_severity=None):
    """메인 드라이버: 모델+test loader 준비 → GATv2Conv edge attention 추출 →
    CoFormer attention 그림 세트 전체 생성(비교/top차이/중요도/strip/개별그리드/
    정답오답/entropy/중증도)."""
    import torch

    out_dir = Path(out_dir) if out_dir else paths.OUTPUTS_ROOT / "analysis" / "attention_viz" / "coformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if _cuda_available() else "cpu"

    test_loader, features = build_coformer_test_loader(data_root, split_path, batch_size)
    num_agents = len(features)
    model = _load_coformer_backbone(model_path, num_agents, device)

    print(f"Attention Analysis — GATv2Conv Edge Attention  |  agents={num_agents}")
    attn, gt = extract_all_attention(model, test_loader, device, num_agents, num_timesteps)
    if attn is None:
        print("Attention capture failed. Check model structure.")
        return None

    np.save(out_dir / "attn_all.npy", attn)
    np.save(out_dir / "gt_all.npy", gt)
    analyze_and_plot(attn, gt, features, out_dir)
    plot_attention_strips(attn, gt, features, out_dir)
    plot_attention_individual(attn, gt, features, out_dir)
    plot_attention_by_severity(attn, gt, features, out_dir, phq9_scores=gt_for_severity)

    # correct-vs-wrong / entropy need model predictions, run one more pass
    all_pred = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["data"].to(device), batch["time"].to(device),
                        batch["mask"].to(device), batch["static"].to(device))
            all_pred.append(torch.argmax(torch.softmax(out, dim=-1), dim=-1).cpu())
    preds = torch.cat(all_pred).numpy()
    if len(preds) == len(gt):
        plot_attention_correct_vs_wrong(attn, gt, preds, features, out_dir)
        plot_attention_entropy(attn, gt, preds, features, out_dir)
    else:
        print(f"  preds({len(preds)}) != gt({len(gt)}), skipping correct-vs-wrong/entropy "
              f"(dataloader order likely changed between passes)")

    print(f"Done -> {out_dir}")
    return attn, gt


# =====================================================================
# CoFormer side — inference-for-LMM CSV export
#   (MTM/analysis/coformer_infer_for_lmm.py, in full)
#
# This must run in the `coformer` conda env (dgl etc). The original script's
# own docstring explained why it did two separate sys.path.insert calls (MTM
# root first, then CoFormer root) instead of a clean import: both the MTM and
# CoFormer original repos have top-level `analysis`/`tasks` packages, so
# import order on sys.path determined which one won. That problem doesn't
# exist here — `src.models.coformer` already handles the vendor_ctx isolation
# for anything touching `external/coformer` internally, and `..augment.packing`/
# `..augment.grouping` are this project's own properly-namespaced modules — so
# no sys.path manipulation is needed at all in this port.
# =====================================================================

# Actual training-run hyperparameters (verified against wandb config.yaml for
# the real checkpoints this was run against, 2026-07-21: cbiqdsa8=baseline,
# k3smegtp=augmented) — matches train_robotai.py's CLI defaults, and
# static_dim=4 matches src.models.coformer.build_backbone's default (see its
# docstring for the static_dim=6→9→4 correction history).
COFORMER_LMM_MODEL_KW = dict(input_dim=1, output_dim=2, num_layers=8, heads=8,
                             d_model=256, d_ff=256, dropout=0.1, num_neighbors=30,
                             agent_encoding_dim=32, static_dim=4)


def _report_test_split_mismatch(strategy_root, canonical_root):
    """strategy 자신의 data_root와 canonical(단일 비증강 test source) 사이 test 구성 차이를
    stdout에 보고만 한다(중단하지 않음) — 작업 0821 B: 통일 전에는 strategy마다 test
    population이 실제로 달랐다는 근거를 남기기 위함.

    subject 집합뿐 아니라 window(sample_id) 집합까지 비교한다 — grouping.
    split_coformer_low_high()의 window-count-matching은 subject 전체가 아니라 개별 window를
    골라 서브샘플하므로, subject 집합은 같아도 실제 test window 내용이 다를 수 있다(교수님
    확인 요청, 0821: augmented_* 전략의 test가 실제로 low.test ∪ high.test == baseline.test
    였는지는 subject 집합만으론 답이 안 나옴 — low/high가 이미 각각 baseline.test의
    window-count-matched 부분집합이라, 그 둘을 합쳐도 low/high의 test window 수가 원래
    같지 않았다면 baseline.test보다 작다).

    확인된 사실(코드 경로 추적, src/augment/grouping.py + src/augment/build.py):
      - low_missing/high_missing: grouping.split_coformer_low_high()가 각 split(train/val/
        test) 안에서 low/high window 수를 맞추려고 많은 쪽을 랜덤 서브샘플한다. 즉
        low_missing.test / high_missing.test는 baseline.test의 진부분집합.
      - augmented_{uniform,age,age_sex,full}: build._combine_coformer()가 val/test를
        저 low_missing.test ∪ high_missing.test로 이어붙인다(train만 augmented-low).
    그래서 strategy별 자기 own data_root로 추론하면 strategy마다 다른 population에서
    평가하게 되어 strategy 간(특히 baseline vs augmented_*) 비교가 무효화된다."""
    try:
        own_split = np.load(strategy_root / "split.npy", allow_pickle=True)
        own_sids = np.array(json.load(open(strategy_root / "subject_ids.json")))
        own_smids = json.load(open(strategy_root / "sample_ids.json"))
        own_test_idx = own_split[2]
        own_test_subjects = set(own_sids[own_test_idx].astype(str).tolist())
        own_test_windows = {own_smids[i] for i in own_test_idx}

        can_split = np.load(canonical_root / "split.npy", allow_pickle=True)
        can_sids = np.array(json.load(open(canonical_root / "subject_ids.json")))
        can_smids = json.load(open(canonical_root / "sample_ids.json"))
        can_test_idx = can_split[2]
        can_test_subjects = set(can_sids[can_test_idx].astype(str).tolist())
        can_test_windows = {can_smids[i] for i in can_test_idx}
    except FileNotFoundError as e:
        print(f"[!] test split 비교 건너뜀 (파일 없음: {e})")
        return

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


def generate_coformer_infer_for_lmm(ckpt_path, strategy, target, seed=None, test_data_root=None,
                                    cfg_id="trans_medical_missing", cfg_dir=None,
                                    out_dir=None, gpu=0, missrate_band=None):
    """학습된 CoFormer 체크포인트를, **모든 strategy가 공유하는 단일 비증강 테스트셋**에
    돌려 window별 CSV를 저장한다(0821 교수님 지시 — y를 확률에서 ΔNLL로 바꾸는 재설계).
    `label_compare.run_lmm_effect_size_coformer`/`build_strategy_long_df`/신규
    `build_lmm_input.py`가 이 CSV를 읽는다.

    테스트셋 통일(작업 B — 코드 경로 추적으로 확인한 문제, `_report_test_split_mismatch`
    docstring 참고): strategy별 자기 own data_root(DATA_ROOT/coformer/{target}/datasets/
    {strategy})의 split.npy를 그대로 쓰면 strategy마다 test population 구성이 달랐다
    (low_missing/high_missing은 baseline의 window-count-matched 부분집합, augmented_*는
    low.test ∪ high.test). 그래서 ckpt가 어느 data_root에서 학습됐든 항상
    test_data_root(기본값: DATA_ROOT/coformer/{target}/datasets/baseline — 최초 전처리
    산출물이자 모든 strategy의 test 상위집합)의 test split만 사용해 물리적으로 동일한
    window 집합으로 평가한다.

    레거시 재현: 이 통일 전 결과(기존 그림/표)를 그대로 다시 뽑아야 하면 test_data_root에
    strategy 자기 own data_root(DATA_ROOT/coformer/{target}/datasets/{strategy})를 그대로
    넘기면 된다 — 그러면 이 fix 이전과 동일하게 그 데이터셋 own test split만 사용한다(신규
    컬럼 추가 외에는 수치가 동일). test_data_root를 생략(기본값=baseline)했을 때만 통일된
    평가가 적용된다.

    strategy: 파일명(coformer_infer_{strategy}.csv)과 출력 CSV의 strategy 컬럼에 씀 —
    DATA_ROOT/coformer/{target}/datasets/{strategy} 폴더명과 일치시킬 것(baseline 포함,
    /tmp/robotai_coformer_rebuild_queue.sh의 DATASETS 목록 참고).
    missrate_band: {subject_id: "low"|"high"} 매핑을 미리 계산해 여러 strategy 호출에서
    재사용하고 싶으면 넘긴다(비워두면 missing_rate.subject_missrate_band()로 DATA_ROOT/mtm/
    {target}/datasets/baseline 기준 매번 새로 계산 — grouping.split_low_high가
    group_membership.json을 만든 것과 동일 데이터/계산이라 diff=0. MTM 쪽
    (label_compare.build_lmm_performance_df)도 기본값이 같은 소스라 자동으로 band가
    맞는다 — subject 결측률 band는 모델과 무관한 인구 속성이다.

    출력 컬럼: subject_id/window_id/strategy/y_true/p_true_class/nll(신규, LMM 재설계용,
    nll = -log(clip(p_true_class, 1e-7, 1-1e-7))) + missrate_band(작업 C) +
    subject/patient/correct/prob_true_class/missrate/missrate_{block}(기존 컬럼 유지 —
    label_compare.py의 구버전 파이프라인(build_lmm_coformer_df/build_strategy_long_df)과
    호환). missrate_{block} 컬럼명은 grouping.BLOCKS 키 그대로 사용 — label_compare.
    build_lmm_performance_df와 반드시 일치해야 함(두 conda env라 런타임에 이름 불일치를
    못 잡음)."""
    import torch

    from .. import paths as _paths
    from ..augment.grouping import BLOCKS, compute_sample_missrate
    from ..augment.packing import parse_wstart_min, unpack_dense
    from ..models.coformer import CoFormerModule, RobotAIConfig, build_backbone
    from .missing_rate import subject_missrate_band

    if target not in {"phq9", "gad7"}:
        raise ValueError(f"unsupported target: {target!r}")
    test_data_root = (Path(test_data_root) if test_data_root else
                      _paths.DATA_ROOT / "coformer" / target / "datasets" / "baseline")
    cfg_dir = Path(cfg_dir) if cfg_dir else _paths.REPO_ROOT / "configs" / "coformer"
    out_dir = Path(out_dir) if out_dir else _paths.OUTPUTS_ROOT / "analysis" / "lmm_effect_size"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print(f"[*] strategy={strategy!r} target={target!r} — test windows sourced from single "
          f"canonical data_root (all strategies share this): {test_data_root}")
    strategy_root = _paths.DATA_ROOT / "coformer" / target / "datasets" / strategy
    if strategy_root.exists() and strategy_root != test_data_root:
        _report_test_split_mismatch(strategy_root, test_data_root)

    print(f"[*] loading packed test data: {test_data_root}")
    array = np.load(test_data_root / "array.npy")
    time_arr = np.load(test_data_root / "time.npy")
    mask = np.load(test_data_root / "mask.npy")
    static = np.load(test_data_root / "static.npy")
    gt = np.load(test_data_root / "gt.npy")
    split = np.load(test_data_root / "split.npy", allow_pickle=True)
    feats = json.load(open(test_data_root / "feature_columns.json"))
    sids = np.array(json.load(open(test_data_root / "subject_ids.json")))
    smids = json.load(open(test_data_root / "sample_ids.json"))
    meta = json.loads((test_data_root / "meta.json").read_text())
    unit_minutes, T = meta["unit_minutes"], meta["window_slots"]

    test_idx = split[2]
    print(f"  test windows: {len(test_idx)}")

    if missrate_band is None:
        mtm_baseline_root = _paths.DATA_ROOT / "mtm" / target / "datasets" / "baseline"
        print(f"[*] computing subject missrate band from MTM baseline {mtm_baseline_root} "
              f"(missing_rate.subject_missrate_band — must match grouping.split_low_high's "
              f"own group_membership.json exactly, diff=0 verified via verify_missrate_band.py) ...")
        missrate_band = subject_missrate_band(mtm_baseline_root, target)["missrate_band"].to_dict()

    # ── missrate: packed -> dense(N,T,C) NaN-grid -> compute_sample_missrate ──
    wstart_min = parse_wstart_min([smids[i] for i in test_idx])
    dense = unpack_dense(array[test_idx], time_arr[test_idx], mask[test_idx], wstart_min, T, unit_minutes)
    M = compute_sample_missrate(np.isnan(dense), feats, unit_minutes)
    win_missrate = M.mean(axis=1)
    M_df = pd.DataFrame(M, columns=feats)
    block_missrate = {}
    for bn, bfeats in BLOCKS.items():
        cols = [f for f in bfeats if f in feats]
        block_missrate[f"missrate_{bn}"] = M_df[cols].mean(axis=1).values

    # ── model load + test inference ──
    cfg = RobotAIConfig(cfg_id, cfg_dir)
    num_agents = array.shape[1]  # channel count, same formula as train_robotai.py
    backbone = build_backbone(src_vocab=COFORMER_LMM_MODEL_KW["input_dim"],
                              tgt_vocab=COFORMER_LMM_MODEL_KW["output_dim"],
                              N=COFORMER_LMM_MODEL_KW["num_layers"], d_model=COFORMER_LMM_MODEL_KW["d_model"],
                              d_ff=COFORMER_LMM_MODEL_KW["d_ff"], h=COFORMER_LMM_MODEL_KW["heads"],
                              dropout=COFORMER_LMM_MODEL_KW["dropout"], num_agents=num_agents,
                              num_neighbors=COFORMER_LMM_MODEL_KW["num_neighbors"],
                              agent_encoding_dim=COFORMER_LMM_MODEL_KW["agent_encoding_dim"],
                              static_dim=COFORMER_LMM_MODEL_KW["static_dim"])

    class _Args:
        pass
    fake_args = _Args()
    for k, v in COFORMER_LMM_MODEL_KW.items():
        setattr(fake_args, k, v)

    print(f"[*] loading checkpoint: {ckpt_path}")
    model = CoFormerModule.load_from_checkpoint(str(ckpt_path), model=backbone, cfg=cfg, args=fake_args)
    model.to(device).eval()

    preds, labels, prob_true = [], [], []
    bs = 32
    with torch.no_grad():
        for s in range(0, len(test_idx), bs):
            idx = test_idx[s:s + bs]
            batch = {
                "data": torch.from_numpy(array[idx]).float().to(device),
                "time": torch.from_numpy(time_arr[idx]).float().to(device),
                "static": torch.from_numpy(static[idx]).float().to(device),
                "mask": torch.from_numpy(mask[idx]).int().to(device),
                "gt": torch.from_numpy(gt[idx]).float().to(device),
            }
            logits, y = model._forward(batch)
            probs = torch.softmax(logits, dim=-1)
            preds.append(logits.argmax(-1).cpu().numpy())
            labels.append(y.cpu().numpy())
            prob_true.append(probs.gather(1, y.long().unsqueeze(1)).squeeze(1).cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    prob_true = np.concatenate(prob_true)
    assert len(preds) == len(test_idx) == len(win_missrate)

    subject_id = sids[test_idx].astype(str)
    nll = -np.log(np.clip(prob_true, 1e-7, 1 - 1e-7))
    df = pd.DataFrame({
        # 신규 스키마 (0821 재설계 — build_lmm_input.py가 읽음)
        "subject_id": subject_id,
        "window_id": [smids[i] for i in test_idx],
        "strategy": strategy,
        "y_true": labels.astype(int),
        "p_true_class": prob_true,
        "nll": nll,
        # 기존 스키마 유지 (label_compare.py 구버전 파이프라인 호환)
        "subject": subject_id,
        "patient": labels.astype(int),
        "correct": (preds == labels).astype(int),
        "prob_true_class": prob_true,
        "missrate": win_missrate,
        **block_missrate,
    })
    df["missrate_band"] = df["subject_id"].map(missrate_band)
    n_unmatched = df["missrate_band"].isna().sum()
    if n_unmatched:
        print(f"[!] missrate_band: {n_unmatched}개 window의 subject가 band 매핑에 없음 "
              f"(test_data_root 인구 집합 밖) — NaN으로 둠")
    if seed is not None:
        df["seed"] = seed  # ΔNLL을 같은 seed끼리 짝짓기 위함(교수님 지시, 0821) — build_lmm_input.py가 읽음

    out_path = out_dir / (f"coformer_infer_{strategy}_seed{seed}.csv" if seed is not None
                          else f"coformer_infer_{strategy}.csv")
    df.to_csv(out_path, index=False)
    print(f"[*] n_windows={len(df)} n_subjects={df['subject_id'].nunique()} "
          f"patient_windows={int(df.y_true.sum())} test_acc={(df.correct.mean()):.4f} "
          f"mean_nll={df['nll'].mean():.4f}")
    print(f"  saved: {out_path}")
    return out_path


# =====================================================================
# CoFormer side — temporal + agent-aggregation attention
#   (analysis/v2_72/extra_attention.py — a different attention path: the encoder's
#    own temporal self-attention and the agent-aggregation MultiHeadedAttention,
#    not the GATv2Conv channel attention above)
# =====================================================================

def plot_temporal_and_aggregation_attention(temporal_p, temporal_c, aggre_p, aggre_c,
                                             feature_names, out_dir, day_ticks=None):
    """Patient/Control/Diff 3패널: (a) temporal attention (feature x time),
    (b) agent-aggregation attention (feature x feature).
    그림 함수만 있고 추출부는 미구현 — 자세한 이유는 파일 상단 '미완성' 절 참고."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(feature_names)

    def _plot_triple(p, c, title, fname, xlabel, figsize=(60, 8)):
        diff = p - c
        vmax = max(abs(p).max(), abs(c).max())
        vlim = max(abs(diff).max(), 1e-8)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, data, ttl, cmap, vmin_val, vmax_val in [
            (axes[0], p, f"Patient - {title}", "YlOrRd", 0, vmax),
            (axes[1], c, f"Control - {title}", "YlOrRd", 0, vmax),
            (axes[2], diff, "Diff (Patient-Control)", "RdBu_r", -vlim, vlim),
        ]:
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin_val, vmax=vmax_val)
            ax.set_title(ttl, fontsize=11); ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("Feature", fontsize=9)
            ax.set_yticks(range(n)); ax.set_yticklabels(feature_names, fontsize=7)
            if day_ticks is not None and data.shape[1] == len(day_ticks):
                ax.set_xticks(day_ticks[::6]); ax.set_xticklabels([str(t) for t in day_ticks[::6]], fontsize=7)
            else:
                ax.set_xticks(range(n)); ax.set_xticklabels(feature_names, fontsize=6, rotation=90)
            plt.colorbar(im, ax=ax)
        plt.suptitle(f"CoFormer {title}", fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_dir / fname}")

    _plot_triple(temporal_p, temporal_c, "Temporal Attention", "1_temporal_attn.png", "Time (hour)")
    _plot_triple(aggre_p, aggre_c, "Agent Aggregation Attention", "2_agent_aggre_attn.png",
                "Key Feature", figsize=(30, 10))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["mtm", "mtm_missing_compare", "mtm_importance", "coformer"],
                    default="mtm")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--split_path", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    if args.mode == "mtm":
        generate_mtm_attention_figures(args.model, split=args.split, layer=args.layer, out_dir=args.out_dir)
    elif args.mode == "mtm_missing_compare":
        generate_mtm_attention_missing_compare(layer=args.layer, out_dir=args.out_dir)
    elif args.mode == "mtm_importance":
        generate_mtm_feature_importance(args.model, split=args.split, out_dir=args.out_dir)
    else:
        generate_coformer_attention_figures(args.model_path, args.data_root, args.split_path, args.out_dir)
