"""MTM(Multi-scale Temporal Mixer) backbone + Lightning wrapper.

The actual model implementation lives in the pristine git submodule
`external/mtm` (upstream zshhans/MTM, pinned commit
5cc68179b98647a836aaf75c265d36e777ba56ca) — that directory is NEVER modified.
Everything here that needs to differ from upstream (variable window length via
`max_len`, exposing attention weights for analysis) is done via adapters:
subclassing pristine classes and monkeypatching the *module namespace* of
`external/mtm/mtm/mtm.py` *before* any pristine class gets instantiated. See
`_load_mtm_base()` below for why that specific patch point is the correct one.

The Lightning module (`MTMModule`) below is this project's own training-loop
code (ported from the old `MTM/tasks/clsf_module.py`, not vendored), so it is
reimplemented directly rather than adapted against anything in external/mtm.
"""
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics import AveragePrecision, AUROC, Accuracy, Precision, Recall, F1Score, MetricCollection
import torchmetrics.functional as tmF

from src.paths import EXTERNAL_MTM, REPO_ROOT
from src.vendor import vendor_ctx

METRICS_MAP = {
    'auprc': AveragePrecision,
    'auroc': AUROC,
    'acc': Accuracy,
    'prec': Precision,
    'rec': Recall,
    'f1': F1Score,
}


# --------------------------------------------------------------------------
# Backbone loading + adapters
# --------------------------------------------------------------------------

def _load_mtm_base():
    """Import the pristine MTM/utils classes from external/mtm and patch the
    ChannelAttn/TokenMixingAttn names used inside mtm.mtm's own namespace so
    that every instance constructed from now on (i.e. every TokenMixingLayer
    built after this call) caches its true attention weights on `.attn_cache`.

    Why patching `mtm.mtm.ChannelAttn`/`mtm.mtm.TokenMixingAttn` (not
    `mtm.utils.*`) works: `external/mtm/mtm/mtm.py` does `from .utils import *`
    at import time, which copies those *names* into `mtm.mtm`'s own module
    __dict__ (a separate binding from `mtm.utils.*`, though initially pointing
    at the same class objects). `TokenMixingLayer.__init__` references the
    bare names `ChannelAttn(...)`/`TokenMixingAttn(...)`; as free variables
    inside a function defined in mtm.mtm, Python resolves them via that
    function's `__globals__`, which *is* `mtm.mtm.__dict__`. So reassigning
    `mtm.mtm.ChannelAttn = <subclass>` (and same for TokenMixingAttn) — a
    plain module-attribute set, done via the imported module object — changes
    what every subsequent `ChannelAttn(...)`/`TokenMixingAttn(...)` call
    inside TokenMixingLayer resolves to, without touching
    `external/mtm/mtm/utils.py` at all. This must happen before `MTM(...)`
    (which builds TokenMixingLayers) is instantiated; it does not need to
    happen before mtm.mtm is imported.

    Why TokenMixingAttn ALSO needs this (and a forward hook isn't enough, per
    the caveat documented on `attach_attention_hooks` below): pristine
    `TokenMixingLayer.forward` always calls `self.mixer(..., True)` —
    `return_imp=True` hardcoded, no other code path exists in this
    architecture — so `TokenMixingAttn.forward()`'s return value is *always*
    `(out, imp)`, a lossy `(b, t)` token-importance reduction of the real
    `(b, c, tq, tk)` attention matrix, never the matrix itself. `_attn_block`
    (called by forward() before that reduction) still computes and returns
    the genuine matrix, so caching it there — exactly like ChannelAttn's
    `_attn_block`, whose `forward()` never returns attn at all — is the only
    way to expose the real attention weights without editing external/mtm.
    `TemporalAttn` needs no such treatment: its `forward()` returns `(out,
    attn)` un-reduced, so a plain forward hook on it (see
    `attach_attention_hooks`) already gets the real matrix.
    """
    with vendor_ctx(EXTERNAL_MTM):
        from mtm.mtm import MTM
        from mtm.utils import (
            PE_QK_FUNC, ChannelAttn, TokenMixingAttn, precompute_ape, precompute_rpe,
        )
        import torch.nn.functional as F
        from einops import rearrange, repeat
        import mtm.mtm as _mtm_mod

        class ChannelAttnWithCache(ChannelAttn):
            """Same as pristine ChannelAttn, except `_attn_block` also stashes
            its attention weights on `self.attn_cache`. Pristine ChannelAttn
            never returns attn from forward() (its `_attn_block` computes it
            as a fully local variable and only `out` propagates out), so
            caching it as an attribute is the only way to expose it without
            editing external/mtm.
            """

            def _attn_block(self, x, x_mask):
                xq = self.wq(x)
                xk = self.wk(x)
                xv = self.wv(x)
                attn = torch.einsum("btqd,btkd->btqk", xq, xk) / math.sqrt(self.d_head)
                attn_mask = x_mask[:, :, None, :] | x_mask[:, :, :, None]
                attn = torch.masked_fill(attn, attn_mask, float("-inf"))
                attn = self.drop(F.softmax(attn, -1).nan_to_num(0))
                self.attn_cache = attn.detach()
                out = torch.einsum("btqk,btkd->btqd", attn, xv)
                return out

        class TokenMixingAttnWithCache(TokenMixingAttn):
            """Same as pristine TokenMixingAttn, except `_attn_block` also
            stashes the true (pre-`return_imp`-reduction) attention weights on
            `self.attn_cache`. See `_load_mtm_base`'s docstring for why this
            can't be done via a forward hook."""

            def _attn_block(self, x, x_mask, p_mask, pos, pe, imp, idx_c, pe_type='rel'):
                bsz, nt, nc, nd = x.shape
                imp = repeat(imp, "b t -> b t c", c=nc)
                idx_c = repeat(torch.where(x_mask, imp, idx_c[:, :nt, ...]),
                               'b t c -> b t c d', d=nd)
                x = torch.gather(x, 2, idx_c)

                if pe_type in PE_QK_FUNC:
                    xq, xk = PE_QK_FUNC[pe_type](self.wq(x), self.wk(x), pos, pe)
                    xq = rearrange(xq, "(b t c) d -> (b c) t d", b=bsz, t=nt)
                    xk = rearrange(xk, "(b t c) d -> (b c) t d", b=bsz, t=nt)
                else:
                    xq, xk = self.wq(x), self.wk(x)
                    xq = rearrange(xq, "b t c d -> (b c) t d")
                    xk = rearrange(xk, "b t c d -> (b c) t d")

                xv = rearrange(self.wv(x), "b t c d -> b c t d")

                attn = rearrange(torch.matmul(xq, xk.transpose(1, 2)) /
                                 math.sqrt(self.d_head),
                                 "(b c) tq tk -> b c tq tk", b=bsz)

                x_mask = x_mask.transpose(1, 2)
                p_mask = p_mask.transpose(1, 2)
                mask = p_mask[:, :, :, None] | p_mask[:, :, None, :]
                attn = F.softmax(attn.masked_fill(mask, float('-inf')), -1).nan_to_num(0)
                self.attn_cache = attn.detach()

                weight = torch.where(x_mask, 1 / nt, 1)
                weighted_attn = attn * weight[:, :, None, :]
                out = torch.einsum("bcmn,bcnd->bmcd", self.drop(weighted_attn), xv)
                return out, attn

        _mtm_mod.ChannelAttn = ChannelAttnWithCache
        _mtm_mod.TokenMixingAttn = TokenMixingAttnWithCache
        return MTM, precompute_rpe, precompute_ape


_MTM, _precompute_rpe, _precompute_ape = _load_mtm_base()


class MTMAdapter(_MTM):
    """Adds a `max_len` kwarg on top of the pristine MTM (which hardcodes
    rpe/ape to `precompute_rpe`/`precompute_ape`'s own default of 640).

    Pristine `MTM.__init__` has no `max_len` parameter at all — it always
    calls `precompute_rpe(d_model)` / `precompute_ape(d_model)` and
    `register_buffer`s the results as `rpe`/`ape`. We first let the pristine
    __init__ run as-is (registering rpe/ape at the default max_len=640), then
    recompute them at the requested `max_len` and reassign
    `self.rpe`/`self.ape`. This is safe without re-calling `register_buffer`:
    `nn.Module.__setattr__` special-cases attribute names already present in
    `self._buffers` — it just replaces the dict entry in place — so a plain
    `self.rpe = new_tensor` after `register_buffer('rpe', ...)` keeps it
    registered as a buffer (it will still show up in `state_dict()`, `.to()`,
    etc.), it does not need a fresh `register_buffer` call.
    """

    def __init__(self, *args, max_len=640, **kwargs):
        super().__init__(*args, **kwargs)
        self.rpe = _precompute_rpe(self.d_model, max_len)
        self.ape = _precompute_ape(self.d_model, max_len)


def build_backbone(num_chn, d_static, num_cls, ratios, d_model=96, r_hid=4, drop=0.2,
                    norm_first=True, down_mode='concat', max_len=640, **kwargs):
    """Factory matching the pristine `MTM.__init__` signature
    (num_chn, d_static, num_cls, ratios, d_model, r_hid, drop, norm_first,
    down_mode, **kwargs), plus our `max_len` adapter kwarg."""
    return MTMAdapter(num_chn=num_chn, d_static=d_static, num_cls=num_cls, ratios=ratios,
                       d_model=d_model, r_hid=r_hid, drop=drop, norm_first=norm_first,
                       down_mode=down_mode, max_len=max_len)


def attach_attention_hooks(model):
    """Registers a forward hook on every TemporalAttn instance in `model` (the
    `inp_layer` TokenMixingLayer plus each `mixers[i]` TokenMixingLayer) to
    capture its attention weights from its forward-return tuple — pristine
    `TemporalAttn.forward` returns `(out, attn)` un-reduced, so a plain hook
    is sufficient (no subclassing needed).

    Returns a `captured` dict (keyed `"{layer_name}.temporal"`) that gets
    (re)populated on every `model(...)` call; inspect it after calling
    forward.

    `TokenMixingAttn` and `ChannelAttn` are NOT hooked here — pristine
    `TokenMixingLayer.forward` always calls `self.mixer(..., True)`
    (`return_imp=True` hardcoded, no other code path exists), so
    `TokenMixingAttn.forward()`'s return is always `(out, imp)`, a lossy
    `(b, t)` token-importance reduction of the real `(b, c, tq, tk)`
    attention matrix — a forward hook on it cannot recover the real matrix.
    `ChannelAttn.forward` never returns attn at all. Both are instead patched
    (in `_load_mtm_base` above, via `TokenMixingAttnWithCache`/
    `ChannelAttnWithCache`) to cache their true attention weights on
    `self.attn_cache` directly — after calling `model(...)`, read
    `layer.mixer.attn_cache`/`layer.channel.attn_cache` (e.g.
    `model.inp_layer.mixer.attn_cache`, `model.mixers[i].channel.attn_cache`).
    """
    captured = {}

    def make_hook(name):

        def hook(module, args, output):
            if isinstance(output, tuple):
                val = output[-1]
                captured[name] = val.detach() if hasattr(val, "detach") else val
            return output

        return hook

    layers = [("inp_layer", model.inp_layer)] + [(f"mixer_{i}", m) for i, m in enumerate(model.mixers)]
    for name, layer in layers:
        layer.temporal.register_forward_hook(make_hook(f"{name}.temporal"))
        # layer.mixer / layer.channel: no forward hook needed — both cache their
        # true attention on `.attn_cache` via the _load_mtm_base patch, see above.
    return captured


# --------------------------------------------------------------------------
# Run config (ported from MTM/config/mtm_clsf_config.py::MTM_Custom_V2_Auto)
#
# The old MTM_Custom_V2_Auto/CustomConfig_* classes lived in the WORKING COPY's
# config/mtm_clsf_config.py, never upstream — `external/mtm`'s pristine
# config/mtm_clsf_config.py only has MTM_P12/MTM_P19/MTM_PAM. So this cannot be
# imported from external/mtm at all; it's reimplemented here against
# configs/mtm/custom_v2.yaml (the fixed hyperparams) + a dataset's own meta.json
# (the window-size-derived ratios/max_len/num_chn), and is the single source of
# truth both `train_mtm.py` and `src/analysis/attention_viz.py` build configs
# from — neither should reimplement this logic separately.
# --------------------------------------------------------------------------

def _auto_ratios(window_units):
    """Downsampling ratios from window size, keeping final T in a reasonable
    range (4~16) after downsampling. Verbatim from the original
    MTM_Custom_V2_Auto's _auto_ratios()."""
    if window_units <= 6:
        return [2]
    elif window_units <= 24:
        return [2, 2]
    elif window_units <= 72:
        return [2, 3]
    elif window_units <= 168:
        return [2, 2, 3]
    else:
        return [2, 2, 4]


def load_run_config(datapath, cfg_path=None):
    """Reads configs/mtm/custom_v2.yaml (fixed hyperparams: seed/d_model/lr/.../
    metrics) and merges in the window-size-derived fields read from
    `<datapath>/meta.json` (num_chn, ratios, max_len, dataset tag) — mirrors
    MTM_Custom_V2_Auto.__init__'s original behavior. Returns a plain dict."""
    cfg_path = Path(cfg_path) if cfg_path else REPO_ROOT / "configs" / "mtm" / "custom_v2.yaml"
    cfg = dict(yaml.safe_load(open(cfg_path)))

    meta = json.loads((Path(datapath) / "meta.json").read_text())
    u, w, s = meta["unit_minutes"], meta["window_units"], meta["stride_units"]
    ver_match = re.search(r"_(v\d+)_", Path(datapath).name)
    ver = ver_match.group(1) if ver_match else meta.get("data_version", "v2")

    cfg.update(
        datapath=str(datapath),
        num_chn=meta["n_features"],
        dataset=f"Custom_{ver}_u{u}_w{w}_s{s}",
        ratios=_auto_ratios(w),
        max_len=w + 16,
    )
    return cfg


class RunConfigView:
    """Adapts a `load_run_config()` dict to the attribute + `dict(self)` /
    `.get_model()` / `.forward_fn` interface the old MTMBaseConfig subclasses
    exposed (`config.dataset`, `config.get_model()`, ...) — used by any
    analysis/training code that drives a custom_v2/v3 run without needing to
    know load_run_config() returns a plain dict. The single shared place for
    this; don't reimplement it per call site (src/analysis/attention_viz.py
    and src/analysis/subgroup_cm.py both use this)."""

    def __init__(self, data):
        self._data = dict(data)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name)

    def keys(self):
        return self._data.keys()

    def __getitem__(self, item):
        return self._data[item]

    def get_model(self):
        return build_backbone(**dict(self))

    @property
    def forward_fn(self):
        return mtm_forward_fn


# --------------------------------------------------------------------------
# forward_fn (ported from MTM/config/mtm_clsf_config.py::MTMBaseConfig.forward_fn)
# --------------------------------------------------------------------------

def mtm_forward_fn(model, batch):
    x, x_mask, t, y, x_static, uids = batch
    y_pred = model(x, x_mask, t, x_static)
    return y_pred, y


# --------------------------------------------------------------------------
# Lightning module (ported from MTM/tasks/clsf_module.py::ClassificationModule)
# --------------------------------------------------------------------------

class MTMModule(pl.LightningModule):

    def __init__(self, model, forward_fn, **config):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'forward_fn'])
        self.model = model

        self.num_cls = config['num_cls']
        if self.num_cls > 2:
            kwargs = {
                'task': "multiclass",
                'num_classes': self.num_cls,
                'average': 'macro',
            }
        else:
            kwargs = {'task': "binary"}
        metrics = MetricCollection(
            {k: METRICS_MAP[k](**kwargs) for k in config['metrics']})
        self.train_metrics = metrics.clone(prefix='epoch/train_')
        self.val_metrics = metrics.clone(prefix='epoch/val_')
        self.test_metrics = metrics.clone(prefix='epoch/test_')

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = config['lr']
        self.lr_factor = config['lr_factor']
        self.optim = "adamw"
        self.weight_decay = config['weight_decay']
        self.patience = config['patience']
        self.monitor = config['monitor']
        self.mon_mode = config['mon_mode']
        self.forward_fn = forward_fn

    def _compute_step_metrics(self, y_pred_prob, y, prefix):
        """배치 단위 step-wise 메트릭을 functional API로 계산해서 반환."""
        task = "binary" if self.num_cls == 2 else "multiclass"
        kwargs = {} if self.num_cls == 2 else {"num_classes": self.num_cls, "average": "macro"}
        result = {}
        for key in self.hparams.get("metrics", []):
            try:
                if key == "f1":
                    val = tmF.f1_score(y_pred_prob, y, task=task, **kwargs)
                elif key == "acc":
                    val = tmF.accuracy(y_pred_prob, y, task=task, **kwargs)
                elif key == "prec":
                    val = tmF.precision(y_pred_prob, y, task=task, **kwargs)
                elif key == "rec":
                    val = tmF.recall(y_pred_prob, y, task=task, **kwargs)
                elif key == "auroc":
                    val = tmF.auroc(y_pred_prob, y, task=task, **kwargs)
                elif key == "auprc":
                    val = tmF.average_precision(y_pred_prob, y, task=task, **kwargs)
                else:
                    continue
                result[f"step/{prefix}{key}"] = val
            except Exception:
                pass
        return result

    def training_step(self, batch, batch_idx):
        batch_size = batch[0].shape[0]
        y_pred, y = self.forward_fn(self.model, batch)
        pred_loss = self.loss_fn(y_pred, y)
        self.log('step/train_loss', pred_loss,
                 on_step=True, on_epoch=False,
                 batch_size=batch_size, prog_bar=True)
        self.log('epoch/train_loss', pred_loss,
                 on_step=False, on_epoch=True,
                 batch_size=batch_size)
        if self.num_cls == 2:
            y_pred_prob = y_pred[:, 1].sigmoid()
        else:
            y_pred_prob = y_pred.softmax(dim=1)
        # step-wise: functional API로 현재 배치 기준 scalar 로그
        step_vals = self._compute_step_metrics(y_pred_prob, y, prefix="train_")
        self.log_dict(step_vals, on_step=True, on_epoch=False, batch_size=batch_size)
        # epoch-wise: MetricCollection 누적
        self.train_metrics.update(y_pred_prob, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, batch_size=batch_size)
        return pred_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        batch_size = batch[0].shape[0]
        y_pred, y = self.forward_fn(self.model, batch)
        pred_loss = self.loss_fn(y_pred, y)
        if self.num_cls == 2:
            y_pred = y_pred[:, 1].sigmoid()
        else:
            y_pred = y_pred.softmax(dim=1)
        # validation이 도는 동안 global_step은 고정 → step-log를 validation당 1개만 찍어
        # epoch보다는 촘촘하고(매 val_check_interval마다 점 1개) 매 batch보다는 부드러운 trend 확보
        do_step_log = (batch_idx == 0)
        if dataloader_idx is None or dataloader_idx == 0:
            # epoch-wise: 매 batch 누적 (val_auroc 등 monitor 정확성 유지)
            self.log('epoch/val_loss', pred_loss,
                     on_step=False, on_epoch=True,
                     batch_size=batch_size, add_dataloader_idx=False)
            self.val_metrics.update(y_pred, y)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True,
                          add_dataloader_idx=False)
            # step-wise: N batch마다 1번만
            if do_step_log:
                step_vals = self._compute_step_metrics(y_pred, y, prefix="val_")
                step_vals['step/val_loss'] = pred_loss
                self.log_dict(step_vals, on_step=True, on_epoch=False,
                              batch_size=batch_size, add_dataloader_idx=False)
        else:
            # epoch-wise: 매 batch 누적
            self.log('epoch/test_loss', pred_loss,
                     on_step=False, on_epoch=True,
                     batch_size=batch_size, add_dataloader_idx=False)
            self.test_metrics.update(y_pred, y)
            self.log_dict(self.test_metrics, on_step=False, on_epoch=True,
                          add_dataloader_idx=False, prog_bar=True)
            # step-wise: N batch마다 1번만
            if do_step_log:
                step_vals = self._compute_step_metrics(y_pred, y, prefix="test_")
                step_vals['step/test_loss'] = pred_loss
                self.log_dict(step_vals, on_step=True, on_epoch=False,
                              batch_size=batch_size, add_dataloader_idx=False)
        return

    def on_validation_epoch_end(self):
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch_size = batch[0].shape[0]
        y_pred, y = self.forward_fn(self.model, batch)
        pred_loss = self.loss_fn(y_pred, y)
        if self.num_cls == 2:
            y_pred = y_pred[:, 1].sigmoid()
        else:
            y_pred = y_pred.softmax(dim=1)
        self.log('step/test_loss', pred_loss,
                 on_step=True, on_epoch=False,
                 batch_size=batch_size, add_dataloader_idx=False)
        self.log('epoch/test_loss', pred_loss,
                 on_step=False, on_epoch=True,
                 batch_size=batch_size, add_dataloader_idx=False)
        step_vals = self._compute_step_metrics(y_pred, y, prefix="test_")
        self.log_dict(step_vals, on_step=True, on_epoch=False, batch_size=batch_size)
        self.test_metrics.update(y_pred, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return

    def configure_optimizers(self):
        if self.optim == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            raise ValueError
        scheduler_config = {}
        scheduler_config["scheduler"] = ReduceLROnPlateau(optimizer,
                                                          self.mon_mode,
                                                          factor=self.lr_factor,
                                                          patience=self.patience,
                                                          min_lr=1e-8)
        scheduler_config["monitor"] = self.monitor
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
