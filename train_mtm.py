"""train_mtm.py
===============
MTM 학습 진입점. 원본 MTM/main.py의 classification CLI + experiments/classification.py의
clsf_exp()를 src/models/mtm.py의 어댑터(build_backbone/MTMModule/load_run_config/
RunConfigView) 위로 재작성.

--datapath의 meta.json(src/preprocess/step3_mtm.py 산출물)에서 num_chn/ratios/max_len을
자동으로 읽는다(원본 MTM_Custom_V2_Auto와 동일 동작) — configs/mtm/custom_v2.yaml은
데이터셋과 무관한 고정 하이퍼파라미터만 담당.

사용법:
  python train_mtm.py --datapath data/mtm/low_missing --wandb_name mtm_low_missing
"""
import argparse
import csv
import json
import math
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.paths import EXTERNAL_MTM, OUTPUTS_ROOT
from src.vendor import vendor_ctx
from src.models.mtm import MTMModule, RunConfigView, load_run_config


class BestValidationMetrics(Callback):
    """각 validation check에서 metric별 최고값(손실은 최저값)을 보관한다."""

    def __init__(self, metric_names):
        super().__init__()
        self.metric_names = list(metric_names)
        self.best = {}

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        names = self.metric_names + ["loss"]
        for name in names:
            key = f"epoch/val_{name}"
            raw = trainer.callback_metrics.get(key)
            if raw is None:
                continue
            value = float(raw.detach().cpu()) if hasattr(raw, "detach") else float(raw)
            if not math.isfinite(value):
                continue
            selection = "min" if name == "loss" else "max"
            current = self.best.get(name)
            improved = current is None or (value < current["value"] if selection == "min"
                                            else value > current["value"])
            if improved:
                self.best[name] = {
                    "value": value,
                    "epoch": int(trainer.current_epoch),
                    "global_step": int(trainer.global_step),
                    "selection": selection,
                }


def write_metrics_summary(path, tracker, test_results, target, dataset, seed, checkpoint):
    rows = []
    for metric, record in tracker.best.items():
        rows.append({
            "target": target, "dataset": dataset, "seed": seed, "split": "validation",
            "metric": metric, "selection": record["selection"], "value": record["value"],
            "epoch": record["epoch"], "global_step": record["global_step"],
            "checkpoint": checkpoint,
        })
    for key, value in (test_results[0] if test_results else {}).items():
        if not key.startswith("epoch/test_"):
            continue
        rows.append({
            "target": target, "dataset": dataset, "seed": seed, "split": "test",
            "metric": key.removeprefix("epoch/test_"),
            "selection": "best_validation_auroc_checkpoint", "value": float(value),
            "epoch": "", "global_step": "", "checkpoint": checkpoint,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["target", "dataset", "seed", "split", "metric", "selection",
                  "value", "epoch", "global_step", "checkpoint"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["phq9", "gad7"], required=True)
    parser.add_argument('--datapath', required=True,
                        help="src/preprocess/step3_mtm.py 산출물 디렉터리 (meta.json 포함), "
                             "e.g. data/mtm/low_missing")
    parser.add_argument('--cfg_path', default=None, help="default: configs/mtm/custom_v2.yaml")
    parser.add_argument('--subset', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None,
                        help="config의 seed를 덮어씀 (반복 학습용)")
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--no_wandb', action='store_true')
    return parser.parse_args()


def main():
    args = build_args()
    datapath = Path(args.datapath).resolve()
    dataset_name = datapath.name
    meta = json.loads((datapath / "meta.json").read_text())
    if meta.get("target") != args.target:
        raise ValueError(f"dataset target={meta.get('target')} does not match --target {args.target}")

    cfg_dict = load_run_config(datapath, cfg_path=args.cfg_path)
    if args.seed is not None:
        cfg_dict["seed"] = args.seed
    config = RunConfigView(cfg_dict)
    pl.seed_everything(config.seed)

    run_root = OUTPUTS_ROOT / "mtm" / args.target / dataset_name / f"seed_{config.seed}"
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run output already exists; refusing to mix files: {run_root}")
    subset_key = f"{args.subset}_{args.target}"

    with vendor_ctx(EXTERNAL_MTM):
        from data_modules.raindrop import RaindropDataModule

        rdm = RaindropDataModule(config.datapath, subset_key, config.batch_size,
                                 dataset=config.dataset, load=True, emr=config.emr,
                                 compact=True, weighted_sampling=config.weighted_sampling)
        train_loader = rdm.train_dataloader()
        val_loader = rdm.val_dataloader()
        test_loader = rdm.test_dataloader()

    csv_logger = CSVLogger(save_dir=str(run_root), name="logs", version="")

    model = MTMModule(config.get_model(), config.forward_fn, **dict(config))

    # val_auroc 기준 best 3개 체크포인트 저장, patience=es_patience로 early stopping
    # (README 관행 유지 — CoFormer는 save_top_k=1, MTM은 top_k=3로 원본과 동일하게 둠).
    checkpoint_dir = run_root / "checkpoints"
    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch:03d}-step={step}",
        auto_insert_metric_name=False,
        monitor=config.monitor,
        save_top_k=3,
        mode=config.mon_mode,
    )
    es_callback = EarlyStopping(monitor=config.monitor, mode=config.mon_mode,
                                patience=config.es_patience)
    metric_tracker = BestValidationMetrics(config.metrics)

    loggers = [csv_logger]
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project="origin_mtm",
            name=args.wandb_name or f"mtm_{args.target}_{dataset_name}",
            group=f"mtm_{args.target}",
            save_dir=str(OUTPUTS_ROOT / "wandb"),
        )
        loggers.append(wandb_logger)

    val_check_interval = min(2500, len(train_loader))
    print(f"validation interval: every {val_check_interval} train batches")

    trainer = pl.Trainer(
        devices=[args.gpu],
        accelerator="gpu",
        precision="16-mixed",
        callbacks=[ckpt_callback, es_callback, metric_tracker],
        logger=loggers,
        enable_checkpointing=True,
        max_epochs=config.max_epochs,
        gradient_clip_val=config.grad_clip_val,
        val_check_interval=val_check_interval,
    )
    trainer.fit(model, train_loader, val_loader)

    best_model = MTMModule.load_from_checkpoint(
        ckpt_callback.best_model_path, model=config.get_model(), forward_fn=config.forward_fn)
    test_results = trainer.test(best_model, test_loader)
    summary_path = checkpoint_dir / f"metrics_summary_{args.target}_{dataset_name}.csv"
    write_metrics_summary(
        summary_path, metric_tracker, test_results, args.target, dataset_name, config.seed,
        ckpt_callback.best_model_path,
    )
    print(f"metric summary -> {summary_path}")

    if not args.no_wandb:
        wandb_logger.experiment.finish()


if __name__ == '__main__':
    main()
