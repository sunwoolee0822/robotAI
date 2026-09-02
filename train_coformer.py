"""train_coformer.py
====================
CoFormer 학습 진입점. 원본 sw/CoFormer/train_robotai.py를 src/models/coformer.py의
어댑터(build_backbone/CoFormerModule/CoFormerDataModule/RobotAIConfig) 위로 재작성.

사전 준비: ./setup.sh (external/coformer 서브모듈 체크아웃 + fp16 masked_fill 패치 적용).
아래 _check_fp16_patch()가 패치 미적용 상태를 감지해 명확한 에러로 안내한다.

사용법:
  python train_coformer.py --cfg trans_medical_missing \
      --data_root data/coformer/low_missing/ --split_path data/coformer/low_missing/split.npy \
      --wandb_name coformer_low_missing
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.paths import EXTERNAL_COFORMER, OUTPUTS_ROOT, REPO_ROOT
from src.models.coformer import CoFormerDataModule, CoFormerModule, RobotAIConfig, build_backbone

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')


def _check_fp16_patch():
    """patches/coformer-fp16-maskfill.patch가 적용됐는지 확인 (fp16 학습 중
    -1e10 masked_fill이 overflow해 softmax가 NaN이 되는 문제 방지). 미적용이면
    학습을 시작하지 않고 setup.sh 실행을 안내한다."""
    src_path = EXTERNAL_COFORMER / "models" / "model_medical_attn_aggre.py"
    text = src_path.read_text()
    if "-1e10" in text and "masked_fill" in text:
        print(f"[!] {src_path}에 fp16 패치가 적용되지 않았습니다 (-1e10 그대로).\n"
              f"    먼저 './setup.sh'를 실행해 patches/coformer-fp16-maskfill.patch를 적용하세요.",
              file=sys.stderr)
        sys.exit(1)


class BestValidationMetrics(Callback):
    """각 validation epoch에서 metric별 최고값(손실은 최저값)을 보관한다."""

    def __init__(self, metric_names):
        super().__init__()
        self.metric_names = list(metric_names)
        self.best = {}

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        for name in self.metric_names + ["loss"]:
            key = f"epoch/val_{name}"
            raw = trainer.callback_metrics.get(key)
            if raw is None:
                continue
            value = float(raw.detach().cpu()) if hasattr(raw, "detach") else float(raw)
            if not math.isfinite(value):
                continue
            selection = "min" if name == "loss" else "max"
            old = self.best.get(name)
            improved = old is None or (value < old["value"] if selection == "min"
                                       else value > old["value"])
            if improved:
                self.best[name] = {
                    "value": value, "epoch": int(trainer.current_epoch),
                    "global_step": int(trainer.global_step), "selection": selection,
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
        if key.startswith("epoch/test_"):
            rows.append({
                "target": target, "dataset": dataset, "seed": seed, "split": "test",
                "metric": key.removeprefix("epoch/test_"),
                "selection": "best_validation_auroc_checkpoint", "value": float(value),
                "epoch": "", "global_step": "", "checkpoint": checkpoint,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["target", "dataset", "seed", "split", "metric", "selection",
              "value", "epoch", "global_step", "checkpoint"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='trans_medical_missing')
    parser.add_argument('--cfg_dir', default=str(REPO_ROOT / "configs" / "coformer"))
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--target', choices=['phq9', 'gad7'], required=True)
    parser.add_argument('--seed', type=int, default=None,
                        help="config seed override (반복 학습용)")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="config batch_size override")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--precision', choices=['bf16-mixed', '16-mixed', '32-true'],
                        default='bf16-mixed',
                        help="BF16 avoids FP16 overflow in CoFormer sensor projections")

    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_agents', type=int, default=None)
    parser.add_argument('--num_neighbors', type=int, default=30)
    parser.add_argument('--agent_encoding_dim', type=int, default=32)
    parser.add_argument('--static_dim', type=int, default=4)  # sex/age/height/weight — 실제 학습 run 확인값

    parser.add_argument('--dataset', default='coformer')
    parser.add_argument('--data_root', required=True, help="e.g. data/coformer/low_missing/")
    parser.add_argument('--split_path', default=None, help="default: <data_root>/split.npy")
    parser.add_argument('--subset', type=int, default=1)

    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--no_wandb', action='store_true')
    return parser.parse_args()


def main():
    args = build_args()
    if args.split_path is None:
        args.split_path = args.data_root.rstrip('/') + '/split.npy'
    _check_fp16_patch()

    data_root = Path(args.data_root).resolve()
    dataset_name = data_root.name
    meta = json.loads((data_root / "meta.json").read_text())
    if meta.get("target") != args.target:
        raise ValueError(f"dataset target={meta.get('target')} does not match --target {args.target}")

    cfg = RobotAIConfig(args.cfg, args.cfg_dir, tmp=args.tmp)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    pl.seed_everything(cfg.seed)

    run_root = OUTPUTS_ROOT / "coformer" / args.target / dataset_name / f"seed_{cfg.seed}"
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"run output already exists; refusing to mix files: {run_root}")

    dm = CoFormerDataModule(str(data_root), args.split_path, cfg.batch_size,
                            num_workers=args.num_workers)
    dm.setup()

    if args.num_agents is None:
        args.num_agents = dm.train_dset.data.shape[1]

    model_kwargs = dict(
        src_vocab=args.input_dim, tgt_vocab=args.output_dim,
        N=args.num_layers, d_model=args.d_model, d_ff=args.d_ff,
        h=args.heads, dropout=args.dropout, num_agents=args.num_agents,
        num_neighbors=args.num_neighbors, agent_encoding_dim=args.agent_encoding_dim,
        static_dim=args.static_dim,
    )
    backbone = build_backbone(**model_kwargs)
    model = CoFormerModule(backbone, cfg, args)

    # MTM(train_mtm.py)과 동일 조건: monitor=epoch/val_auroc(max), es_patience=100.
    checkpoint_dir = run_root / "checkpoints"
    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch:03d}-step={step}",
        auto_insert_metric_name=False,
        monitor='epoch/val_auroc', save_top_k=1, mode='max')
    es_callback = EarlyStopping(monitor='epoch/val_auroc', mode='max', patience=100)
    metric_tracker = BestValidationMetrics(['f1', 'acc', 'prec', 'rec', 'auroc', 'auprc'])

    run_name = args.wandb_name or f"coformer_{args.target}_{dataset_name}_seed{cfg.seed}"
    loggers = [CSVLogger(save_dir=str(run_root), name="logs", version="")]
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project="origin_mtm",
            name=run_name,
            group=f"coformer_{args.target}",
            save_dir=str(OUTPUTS_ROOT / "wandb"),
            config={
                "cfg": args.cfg, "num_epochs": cfg.num_epochs,
                "batch_size": cfg.batch_size, "lr": cfg.lr,
                "num_layers": args.num_layers, "heads": args.heads,
                "d_model": args.d_model, "d_ff": args.d_ff,
                "dropout": args.dropout, "num_neighbors": args.num_neighbors,
                "agent_encoding_dim": args.agent_encoding_dim,
                "static_dim": args.static_dim, "output_dim": args.output_dim,
                "target": args.target, "dataset": dataset_name,
                "seed": cfg.seed, "num_agents": args.num_agents,
                "data_root": str(data_root), "precision": args.precision,
            },
        )
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        devices=[args.gpu],
        accelerator="gpu",
        precision=args.precision,
        max_epochs=cfg.num_epochs,
        callbacks=[ckpt_callback, es_callback, metric_tracker],
        logger=loggers,
        enable_checkpointing=True,
        default_root_dir=str(run_root),
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    best_model = CoFormerModule.load_from_checkpoint(
        ckpt_callback.best_model_path,
        model=build_backbone(**model_kwargs),
        cfg=cfg, args=args,
    )
    test_results = trainer.test(best_model, dm.test_dataloader())
    summary_path = checkpoint_dir / f"metrics_summary_{args.target}_{dataset_name}.csv"
    write_metrics_summary(
        summary_path, metric_tracker, test_results, args.target, dataset_name,
        cfg.seed, ckpt_callback.best_model_path)
    print(f"metric summary -> {summary_path}")

    if not args.no_wandb:
        wandb_logger.experiment.finish()


if __name__ == '__main__':
    main()
