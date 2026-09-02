"""CoFormer 백본을 이 프로젝트에 통합하기 위한 어댑터 모듈.

세 가지 출처를 조합한다:
1. 데이터 모듈 (구 `CoFormer/data/dataloader.py`) — `seq_collate_irregular`,
   `medical_dataloader`, `CoFormerDataModule`만 유지. Raindrop P12/PAM 벤치마크용
   `medicalp12_dataloader`/`medicalpam_dataloader`/`seq_collate_irregular_wo_static`은
   이 프로젝트에서 쓰지 않으므로 제외.
2. Lightning 모듈 (구 `CoFormer/tasks/coformer_module.py`) — 이 프로젝트 자체 코드라
   거의 그대로 복사.
3. `external/coformer`(pristine upstream submodule, commit 69261db)에서 가져오는
   백본/Config에 대한 어댑터. `external/coformer`는 절대 수정하지 않으며, 원본과 다르게
   동작해야 하는 부분(설정 경로 해석, static_dim, GAT attention 캡처)은 전부 이 파일의
   서브클래스/팩토리/hook으로 처리한다. `external/coformer` 내부 모듈은 무접두
   import(`from models... import ...`, `from utils... import ...`)라 반드시
   `vendor_ctx(EXTERNAL_COFORMER)` 안에서만 import해야 한다.
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics import AveragePrecision, AUROC, Accuracy, Precision, Recall, F1Score, MetricCollection
import torchmetrics.functional as tmF

from src.paths import EXTERNAL_COFORMER, OUTPUTS_ROOT
from src.vendor import vendor_ctx


# ---------------------------------------------------------------------------
# 1. 데이터 모듈 (원본: CoFormer/data/dataloader.py)
# ---------------------------------------------------------------------------

def seq_collate_irregular(data):
    (data, time, static, mask, gt) = zip(*data)

    data = torch.stack(data, dim=0)
    time = torch.stack(time, dim=0)
    static = torch.stack(static, dim=0)
    mask = torch.stack(mask, dim=0)
    gt = torch.stack(gt, dim=0)
    data = {
        'data': data,
        'time': time,
        'static': static,
        'mask': mask,
        'gt': gt,
    }
    return data


class medical_dataloader(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, root, split_path, training=True, split=None):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(medical_dataloader, self).__init__()

        data_root = Path(root)

        split_data = np.load(split_path, allow_pickle=True)
        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split is not None:
            index = split_data[split_map[split]]
        elif training:
            index = split_data[0]
        else:
            index = split_data[2]
        self.data = np.load(data_root / 'array.npy')[index, :]

        self.time = np.load(data_root / 'time.npy')[index, :]

        self.gt = np.load(data_root / 'gt.npy')[index, :]

        self.static = np.load(data_root / 'static.npy')[index, :]

        self.mask = np.load(data_root / 'mask.npy')[index, :]

        self.data = torch.from_numpy(self.data).type(torch.float)
        self.time = torch.from_numpy(self.time).type(torch.float)
        self.gt = torch.from_numpy(self.gt).type(torch.float)
        self.static = torch.from_numpy(self.static).type(torch.float)
        self.mask = torch.from_numpy(self.mask).type(torch.int)

        self.batch_len = len(self.data)
        print(self.batch_len)
        print(self.data.shape)
        print(self.static.shape)

    def __len__(self):
        return self.batch_len

    def __getitem__(self, index):

        data = self.data[index]
        time = self.time[index]
        gt = self.gt[index]
        static = self.static[index]
        mask = self.mask[index]

        out = \
            [data, time, static, mask, gt]

        return out


class CoFormerDataModule(pl.LightningDataModule):
    def __init__(self, data_root, split_path, batch_size, num_workers=4):
        super().__init__()
        self.data_root = data_root
        self.split_path = split_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dset = medical_dataloader(self.data_root, self.split_path, split='train')
        self.val_dset = medical_dataloader(self.data_root, self.split_path, split='val')
        self.test_dset = medical_dataloader(self.data_root, self.split_path, split='test')

    def train_dataloader(self):
        labels = self.train_dset.gt.squeeze().long()
        class_counts = torch.bincount(labels)
        weights = 1.0 / class_counts[labels].float()
        n_total = int(class_counts.max().item() * 2)
        sampler = WeightedRandomSampler(weights, num_samples=n_total, replacement=True)
        return DataLoader(self.train_dset, batch_size=self.batch_size, sampler=sampler,
                           num_workers=self.num_workers, collate_fn=seq_collate_irregular,
                           pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size, shuffle=False,
                           num_workers=self.num_workers, collate_fn=seq_collate_irregular,
                           pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.batch_size, shuffle=False,
                           num_workers=self.num_workers, collate_fn=seq_collate_irregular,
                           pin_memory=True)


# ---------------------------------------------------------------------------
# 2. Lightning 모듈 (원본: CoFormer/tasks/coformer_module.py, 이 프로젝트 자체 코드)
# ---------------------------------------------------------------------------

class CoFormerModule(pl.LightningModule):

    def __init__(self, model, cfg, args):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'cfg', 'args'])
        self.model = model
        self.cfg = cfg
        self.args = args

        kwargs = {'task': 'binary'}
        metrics = MetricCollection({
            'auprc': AveragePrecision(**kwargs),
            'auroc': AUROC(**kwargs),
            'acc': Accuracy(**kwargs),
            'prec': Precision(**kwargs),
            'rec': Recall(**kwargs),
            'f1': F1Score(**kwargs),
        })
        # MTM(tasks/clsf_module.py)과 동일한 step/epoch 네이밍 컨벤션:
        #   step/{train,val,test}_{metric}  : 배치 단위 즉석 계산
        #   epoch/{train,val,test}_{metric} : MetricCollection 누적치
        self.train_metrics = metrics.clone(prefix='epoch/train_')
        self.val_metrics = metrics.clone(prefix='epoch/val_')
        self.test_metrics = metrics.clone(prefix='epoch/test_')
        self.loss_fn = nn.CrossEntropyLoss()
        self._metric_keys = ['f1', 'acc', 'prec', 'rec', 'auroc', 'auprc']

    def _forward(self, batch):
        array = batch['data']
        time = batch['time']
        static = batch['static']
        mask = batch['mask']
        gt = batch['gt'].squeeze(1).long()
        for name, tensor in (("array", array), ("time", time), ("static", static)):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"non-finite CoFormer {name} input")
        input_max = float(array.detach().abs().max().cpu())
        if input_max >= 1e9:
            raise FloatingPointError(
                f"CoFormer input magnitude is invalid (max_abs={input_max:.6g}); "
                "regenerate the dataset with src.preprocess.step2_arrays"
            )
        logits = self.model(array, time, mask, static)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                f"non-finite CoFormer logits (input_max_abs={input_max:.6g})"
            )
        return logits, gt

    def _loss(self, logits, gt, split):
        loss = self.loss_fn(logits, gt)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CoFormer {split} loss")
        return loss

    def _compute_step_metrics(self, probs, gt, prefix):
        """배치 단위 step-wise 메트릭 (functional API). MTM과 동일 방식."""
        result = {}
        for key in self._metric_keys:
            try:
                if key == "f1":
                    val = tmF.f1_score(probs, gt, task="binary")
                elif key == "acc":
                    val = tmF.accuracy(probs, gt, task="binary")
                elif key == "prec":
                    val = tmF.precision(probs, gt, task="binary")
                elif key == "rec":
                    val = tmF.recall(probs, gt, task="binary")
                elif key == "auroc":
                    val = tmF.auroc(probs, gt, task="binary")
                elif key == "auprc":
                    val = tmF.average_precision(probs, gt, task="binary")
                else:
                    continue
                result[f"step/{prefix}{key}"] = val
            except Exception:
                pass
        return result

    def training_step(self, batch, batch_idx):
        logits, gt = self._forward(batch)
        loss = self._loss(logits, gt, "train")
        batch_size = logits.shape[0]

        self.log('step/train_loss', loss, on_step=True, on_epoch=False,
                 batch_size=batch_size, prog_bar=True)
        self.log('epoch/train_loss', loss, on_step=False, on_epoch=True,
                 batch_size=batch_size)

        probs = torch.softmax(logits, dim=1)[:, 1]
        step_vals = self._compute_step_metrics(probs, gt, prefix="train_")
        self.log_dict(step_vals, on_step=True, on_epoch=False, batch_size=batch_size)

        self.train_metrics.update(probs, gt)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, gt = self._forward(batch)
        loss = self._loss(logits, gt, "validation")
        batch_size = logits.shape[0]
        probs = torch.softmax(logits, dim=1)[:, 1]

        self.val_metrics.update(probs, gt)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        self.log('epoch/val_loss', loss, on_step=False, on_epoch=True,
                 batch_size=batch_size, prog_bar=True)

        step_vals = self._compute_step_metrics(probs, gt, prefix="val_")
        step_vals['step/val_loss'] = loss
        self.log_dict(step_vals, on_step=True, on_epoch=False, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        logits, gt = self._forward(batch)
        loss = self._loss(logits, gt, "test")
        batch_size = logits.shape[0]
        probs = torch.softmax(logits, dim=1)[:, 1]

        self.test_metrics.update(probs, gt)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        self.log('epoch/test_loss', loss, on_step=False, on_epoch=True,
                 batch_size=batch_size)

        step_vals = self._compute_step_metrics(probs, gt, prefix="test_")
        step_vals['step/test_loss'] = loss
        self.log_dict(step_vals, on_step=True, on_epoch=False, batch_size=batch_size)

    def on_validation_epoch_end(self):
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

        scheduler_type = self.cfg.get('lr_scheduler', 'linear')
        if scheduler_type == 'linear':
            fix = self.cfg.lr_fix_epochs
            total = self.cfg.num_epochs

            def lr_lambda(epoch):
                return 1.0 - max(0, epoch - fix) / float(total - fix + 1)
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif scheduler_type == 'step':
            scheduler = StepLR(optimizer,
                                step_size=self.cfg.decay_step,
                                gamma=self.cfg.decay_gamma)
        else:
            raise ValueError(f'unknown scheduler type: {scheduler_type}')

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# 3. Config 어댑터 (원본: external/coformer/utils/config.py의 Config를 상속)
#
# Pristine `Config.__init__`은 `cfg_path = 'cfg/**/%s.yml' % cfg_id`로 cwd 상대
# 경로에서 cfg yml을 찾고, `results_root_dir`을 yml 자체에서 읽어 산출물 루트로 쓴다.
# 이 프로젝트의 cfg yml들은 `external/coformer/cfg/` 밑이 아니라 REPO_ROOT 하위
# `configs/coformer/`에 있고, 산출물도 yml 내용과 무관하게 항상
# `OUTPUTS_ROOT / "coformer"` 밑에 고정하고 싶으므로(= cwd에도, yml 내용에도 의존하지
# 않는 경로 해석), `Config.__init__`의 본문을 복제하되 `cfg_dir`을 명시 인자로 받고
# `results_root_dir`을 무시하는 서브클래스를 정의한다.
#
# `Config`는 `external/coformer`가 sys.path/cwd에 있어야만 import 가능하므로,
# 모듈 최초 사용 시점에 `vendor_ctx` 안에서 1회 로드하고 캐시한다.
# ---------------------------------------------------------------------------

_RobotAIConfig = None


def _load_config_base():
    """external/coformer의 pristine Config를 vendor_ctx 안에서 import하고,
    이를 상속하는 RobotAIConfig를 최초 1회 정의해 캐시한다."""
    global _RobotAIConfig
    if _RobotAIConfig is not None:
        return _RobotAIConfig

    with vendor_ctx(EXTERNAL_COFORMER):
        # 일부 conda 환경에는 PyPI 'utils'(pyutils) 패키지가 site-packages에 설치돼
        # 있다. 그건 __init__.py가 있는 "정식" 패키지라서, PEP 420 규칙상
        # sys.path에서의 순서와 무관하게 external/coformer/utils(=__init__.py가
        # 없는 namespace package)보다 항상 우선한다 — vendor_ctx가 EXTERNAL_COFORMER를
        # sys.path 맨 앞에 꽂아도 `import utils`는 site-packages의 pyutils로
        # resolve되어 `utils.config`가 없다며 ModuleNotFoundError가 난다(실제로 이
        # 환경에서 재현 확인함). external/coformer를 수정할 수 없으므로, 여기서는
        # import 직전에 sys.modules의 기존 'utils'(및 하위 모듈) 캐시를 잠시 걷어내고
        # external/coformer/utils만 가리키는 namespace 모듈을 직접 만들어 넣어
        # 정확한 'utils.config'가 import되도록 한 뒤, 끝나면 원래 상태로 복원한다.
        saved = {k: v for k, v in sys.modules.items()
                 if k == 'utils' or k.startswith('utils.')}
        for k in saved:
            del sys.modules[k]
        try:
            ns = types.ModuleType('utils')
            ns.__path__ = [str(EXTERNAL_COFORMER / 'utils')]
            sys.modules['utils'] = ns
            from utils.config import Config as _PristineConfig
        finally:
            for k in [k for k in sys.modules if k == 'utils' or k.startswith('utils.')]:
                del sys.modules[k]
            sys.modules.update(saved)

    import glob
    from pathlib import Path

    import yaml
    from easydict import EasyDict

    class RobotAIConfig(_PristineConfig):
        """`Config.__init__`을 복제하되, cfg yml 탐색 디렉터리를 명시 인자
        `cfg_dir`로 받고(cwd 비의존), 산출물 루트를 yml의 `results_root_dir`
        대신 항상 `OUTPUTS_ROOT / "coformer"`로 고정한다(cwd/yml 내용 비의존)."""

        def __init__(self, cfg_id, cfg_dir, tmp=False):
            self.id = cfg_id
            files = glob.glob(str(Path(cfg_dir) / "**" / f"{cfg_id}.yml"), recursive=True)
            assert len(files) == 1, files
            self.yml_dict = EasyDict(yaml.safe_load(open(files[0])))

            cfg_root_dir = "/tmp/agentformer" if tmp else str(OUTPUTS_ROOT / "coformer")
            self.cfg_root_dir = cfg_root_dir
            self.cfg_dir = f"{self.cfg_root_dir}/{cfg_id}"
            self.model_dir = f"{self.cfg_dir}/models"
            self.result_dir = f"{self.cfg_dir}/results"
            self.log_dir = f"{self.cfg_dir}/log"
            self.tb_dir = f"{self.cfg_dir}/tb"
            self.model_path = os.path.join(self.model_dir, "model_%04d.p")
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.result_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)

    _RobotAIConfig = RobotAIConfig
    return _RobotAIConfig


def RobotAIConfig(cfg_id, cfg_dir, tmp=False):
    """`_load_config_base()`가 캐시한 RobotAIConfig 클래스를 인스턴스화하는 팩토리.
    (클래스 자체가 pristine `Config`를 import해야 정의 가능하므로, 정의를 지연시키는
    `_load_config_base`를 거쳐 최초 호출 시점에만 vendor_ctx를 연다.)"""
    cls = _load_config_base()
    return cls(cfg_id, cfg_dir, tmp=tmp)


# ---------------------------------------------------------------------------
# 4. 백본 팩토리 (원본: external/coformer/models/model_medical_attn_aggre.py)
#
# Pristine `make_model()`은 이미 `static_dim`을 인자로 받는다(시그니처 기본값 6) —
# 원본 내부에 하드코딩된 곳이 없으므로 서브클래스가 필요 없고, 호출부에서 명시적으로
# 값을 넘기기만 하면 된다.
#
# 기본값 정정: 이전 버전은 워킹카피의 EncoderDecoder.__init__ 기본값이 6→9로 바뀐
# diff만 보고 9를 기본값으로 삼았는데, 그 9는 실제로 쓰인 적 없는 죽은 기본값이었다
# (호출부가 항상 명시적으로 넘겨서 클래스 자체 기본값은 발동 안 함). 실제 학습에 쓰인
# 값은 4 — analysis/coformer_infer_for_lmm.py가 실제 run(cbiqdsa8/k3smegtp)의
# wandb config.yaml로 확인한 값이고, src/preprocess/step2_arrays.py가 만드는 static
# 벡터도 정확히 4개(sex/age/height/weight)라 데이터 자체와도 일치한다.
# ---------------------------------------------------------------------------

def build_backbone(static_dim=4, **kwargs):
    """`external/coformer`의 pristine `make_model`을 vendor_ctx 안에서 호출해
    CoFormer 백본(EncoderDecoder)을 생성한다.

    kwargs: src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout, num_agents,
    num_neighbors, agent_encoding_dim — CoFormer/train_robotai.py의 make_model
    호출부와 동일하게 그대로 전달한다.
    """
    with vendor_ctx(EXTERNAL_COFORMER):
        from models.model_medical_attn_aggre import make_model
        return make_model(static_dim=static_dim, **kwargs)


# ---------------------------------------------------------------------------
# 5. GAT attention 캡처 hook (원본 워킹카피는 GATlayer.forward를 직접 수정해
#    `x, attn = self.gatconv(graph, x, get_attention=True); self.attn = attn`로
#    바꿨지만, pristine external/coformer의 GATlayer.forward는
#    `x = self.gatconv(graph, x)`로 get_attention을 전혀 넘기지 않는다 — attention
#    텐서가 애초에 계산되지 않으므로 평범한 forward_hook만으로는 얻을 수 없다.
#
#    대신 각 GATlayer.gatconv(dgl GATv2Conv, forward(graph, feat,
#    get_attention=False))에 hook 쌍을 건다:
#      - forward_pre_hook(with_kwargs=True): 호출 직전에 kwargs에
#        get_attention=True를 주입해 GATv2Conv가 내부적으로 (out, attn) 튜플을
#        반환하도록 강제한다.
#      - forward_hook: (out, attn) 튜플을 받아 attn을 captured dict에 저장하고
#        out만 반환한다 — forward_hook의 반환값이 모듈 출력을 대체하므로,
#        이렇게 해야 GATlayer.forward의 `x = self.gatconv(graph, x)`가 여전히
#        평범한 텐서를 받아 이후 `x = x.reshape(last_shape)`가 깨지지 않는다.
#
#    확인한 속성 경로: EncoderDecoder.encode -> self.encoder(...) ->
#    Encoder.__init__에서 self.layer2s = clones(layer2, N//2) (layer2 ==
#    EncoderLayer_GAT 인스턴스) -> EncoderLayer_GAT.__init__에서
#    self.self_attn = self_attn (make_model()에서 attn_a = GATlayer(...)가
#    self_attn으로 전달됨) -> GATlayer.__init__에서 self.gatconv = GATv2Conv(...).
#    즉 backbone.encoder.layer2s[i].self_attn.gatconv 가 정확한 경로다.
# ---------------------------------------------------------------------------

def attach_gat_attention_hooks(backbone):
    """returns {layer_idx: tensor_or_None}; dict values populate after each forward() call."""
    captured = {}

    def make_pre_hook(idx):
        def pre_hook(module, args, kwargs):
            kwargs["get_attention"] = True
            return args, kwargs
        return pre_hook

    def make_hook(idx):
        def hook(module, args, output):
            out, attn = output
            captured[idx] = attn.detach()
            return out
        return hook

    for idx, layer2 in enumerate(backbone.encoder.layer2s):
        gatconv = layer2.self_attn.gatconv
        gatconv.register_forward_pre_hook(make_pre_hook(idx), with_kwargs=True)
        gatconv.register_forward_hook(make_hook(idx))
    return captured


# ---------------------------------------------------------------------------
# 6. fp16 masked_fill: pristine `attention()`의 `scores.masked_fill(mask == 0,
#    -1e10)`는 fp16(16-mixed precision)에서 -1e10이 fp16 표현 범위를 넘어 overflow를
#    일으킨다. 이 문제는 `patches/coformer-fp16-maskfill.patch`가 setup.sh 단계에서
#    external/coformer 소스에 직접 패치를 적용해 해결하므로, 여기서는 코드가 필요 없다.
# ---------------------------------------------------------------------------
