# -*- coding: utf-8 -*-
# RKD_Train.py — fixed & aligned version (v3)
# Changes: add --rkd, --rkd-hidden, --rkd-gamma-init, --temperature alias; 
# robust DataSets/Models import; consistent transforms for student & teacher;
# safe KD kd_logits = s.detach() + gamma*delta; optional CE on combined;
# early stopping, logging, AMP, cosine warmup or step scheduler.

import os, json, math, time, argparse, random
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset

import torchvision
import torchvision.transforms as T
from torchvision import datasets

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")

_DEF_RESNETS = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet101": torchvision.models.resnet101,
    "resnet152": torchvision.models.resnet152,
}

def _build_torchvision_model(name: str, num_classes: int, img_size: int) -> nn.Module:
    name = name.lower()
    if name in _DEF_RESNETS:
        m = _DEF_RESNETS[name](weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name in ("mobilenetv2", "mnv2", "mobilev2"):
        m = torchvision.models.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name in ("vit_tiny", "vit_t16", "vit"):
        try:
            m = torchvision.models.vit_b_16(weights=None)
            m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
            return m
        except Exception:
            m = torchvision.models.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            return m
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def get_model(name: str, num_classes: int, img_size: int) -> nn.Module:
    try:
        import Models  # type: ignore
        if hasattr(Models, "get_model"):
            return Models.get_model(name, num_classes=num_classes, img_size=img_size)  # type: ignore
    except Exception:
        pass
    return _build_torchvision_model(name, num_classes, img_size)

_DATASET_META = {
    "cifar10":  {"num_classes": 10,  "mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
    "cifar100": {"num_classes": 100, "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
    "stl10":    {"num_classes": 10,  "mean": [0.4467, 0.4398, 0.4066], "std": [0.2241, 0.2215, 0.2239]},
}

def build_transforms(dataset: str, img_size: int, train: bool) -> T.Compose:
    mean = _DATASET_META[dataset]["mean"]
    std  = _DATASET_META[dataset]["std"]
    if train:
        tfms: List[Any] = []
        if img_size >= 32:
            tfms += [T.RandomCrop(img_size, padding=4)]
        tfms += [T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)]
        return T.Compose(tfms)
    else:
        return T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor(), T.Normalize(mean, std)])

def get_dataset(dataset: str, split: str, root: str, img_size: int, transform: Optional[Any] = None, download: bool = True):
    try:
        import DataSets  # type: ignore
        if hasattr(DataSets, "get_dataset"):
            return DataSets.get_dataset(dataset, split=split, root=root, transform=transform, download=download)  # type: ignore
    except Exception:
        pass
    dataset = dataset.lower()
    if transform is None:
        transform = build_transforms(dataset, img_size, train=(split == "train"))
    if dataset == "cifar10":
        return datasets.CIFAR10(root=root, train=(split == "train"), transform=transform, download=download)
    if dataset == "cifar100":
        return datasets.CIFAR100(root=root, train=(split == "train"), transform=transform, download=download)
    if dataset == "stl10":
        assert split in ("train", "test")
        return datasets.STL10(root=root, split=split, transform=transform, download=download)
    raise ValueError("Unsupported dataset: " + dataset)

class NoisyLabels(torch.utils.data.Dataset):
    def __init__(self, base, noise_rate: float, num_classes: int):
        self.base = base
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self._labels = []
        g = torch.Generator().manual_seed(12345)
        for _, y in base:
            if random.random() < noise_rate:
                ny = int(torch.randint(0, num_classes, (1,), generator=g).item())
                self._labels.append(ny)
            else:
                self._labels.append(int(y))
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, _ = self.base[idx]
        return x, self._labels[idx]

class RKDHead(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 0, gamma_init: float = 1.0, learn_gamma: bool = True):
        super().__init__()
        self.learn_gamma = learn_gamma
        self.hidden = int(hidden)
        if self.hidden > 0:
            self.lin = nn.Sequential(
                nn.Linear(num_classes, self.hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden, num_classes),
            )
            for m in self.lin.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            self.lin = nn.Linear(num_classes, num_classes, bias=True)
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)
        if learn_gamma:
            if gamma_init <= 0.0:
                init_val = -10.0
            else:
                init_val = math.log(math.exp(float(gamma_init)) - 1.0)
            self.gamma_param = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        else:
            self.register_buffer("gamma_const", torch.tensor(float(gamma_init)))
    def gamma(self) -> torch.Tensor:
        if self.learn_gamma:
            return torch.nn.functional.softplus(self.gamma_param)
        else:
            return self.gamma_const
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_det = s.detach()
        delta = self.lin(s_det)
        g = self.gamma()
        comb = s + g * delta
        kd_logits = s_det + g * delta
        return comb, kd_logits, delta

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.0):
        super().__init__()
        assert 0.0 <= eps < 1.0
        self.eps = eps
    def forward(self, logits, target):
        if self.eps == 0.0:
            return nn.functional.cross_entropy(logits, target)
        n = logits.size(-1)
        logp = nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(self.eps / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
        return torch.mean(torch.sum(-true_dist * logp, dim=-1))

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 4.0) -> torch.Tensor:
    log_p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
    p_t = nn.functional.softmax(teacher_logits / T, dim=-1)
    loss = nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return loss

class WarmupCosine:
    def __init__(self, optimizer: optim.Optimizer, base_lr: float, warmup_epochs: int, max_epochs: int):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warm = warmup_epochs
        self.total = max_epochs
        self.t = 0
    def step(self, epoch: int):
        self.t = epoch
        if self.t < self.warm:
            lr = self.base_lr * float(self.t + 1) / float(max(1, self.warm))
        else:
            t = self.t - self.warm
            Tt = max(1, self.total - self.warm)
            lr = 0.5 * (1 + math.cos(math.pi * t / Tt)) * self.base_lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

@torch.no_grad()
def evaluate(model: nn.Module, head: Optional[RKDHead], loader: DataLoader, device: torch.device,
             ce_fn: LabelSmoothingCrossEntropy, is_rkd: bool, rkd_l2: float = 0.0) -> Tuple[float, float]:
    model.eval()
    if head is not None:
        head.eval()
    total_loss, total_samples = 0.0, 0
    total_correct_base = 0
    total_correct_comb = 0
    total_correct_plain = 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        s = model(images)
        if is_rkd and head is not None:
            comb, kd_logits, delta = head(s)
            preds_comb = comb.argmax(1)
            preds_base = s.argmax(1)
            total_correct_comb += (preds_comb == targets).sum().item()
            total_correct_base += (preds_base == targets).sum().item()
            loss_ce = ce_fn(comb, targets)  # 评估时对 comb 计 CE
            loss = loss_ce + rkd_l2 * (delta.pow(2).mean())
        else:
            preds = s.argmax(1)
            total_correct_plain += (preds == targets).sum().item()
            loss = ce_fn(s, targets)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
    if is_rkd and head is not None:
        acc_best = 100.0 * max(total_correct_base, total_correct_comb) / max(1, total_samples)
        return total_loss / max(1, total_samples), acc_best
    return total_loss / max(1, total_samples), 100.0 * total_correct_plain / max(1, total_samples)


def train_one_epoch(model: nn.Module, head: Optional[RKDHead], teacher: Optional[nn.Module], loader: DataLoader,
                    optimizer: optim.Optimizer, device: torch.device, ce_fn: LabelSmoothingCrossEntropy,
                    amp: bool, kd_weight: float, ce_on_combined: bool, T: float, rkd_l2: float,
                    log_interval: int = 100) -> Tuple[float, float]:
    model.train()
    if head is not None:
        head.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    total_loss, total_correct, total_samples = 0.0, 0, 0
    total_correct_base, total_correct_comb = 0, 0

    for step, (images, targets) in enumerate(loader, 1):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            s = model(images)
            if head is not None and teacher is not None:
                comb, kd_logits, delta = head(s)
                with torch.no_grad():
                    t_logits = teacher(images)
                loss_kd = kd_loss(kd_logits, t_logits, T=T)
                if ce_on_combined:
                    loss_ce = ce_fn(comb, targets)
                    logits_for_acc = comb
                else:
                    loss_ce = ce_fn(s, targets)
                    logits_for_acc = s
                loss = (1.0 - kd_weight) * loss_ce + kd_weight * loss_kd + rkd_l2 * delta.pow(2).mean()
                preds_base = s.argmax(1)
                preds_comb = comb.argmax(1)
                total_correct_base += (preds_base == targets).sum().item()
                total_correct_comb += (preds_comb == targets).sum().item()
            else:
                loss = ce_fn(s, targets)
                logits_for_acc = s
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = targets.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs
        total_correct += (logits_for_acc.argmax(1) == targets).sum().item()

        if (step % log_interval == 0) or (step == len(loader)):
            cur_acc = 100.0 * total_correct / max(1, total_samples)
            if head is not None and teacher is not None:
                acc_base = 100.0 * total_correct_base / max(1, total_samples)
                acc_comb = 100.0 * total_correct_comb / max(1, total_samples)
                print(f"  [train] step {step:4d}/{len(loader):4d} | loss={total_loss/max(1,total_samples):.4f} | acc={cur_acc:.2f}% (base={acc_base:.2f} / comb={acc_comb:.2f})")
            else:
                print(f"  [train] step {step:4d}/{len(loader):4d} | loss={total_loss/max(1,total_samples):.4f} | acc={cur_acc:.2f}%")

    avg_loss = total_loss / max(1, total_samples)
    if head is not None and teacher is not None:
        avg_acc = 100.0 * max(total_correct_base, total_correct_comb) / max(1, total_samples)
    else:
        avg_acc = 100.0 * total_correct / max(1, total_samples)
    return avg_loss, avg_acc

class EarlyStopper:
    def __init__(self, monitor: str, patience: int, min_epochs: int = 0, improve_delta: float = 0.0, mode: Optional[str] = None):
        assert monitor in ("val_acc", "val_loss")
        self.monitor = monitor
        self.patience = patience
        self.min_epochs = min_epochs
        self.improve_delta = improve_delta
        self.wait = 0
        self.best = None
        self.mode = mode or ("max" if monitor == "val_acc" else "min")

    def is_better(self, cur: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return cur > (self.best + self.improve_delta)
        return cur < (self.best - self.improve_delta)

    def update(self, cur: float, epoch: int) -> bool:
        if self.is_better(cur):
            self.best = cur
            self.wait = 0
            return True
        else:
            if epoch >= self.min_epochs:
                self.wait += 1
            return False

    def should_stop(self) -> bool:
        return self.wait >= self.patience

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_curves(fig_loss: str, fig_acc: str, history: Dict[str, List[float]]):
    plt.figure(figsize=(7, 5), dpi=140)
    if "train_loss" in history: plt.plot(history["train_loss"], label="train_total")
    if "val_loss" in history:   plt.plot(history["val_loss"], label="val_total")
    if "train_ce" in history:   plt.plot(history["train_ce"], label="train_ce", linestyle="--")
    if "val_ce" in history:     plt.plot(history["val_ce"], label="val_ce", linestyle="--")
    if "train_kd" in history:   plt.plot(history["train_kd"], label="train_kd", linestyle=":")
    if "val_kd" in history:     plt.plot(history["val_kd"], label="val_kd", linestyle=":")
    if "train_rkd" in history:  plt.plot(history["train_rkd"], label="train_rkd", linestyle="-.")
    if "val_rkd" in history:    plt.plot(history["val_rkd"], label="val_rkd", linestyle="-.")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_loss); plt.close()

    # --- 精度曲线另存 ---
    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_acc); plt.close()



def _load_state_dict_from_ckpt(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported checkpoint structure")


def _clean_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    return new_sd


def main():
    parser = argparse.ArgumentParser("RKD_Train (fixed v3)")
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100", "stl10"])
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--data-root", type=str, default="./data")

    parser.add_argument("--label-noise-rate", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--step-epochs", type=int, nargs="*", default=[150, 225])
    parser.add_argument("--step-gamma", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--monitor", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--improve-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--test-once", action="store_true")

    parser.add_argument("--rkd", action="store_true")
    parser.add_argument("--teacher", type=str, default="resnet152")
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--T", dest="T", type=float, default=4.0)
    parser.add_argument("--temperature", dest="T", type=float)
    parser.add_argument("--ce-on-combined", action="store_true")
    parser.add_argument("--rkd-learn-gamma", action="store_true")
    parser.add_argument("--rkd-l2", type=float, default=0.0)
    parser.add_argument("--rkd-hidden", type=int, default=0)
    parser.add_argument("--rkd-gamma-init", type=float, default=1.0)

    parser.add_argument("--log-root", type=str, default="./logs")
    parser.add_argument("--exp-name", type=str, default=None)

    args = parser.parse_args()

    if args.config is not None and os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key):
                cur_val = getattr(args, key)
                if cur_val == parser.get_default(key) or cur_val is None:
                    setattr(args, key, v)
        if getattr(args, "T", None) is None and "temperature" in cfg:
            args.T = float(cfg["temperature"])  # type: ignore

    set_seed(int(args.seed))
    device = get_device(args.device)

    train_tfms = build_transforms(args.dataset, args.img_size, train=True)
    val_tfms = build_transforms(args.dataset, args.img_size, train=False)

    full_train = get_dataset(args.dataset, split="train", root=args.data_root, img_size=args.img_size, transform=train_tfms, download=True)
    test_set = get_dataset(args.dataset, split="test", root=args.data_root, img_size=args.img_size, transform=val_tfms, download=True)

    n_train = len(full_train)
    n_val = int(n_train * float(args.val_split))
    n_train_final = n_train - n_val
    train_set, val_set = random_split(full_train, [n_train_final, n_val], generator=torch.Generator().manual_seed(2025))

    if args.train_fraction < 1.0:
        n_sub = int(n_train_final * float(args.train_fraction))
        train_set = Subset(train_set, list(range(n_sub)))

    num_classes = _DATASET_META[args.dataset]["num_classes"]
    if args.label_noise_rate > 0.0:
        base = train_set
        class _Wrapper(torch.utils.data.Dataset):
            def __init__(self, subset, noise_rate, num_classes):
                self.subset = subset
                self.nl = NoisyLabels(subset.dataset, noise_rate, num_classes)
            def __len__(self):
                return len(self.subset)
            def __getitem__(self, idx):
                x, y = self.nl[self.subset.indices[idx]]
                return x, y
        train_set = _Wrapper(train_set, args.label_noise_rate, num_classes)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=args.pin_memory, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=args.pin_memory)

    model = get_model(args.model, num_classes=num_classes, img_size=args.img_size).to(device)

    use_rkd = bool(args.rkd) or (args.teacher_ckpt is not None)
    head: Optional[RKDHead] = None
    teacher: Optional[nn.Module] = None
    if use_rkd:
        head = RKDHead(num_classes=num_classes, hidden=int(args.rkd_hidden), gamma_init=float(args.rkd_gamma_init), learn_gamma=bool(args.rkd_learn_gamma)).to(device)
        assert args.teacher_ckpt is not None and os.path.exists(args.teacher_ckpt), "必须提供有效的 --teacher-ckpt"
        teacher = get_model(args.teacher, num_classes=num_classes, img_size=args.img_size).to(device)
        sd = _clean_keys(_load_state_dict_from_ckpt(args.teacher_ckpt))
        teacher.load_state_dict(sd, strict=False)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    if args.opt == "sgd":
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt == "adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = WarmupCosine(optimizer, base_lr=args.lr, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs)
    elif args.scheduler == "step":
        scheduler = None
    else:
        scheduler = None

    ce_fn = LabelSmoothingCrossEntropy(eps=float(args.label_smoothing))

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = args.exp_name or f"{args.dataset}_{args.model}_RKD_{stamp}"
    log_dir = os.path.join(args.log_root, exp_name)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "config_merged.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.val_only:
        print("[VAL-ONLY] evaluating on VAL & TEST and exit ...")
        val_loss, val_acc = evaluate(model, head, val_loader, device, ce_fn, is_rkd=use_rkd, rkd_l2=args.rkd_l2)
        print(f"VAL  | loss={val_loss:.4f}, acc={val_acc:.2f}%")
        test_loss, test_acc = evaluate(model, head, test_loader, device, ce_fn, is_rkd=use_rkd, rkd_l2=args.rkd_l2)
        print(f"TEST | loss={test_loss:.4f}, acc={test_acc:.2f}%")
        return

    history: Dict[str, List[float]] = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    logs: List[Dict[str, Any]] = []
    stopper = EarlyStopper(args.monitor, patience=int(args.early_stop_patience), min_epochs=int(args.min_epochs),
                           improve_delta=float(args.improve_delta))

    best_metric = None
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        if args.scheduler == "cosine":
            scheduler.step(epoch-1)
        elif args.scheduler == "step":
            if epoch in set(args.step_epochs):
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] * float(args.step_gamma)

        print(f"\nEpoch {epoch}/{args.epochs}")
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  LR={lr_now:.6f} | device={device} | amp={'on' if args.amp else 'off'} | RKD={'on' if use_rkd else 'off'}")

        e_t0 = time.time()
        if use_rkd:
            tr_loss, tr_acc = train_one_epoch(model, head, teacher, train_loader, optimizer, device, ce_fn,
                                              amp=args.amp, kd_weight=float(args.kd_weight), ce_on_combined=bool(args.ce_on_combined),
                                              T=float(args.T), rkd_l2=float(args.rkd_l2))
        else:
            tr_loss, tr_acc = train_one_epoch(model, None, None, train_loader, optimizer, device, ce_fn,
                                              amp=args.amp, kd_weight=0.0, ce_on_combined=False, T=1.0, rkd_l2=0.0)
        e_t1 = time.time()
        val_loss, val_acc = evaluate(model, head if use_rkd else None, val_loader, device, ce_fn, is_rkd=use_rkd, rkd_l2=float(args.rkd_l2))

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": e_t1 - e_t0,
        }
        logs.append(row)
        save_csv(os.path.join(log_dir, "train_log.csv"), logs)
        save_curves(os.path.join(log_dir, "loss_curve.png"), os.path.join(log_dir, "acc_curve.png"), history)

        metric = val_acc if args.monitor == "val_acc" else val_loss
        is_best = stopper.update(metric, epoch)
        if is_best:
            best_metric = metric
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "head": (head.state_dict() if head is not None else None)}, os.path.join(log_dir, "model_best.pth"))
            print(f"  >> New best saved (epoch={epoch}, {args.monitor}={metric:.4f})")
        else:
            print(f"  >> No improve | wait={stopper.wait}/{stopper.patience} | best@{best_epoch}={best_metric:.4f}")

        torch.save({"model": model.state_dict(), "head": (head.state_dict() if head is not None else None)}, os.path.join(log_dir, "model_last.pth"))

        if stopper.should_stop():
            print("\n[EARLY STOP] patience reached.")
            break

    total_time = time.time() - t0
    with open(os.path.join(log_dir, "total_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"total_time_sec={total_time:.3f}\n")
    print(f"\n[DONE] total_time={total_time:.2f}s | best_epoch={best_epoch} | logdir={log_dir}")

    if args.test_once:
        print("\n[FINAL TEST] evaluate best checkpoint on TEST ...")
        ck = torch.load(os.path.join(log_dir, "model_best.pth"), map_location="cpu")
        model.load_state_dict(ck["model"])
        if use_rkd and ck.get("head") is not None:
            head.load_state_dict(ck["head"])  # type: ignore
        test_loss, test_acc = evaluate(model, head if use_rkd else None, test_loader, device, ce_fn, is_rkd=use_rkd, rkd_l2=float(args.rkd_l2))
        print(f"TEST | loss={test_loss:.4f}, acc={test_acc:.2f}%")


if __name__ == "__main__":
    main()