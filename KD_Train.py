# -*- coding: utf-8 -*-
"""
KD_Train.py — 知识蒸馏训练脚本（移植到现有代码组）
目标：
  - 支持现有代码组中的所有模型（可充当教师或学生）
  - 数据与增强、日志/可视化、保存逻辑与 Single_Train.py 风格对齐
  - 稳健的 AMP 与早停/调度器
  - 典型增强：CIFAR 经典增强；支持 Mixup/CutMix；可选对称标签噪声；可选子集训练

依赖本项目文件：
  - DataSets.py  提供基础数据集：cifar10 / stl10 / cifar100
  - Models.py    统一模型工厂 get_model(...)
"""

import os, json, math, time, argparse, random
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DataSets import get_dataset, DATASET_META
from Models   import get_model


# ============== 实用工具 ==============
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def now_ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(1)
    return (preds == labels).float().mean().item() * 100.0

def save_curves(curves_path: str,
                hist: Dict[str, List[float]]):
    plt.figure(figsize=(11, 4))
    # 左：精度
    plt.subplot(1, 2, 1)
    plt.plot(hist["train_acc"], label="Train Acc")
    plt.plot(hist["val_acc"],   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    # 右：损失
    plt.subplot(1, 2, 2)
    plt.plot(hist["train_loss"], label="Train Loss")
    plt.plot(hist["val_loss"],   label="Val Loss")
    if hist.get("train_ce"): plt.plot(hist["train_ce"], label="Train CE")
    if hist.get("train_kd"): plt.plot(hist["train_kd"], label="Train KD")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(curves_path); plt.close()


class CSVLogger:
    def __init__(self, path: str, header: List[str]):
        self.path = path
        self.header = header
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")

    def log(self, row: Dict[str, Union[int, float, str]]):
        with open(self.path, "a", encoding="utf-8") as f:
            values = [row.get(h, "") for h in self.header]
            f.write(",".join([str(v) for v in values]) + "\n")


# ============== 蒸馏与增强 ==============
class DistillationLoss(nn.Module):
    """
    L = (1 - alpha) * CE(student_logits, y) + alpha * T^2 * KL( log_softmax(s/T), softmax(t/T) )
    - KL 使用 batchmean
    - 乘 T^2 以保持梯度量级
    - 支持 Mixup/CutMix 标签：targets 可能为 ((y_a, y_b), lam)
    - 支持 label smoothing
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0,
                 ce_weight: Optional[float] = None, label_smoothing: float = 0.0):
        super().__init__()
        assert 0.0 <= alpha <= 1.0 and temperature > 0.0
        self.alpha = alpha
        self.T = temperature
        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_weight = ce_weight

    def _ce(self, logits: torch.Tensor, targets: Union[torch.Tensor, tuple]):
        if isinstance(targets, tuple):
            (y_a, y_b), lam = targets
            return lam * self.ce(logits, y_a) + (1.0 - lam) * self.ce(logits, y_b)
        else:
            return self.ce(logits, targets)

    def forward(self, s_logits, t_logits, targets):
        T = self.T
        with torch.no_grad():
            p_t = torch.softmax(t_logits / T, dim=1)
        log_p_s = torch.log_softmax(s_logits / T, dim=1)
        kd = self.kldiv(log_p_s, p_t) * (T * T)

        ce_w = (1.0 - self.alpha) if self.ce_weight is None else self.ce_weight
        kd_w = self.alpha
        ce = self._ce(s_logits, targets)
        loss = ce_w * ce + kd_w * kd
        return loss, ce.detach(), kd.detach()


def do_mixup_cutmix(x, y, mixup_alpha: float, cutmix_alpha: float):
    if cutmix_alpha and cutmix_alpha > 0.0:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        B, C, H, W = x.size()
        rx, ry = random.random(), random.random()
        rw, rh = int(W * math.sqrt(1 - lam)), int(H * math.sqrt(1 - lam))
        cx, cy = int(rx * W), int(ry * H)
        x1, y1 = max(cx - rw // 2, 0), max(cy - rh // 2, 0)
        x2, y2 = min(cx + rw // 2, W), min(cy + rh // 2, H)
        index = torch.randperm(B, device=x.device)
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
        y_a, y_b = y, y[index]
        return x, ((y_a, y_b), lam)
    if mixup_alpha and mixup_alpha > 0.0:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        index = torch.randperm(x.size(0), device=x.device)
        mixed = lam * x + (1.0 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed, ((y_a, y_b), lam)
    return x, y


class NoisyLabels(Dataset):
    """对称标签噪声：以 prob 概率随机替换成其他类"""
    def __init__(self, base: Dataset, num_classes: int, prob: float, seed: int = 0):
        self.base = base
        self.num_classes = num_classes
        self.prob = prob
        g = random.Random(seed)
        # 预先采样翻转标签（确定性）
        self.flips = []
        for i in range(len(base)):
            flip = g.random() < prob
            if flip:
                # 生成一个 != y 的新标签
                self.flips.append(True)
            else:
                self.flips.append(False)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.prob > 0.0 and self.flips[idx]:
            new_y = random.randrange(self.num_classes)
            while new_y == y:
                new_y = random.randrange(self.num_classes)
            y = new_y
        return x, y


# ============== 数据与变换（与 Single_Train 风格对齐） ==============
C10_MEAN, C10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
C100_MEAN, C100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
STL_MEAN, STL_STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)

def build_transforms(dataset: str, img_size: int, train: bool):
    name = dataset.lower()
    if name == "cifar10":
        mean, std = C10_MEAN, C10_STD
    elif name == "cifar100":
        mean, std = C100_MEAN, C100_STD
    elif name == "stl10":
        mean, std = STL_MEAN, STL_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if train:
        tfm = []
        if img_size == 32:
            tfm += [transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip()]
        else:
            tfm += [transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip()]
        tfm += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(tfm)
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

def build_datasets_and_loaders(args):
    # 基于 DataSets.get_dataset 获取原始 TRAIN/TEST，再从 TRAIN 划出 VAL
    train_set = get_dataset(args.dataset, split="train", root=args.data_root, download=True,
                            transform=build_transforms(args.dataset, args.img_size, train=True))
    test_set  = get_dataset(args.dataset, split="test",  root=args.data_root, download=True,
                            transform=build_transforms(args.dataset, args.img_size, train=False))
    # 训练子集（可选）
    if args.subset_ratio < 1.0:
        n = int(len(train_set) * args.subset_ratio)
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(train_set), generator=g)[:n]
        train_set = Subset(train_set, idx.tolist())
    # 标签噪声（可选）
    num_classes = DATASET_META[args.dataset.lower()]["num_classes"]
    if args.label_noise > 0.0:
        train_set = NoisyLabels(train_set, num_classes=num_classes, prob=args.label_noise, seed=args.seed)

    # 划分验证集
    val_size = int(len(train_set) * args.val_split)
    if val_size > 0:
        train_size = len(train_set) - val_size
        g = torch.Generator().manual_seed(args.seed)
        train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size], generator=g)
    else:
        val_set = None

    pin = args.pin_memory and torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=max(args.batch_size, 256), shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=max(args.batch_size, 256), shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    return train_loader, val_loader, test_loader


# ============== AMP 封装 ==============
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def amp_autocast(enabled: bool): return _autocast("cuda", enabled=enabled)
    def make_scaler(enabled: bool):  return _GradScaler("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast_cuda, GradScaler as _GradScaler_cuda
    def amp_autocast(enabled: bool): return _autocast_cuda(enabled=enabled)
    def make_scaler(enabled: bool):  return _GradScaler_cuda(enabled=enabled)


# ============== 评估 ==============
@torch.no_grad()
def evaluate_ce(model: nn.Module, loader: DataLoader, device: torch.device,
                label_smoothing: float = 0.0) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), 100.0 * total_correct / max(total, 1)


# ============== 训练一个 epoch ==============
def train_one_epoch_kd(args, epoch: int,
                       student: nn.Module, teacher: nn.Module,
                       loader: DataLoader, device: torch.device,
                       optimizer: torch.optim.Optimizer,
                       scaler,
                       criterion: DistillationLoss):
    student.train()
    total_loss = total_ce = total_kd = 0.0
    total_correct = total = 0
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup/CutMix
        targets = labels
        if args.cutmix_alpha > 0.0 or args.mixup_alpha > 0.0:
            images, targets = do_mixup_cutmix(images, labels, args.mixup_alpha, args.cutmix_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            t_logits = teacher(images)

        with amp_autocast(args.amp):
            s_logits = student(images)
            loss, ce, kd = criterion(s_logits, t_logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_ce   += ce.item() * bs
        total_kd   += kd.item() * bs
        total_correct += (s_logits.argmax(1) == labels).sum().item()
        total += bs

    avg_loss = total_loss / max(total, 1)
    avg_ce   = total_ce   / max(total, 1)
    avg_kd   = total_kd   / max(total, 1)
    acc      = 100.0 * total_correct / max(total, 1)
    return avg_loss, avg_ce, avg_kd, acc


# ============== 调度器（Cosine + 线性 Warmup；或 StepLR；或 OneCycle） ==============
def build_scheduler(args, optimizer, iters_per_epoch: int):
    if args.scheduler == "cosine":
        # 线性 warmup（按 epoch 粗粒度）+ Cosine
        def lr_lambda(current_epoch):
            if args.warmup_epochs > 0 and current_epoch < args.warmup_epochs:
                return float(current_epoch + 1) / float(max(1, args.warmup_epochs))
            # 余下 epochs 做 cosine
            progress = (current_epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    elif args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=iters_per_epoch
        )
    else:
        return None


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description="KD_Train — 知识蒸馏（与 Single_Train 风格对齐）")
    # 数据
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "stl10", "cifar100"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1, help="从训练集中划出的验证比例，0 表示不用验证集")
    parser.add_argument("--subset-ratio", type=float, default=1.0, help="仅使用训练集前比例（欠拟合对照）")
    parser.add_argument("--label-noise", type=float, default=0.0, help="对称标签噪声概率 p∈[0,1]")
    # 硬件与运行
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="启用 AMP 混合精度")
    # 模型
    parser.add_argument("--teacher", type=str, default="resnet152")
    parser.add_argument("--student", type=str, default="resnet18")
    parser.add_argument("--teacher-ckpt", type=str, required=True, help="教师模型权重路径 (.pth)")
    # 优化
    parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "onecycle"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--step-gamma", type=float, default=0.1)
    # 蒸馏
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="KD 权重（越大越倚赖教师）")
    parser.add_argument("--ce-weight", type=float, default=None, help="覆盖 (1-alpha) 的显式权重")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    # 增强
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    # 早停/监控
    parser.add_argument("--monitor", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--improve-delta", type=float, default=0.05)
    parser.add_argument("--early-stop-patience", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=60)
    # 日志与保存
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--test-once", action="store_true", help="最后用 TEST 评估一次（载入 best）")

    args = parser.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 组件化元信息
    meta = DATASET_META[args.dataset]
    num_classes = int(meta["num_classes"])
    channels, H, W = meta["input_size"]

    set_seed(args.seed)
    device = torch.device(args.device)

    # 日志目录
    tag = args.exp_name if args.exp_name else f"{args.dataset}_{args.student}_by_{args.teacher}_{now_ts()}"
    log_dir = os.path.join(args.log_dir, tag)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # 数据与加载器
    train_loader, val_loader, test_loader = build_datasets_and_loaders(args)

    # 模型：教师（冻结）与学生
    small_input = (args.img_size <= 40)
    teacher = get_model(args.teacher, num_classes=num_classes, channels=channels, img_size=args.img_size)
    student = get_model(args.student, num_classes=num_classes, channels=channels, img_size=args.img_size)
    teacher.to(device); student.to(device)

    # 教师权重加载（兼容 module. / weights_only）
    if not os.path.exists(args.teacher_ckpt):
        raise FileNotFoundError(f"Teacher ckpt not found: {args.teacher_ckpt}")
    sd = torch.load(args.teacher_ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    # 去除可能的 DataParallel 前缀
    new_sd = {}
    for k, v in sd.items():
        new_sd[k.replace("module.", "")] = v
    teacher.load_state_dict(new_sd, strict=False)
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher.eval()

    # 优化器
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum,
                                    nesterov=True, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = make_scaler(enabled=args.amp)
    iters_per_epoch = len(train_loader)
    scheduler = build_scheduler(args, optimizer, iters_per_epoch)

    criterion = DistillationLoss(alpha=args.alpha, temperature=args.temperature,
                                 ce_weight=args.ce_weight, label_smoothing=args.label_smoothing).to(device)

    # 训练状态
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "train_ce", "train_kd"]}
    logger = CSVLogger(os.path.join(log_dir, "history.csv"),
                       ["epoch", "lr", "train_loss", "train_ce", "train_kd", "train_acc",
                        "val_loss", "val_acc", "best"])

    best_metric = float("-inf") if args.monitor == "val_acc" else float("inf")
    best_state = None
    epochs_no_improve = 0
    t0 = time.time()
    time_log_lines = []

    # ====== 训练循环 ======
    for epoch in range(1, args.epochs + 1):
        e_start = time.time()

        train_loss, train_ce, train_kd, train_acc = train_one_epoch_kd(
            args, epoch, student, teacher, train_loader, device, optimizer, scaler, criterion
        )

        # 验证（若无 val_split，则用 test 充当 val）
        eval_loader = val_loader if val_loader is not None else test_loader
        val_loss, val_acc = evaluate_ce(student, eval_loader, device, label_smoothing=args.label_smoothing)

        # 调度器 epoch 级 step（OneCycle 已在构造时绑定步数，这里也按 epoch step 一次以兼容）
        if scheduler is not None and args.scheduler in ("cosine", "step"):
            scheduler.step()

        # 记录
        lr_now = optimizer.param_groups[0]["lr"]
        hist["train_loss"].append(train_loss); hist["train_acc"].append(train_acc)
        hist["train_ce"].append(train_ce);     hist["train_kd"].append(train_kd)
        hist["val_loss"].append(val_loss);     hist["val_acc"].append(val_acc)
        save_curves(os.path.join(log_dir, "curves.png"), hist)

        # 打印+CSV
        cur_best = best_metric if best_state is not None else (val_acc if args.monitor == "val_acc" else val_loss)
        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"Loss={train_loss:.4f} (CE={train_ce:.4f}, KD={train_kd:.4f}) "
              f"| TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}% | ValLoss={val_loss:.4f} "
              f"| Best={cur_best:.2f} | LR={lr_now:.5f}")
        logger.log({
            "epoch": epoch, "lr": lr_now,
            "train_loss": train_loss, "train_ce": train_ce, "train_kd": train_kd, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "best": cur_best
        })

        # 显存信息
        if torch.cuda.is_available() and "cuda" in args.device:
            mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
            mem_alloc    = torch.cuda.memory_allocated(device) / (1024**2)
            mem_total    = torch.cuda.get_device_properties(device).total_memory / (1024**2)
            print(f"  [CUDA] reserved={mem_reserved:.0f}MB | allocated={mem_alloc:.0f}MB | total={mem_total:.0f}MB")

        # 早停/保存
        improved = False
        monitored = val_acc if args.monitor == "val_acc" else val_loss
        if args.monitor == "val_acc":
            improved = (monitored > best_metric + args.improve_delta)
        else:
            improved = (monitored < best_metric - args.improve_delta)

        if improved and epoch >= args.min_epochs:
            best_metric = monitored
            best_state = {
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best": best_metric,
                "args": vars(args),
            }
            torch.save(best_state, os.path.join(log_dir, "model_best.pth"))
            print(f"  >> New best saved (epoch={epoch}, {args.monitor}={best_metric:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 保存 last
        last_state = {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": best_metric if best_state is not None else None,
            "args": vars(args),
        }
        torch.save(last_state, os.path.join(log_dir, "model_last.pth"))

        # 记录时间
        e_dur = time.time() - e_start
        time_log_lines.append(f"Epoch {epoch:03d}: {e_dur:.2f}s")

        # 触发早停
        if epoch >= args.min_epochs and epochs_no_improve >= args.early_stop_patience:
            print(f"[EarlyStop] no improvement in {args.early_stop_patience} epochs; stop at {epoch}")
            break

    # 写入时间统计
    t1 = time.time()
    with open(os.path.join(log_dir, "time.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(time_log_lines) + "\n")
        f.write(f"Total: {t1 - t0:.2f}s\n")

    # ====== TEST（可选） ======
    if args.test_once and best_state is not None:
        print("\n[FINAL TEST] evaluating best checkpoint on TEST set ...")
        student.load_state_dict(best_state["model"])
        test_loss, test_acc = evaluate_ce(student, test_loader, device, label_smoothing=args.label_smoothing)
        print(f"[FINAL TEST] loss={test_loss:.4f}, acc={test_acc:.2f}%")

if __name__ == "__main__":
    main()
