# -*- coding: utf-8 -*-
"""
Single_Train.py
- 规范数据管线：训练/验证 transforms 分离；CIFAR/STL 统计量 Normalize；CIFAR 经典增强
- 训练集切分验证：--val-split 从 TRAIN 中划出 VAL；TEST 仅在最后一次性评估（可用 --test-once）
- 调度器：Cosine + 线性 Warmup（可选）；也支持 StepLR
- 早停：基于验证集指标（val_loss/val_acc），支持 --min-epochs 保护期
- AMP：--amp 开启 torch.cuda.amp
- 日志：csv/曲线图/时间统计 + best/last checkpoint
"""
import os, json, math, time, random, argparse
from datetime import datetime
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DataSets import get_dataset, dataset_meta
from Models   import get_model


# ================ 实用函数 ================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ================ Label Smoothing CE ================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.0, reduction: str = "mean"):
        super().__init__()
        assert 0.0 <= eps < 1.0
        self.eps, self.reduction = eps, reduction

    def forward(self, logits, target):
        log_probs = torch.log_softmax(logits, dim=-1)
        nll    = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        loss   = (1.0 - self.eps) * nll + self.eps * smooth
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss


# ================ 对称标签噪声封装 ================
class NoisyLabelsDataset(Dataset):
    def __init__(self, base: Dataset, num_classes: int, noise_rate: float, seed: int = 42):
        super().__init__()
        assert 0.0 <= noise_rate < 1.0
        self.base, self.num_classes = base, num_classes
        rng = random.Random(seed)
        self._labels: List[int] = []
        for i in range(len(base)):
            _, y = base[i]
            if rng.random() < noise_rate:
                ny = y
                while ny == y:
                    ny = rng.randint(0, num_classes - 1)
                self._labels.append(ny)
            else:
                self._labels.append(y)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        return x, self._labels[idx]


# ================ 训练 & 验证 ================
def train_one_epoch(model, loader, optimizer, device, criterion, amp_enabled=False, log_interval=100):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, (images, targets) in enumerate(loader, 1):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        bs = targets.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_samples += bs

        if (step % log_interval == 0) or (step == len(loader)):
            avg_loss = total_loss / max(total_samples, 1)
            avg_acc  = 100.0 * total_correct / max(total_samples, 1)
            print(f"  [Train] step {step:4d}/{len(loader)} | avg_loss={avg_loss:.4f} | avg_acc={avg_acc:.2f}%")

    return total_loss / max(total_samples, 1), 100.0 * total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits  = model(images)
        loss    = criterion(logits, targets)

        bs = targets.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_samples += bs

    return total_loss / max(total_samples, 1), 100.0 * total_correct / max(total_samples, 1)


# ================ 可视化 ================
def save_curves(fig_path_loss: str, fig_path_acc: str, history: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"],   label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss Curves"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_path_loss); plt.close()

    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"],   label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)"); plt.title("Accuracy Curves"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_path_acc); plt.close()


# ================ 主流程 ================
def main():
    parser = argparse.ArgumentParser(description="Single Model Training with Logging")

    # 数据集/模型
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "stl10", "cifar100"])
    parser.add_argument("--model",   type=str, required=True)
    parser.add_argument("--img-size", type=int, default=0, help="0=use dataset default (CIFAR=32, STL=96)")

    # 数据处理
    parser.add_argument("--label-noise-rate", type=float, default=0.0)
    parser.add_argument("--label-smoothing",  type=float, default=0.0)
    parser.add_argument("--train-fraction",   type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="fraction from TRAIN as validation (recommended >0). If 0, will use TEST as val.")
    parser.add_argument("--test-once", action="store_true",
                        help="evaluate TEST set only once at the end using the best checkpoint")

    # 训练超参
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--gamma",     type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="linear warmup epochs for cosine")

    # 设备/并行
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training with torch.cuda.amp")

    # 早停
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=60, help="do not early-stop before this epoch")
    parser.add_argument("--monitor", type=str, default="val_loss", choices=["val_acc", "val_loss"])
    parser.add_argument("--improve-delta", type=float, default=0.0)

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--exp-name",  type=str, default="exp")
    parser.add_argument("--log-root",  type=str, default="./logs")
    parser.add_argument("--log-interval", type=int, default=100)

    args = parser.parse_args()
    set_seed(args.seed)

    # 设备
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # 数据集元信息
    meta = dataset_meta(args.dataset)
    num_classes  = meta["num_classes"]
    native_h     = meta["input_size"][1]   # (C,H,W)
    final_img_sz = args.img_size if args.img_size > 0 else native_h

    # ====== 统计量 ======
    CIFAR_STATS = {
        "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        "stl10":    ((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239)),
    }
    mean, std = CIFAR_STATS.get(args.dataset.lower(), ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    # ====== transforms：训练有增强；验证/测试仅 Normalize ======
    train_tfms, val_tfms = [], []
    if args.img_size > 0:
        resize = transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BILINEAR)
        train_tfms.append(resize); val_tfms.append(resize)

    if args.dataset.lower() in ("cifar10", "cifar100") and (args.img_size == 0 or args.img_size == 32):
        train_tfms += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    else:
        train_tfms += [transforms.RandomHorizontalFlip()]

    train_tfms += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    val_tfms   += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train_tfms = transforms.Compose(train_tfms)
    val_tfms   = transforms.Compose(val_tfms)
    print("[INFO] Train transforms:", train_tfms)
    print("[INFO] Val   transforms:", val_tfms)

    # ====== 构建底层数据集（两个 train 副本，确保 transform 不串） ======
    train_base_for_train = get_dataset(args.dataset, split="train", root=args.data_root, download=True, transform=train_tfms)
    train_base_for_val   = get_dataset(args.dataset, split="train", root=args.data_root, download=True, transform=val_tfms)
    test_set             = get_dataset(args.dataset, split="test",  root=args.data_root, download=True, transform=val_tfms)

    # 训练集抽样比例（在切分之前先决定索引集合）
    total_train = len(train_base_for_train)
    idx_all = list(range(total_train))
    if args.train_fraction < 1.0:
        assert args.train_fraction > 0.0
        keep = max(1, int(total_train * args.train_fraction))
        g = torch.Generator().manual_seed(args.seed + 123)
        perm = torch.randperm(total_train, generator=g).tolist()
        idx_all = perm[:keep]
        print(f"[INFO] Train fraction: {keep}/{total_train} ({args.train_fraction*100:.1f}%)")

    # 从 TRAIN 中再划出 VAL（推荐）
    if args.val_split > 0.0:
        n_total = len(idx_all)
        n_val   = int(round(n_total * args.val_split))
        n_train = n_total - n_val

        g = torch.Generator().manual_seed(args.seed + 456)
        perm = torch.randperm(n_total, generator=g).tolist()
        raw_idx_val   = [idx_all[i] for i in perm[:n_val]]
        raw_idx_train = [idx_all[i] for i in perm[n_val:]]

        train_set = Subset(train_base_for_train, raw_idx_train)   # 有随机增强
        val_set   = Subset(train_base_for_val,   raw_idx_val)     # 仅 Normalize
        print(f"[INFO] Split train-> train:{len(train_set)} / val:{len(val_set)} (val_split={args.val_split:.2f})")
    else:
        # 不推荐：直接用 TEST 做验证
        train_set = Subset(train_base_for_train, idx_all)
        val_set   = test_set
        print("[WARN] --val-split=0 : using TEST as validation (may overfit to test).")

    # 标签噪声仅施加到训练集
    if args.label_noise_rate > 0.0:
        train_set = NoisyLabelsDataset(train_set, num_classes=num_classes, noise_rate=args.label_noise_rate, seed=args.seed)
        print(f"[INFO] Applied symmetric label noise: rate={args.label_noise_rate}")

    # DataLoader
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_set, shuffle=True,  drop_last=False, **loader_args)
    val_loader   = DataLoader(val_set,   shuffle=False, drop_last=False, **loader_args)

    # 模型（Models.get_model 会根据 img_size 自动 small_input=True）
    model = get_model(args.model, num_classes=num_classes, channels=3, img_size=final_img_sz)
    model.to(device)

    # 输出 stem 供自检
    if hasattr(model, "conv1"):
        print("[INFO] Model stem:", getattr(model, "conv1", None))
        if hasattr(model, "max_pool"):
            print("[INFO] MaxPool:", model.max_pool)

    # 损失
    criterion = LabelSmoothingCrossEntropy(eps=args.label_smoothing) if args.label_smoothing > 0.0 else nn.CrossEntropyLoss()
    if args.label_smoothing > 0.0:
        print(f"[INFO] Using Label Smoothing eps={args.label_smoothing}")

    # 优化器
    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 调度器（Cosine + Warmup / Step / None）
    scheduler = None
    if args.scheduler == "cosine":
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
        if args.warmup_epochs > 0:
            warmup = LambdaLR(optimizer,
                              lr_lambda=lambda e: float(e + 1) / float(max(1, args.warmup_epochs)))
            cosine = CosineAnnealingLR(optimizer,
                                       T_max=max(1, args.epochs - args.warmup_epochs))
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                     milestones=[args.warmup_epochs])
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    # 日志目录
    stamp   = now_str()
    runname = f"{args.exp_name}_{args.dataset}_{args.model}_{stamp}"
    log_dir = os.path.join(args.log_root, runname)
    ensure_dir(log_dir)
    print(f"[INFO] Log dir: {log_dir}")

    with open(os.path.join(log_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(log_dir, "epoch_log.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr,epoch_time_sec\n")

    times_path = os.path.join(log_dir, "epoch_times.csv")
    with open(times_path, "w", encoding="utf-8") as f:
        f.write("epoch,time_sec\n")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # 早停初始化（统一成“更高更好”的度量）
    best_metric   = -math.inf  # 若 monitor=val_loss，我们会比较 -val_loss
    best_epoch    = 0
    patience_cnt  = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        ep_start = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, criterion,
                                          amp_enabled=args.amp, log_interval=args.log_interval)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        lr_val = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        ep_time = time.time() - ep_start
        print(f"[Epoch {epoch:03d}] Train: loss={tr_loss:.4f}, acc={tr_acc:.2f}% | "
              f"Val: loss={val_loss:.4f}, acc={val_acc:.2f}% | LR={lr_val:.6f} | time={ep_time:.2f}s")

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss);   history["val_acc"].append(val_acc)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.4f},{val_loss:.6f},{val_acc:.4f},{lr_val:.8f},{ep_time:.4f}\n")
        with open(times_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{ep_time:.4f}\n")

        # 保存 last
        torch.save({"epoch": epoch, "model": model.state_dict()},
                   os.path.join(log_dir, "model_last.pth"))

        # 统一比较指标（更高更好）：val_acc 或 -val_loss
        cur_metric = val_acc if args.monitor == "val_acc" else -val_loss
        improved = (cur_metric - best_metric) > args.improve_delta

        if improved:
            best_metric, best_epoch = cur_metric, epoch
            patience_cnt = 0
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       os.path.join(log_dir, "model_best.pth"))
            print(f"  >> New best saved at epoch {epoch} | monitor={args.monitor} | best={best_metric:.6f}")
        else:
            # 保护期内不早停
            if epoch < args.min_epochs:
                patience_cnt = 0
                print(f"  >> No improvement (under min-epochs={args.min_epochs}, patience not counted)")
            else:
                patience_cnt += 1
                print(f"  >> No improvement. Patience {patience_cnt}/{args.early_stop_patience}")
                if patience_cnt >= args.early_stop_patience:
                    print(f"[Early Stop] No improvement for {args.early_stop_patience} epochs.")
                    break

        # 曲线
        save_curves(os.path.join(log_dir, "loss_curve.png"),
                    os.path.join(log_dir, "acc_curve.png"),
                    history)

    total_time = time.time() - t0
    with open(os.path.join(log_dir, "total_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"total_time_sec={total_time:.4f}\n")
    print(f"\n[DONE] total training time: {total_time:.2f}s | best_epoch={best_epoch} | logdir={log_dir}")

    # ====== 可选：仅在最后 TEST 一次（用 best.pth） ======
    if args.test_once:
        print("\n[FINAL TEST] evaluating best checkpoint on TEST set ...")
        best = torch.load(os.path.join(log_dir, "model_best.pth"), map_location="cpu")["model"]
        model.load_state_dict(best)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                 pin_memory=args.pin_memory, shuffle=False, drop_last=False)
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        print(f"[FINAL TEST] loss={test_loss:.4f}, acc={test_acc:.2f}%")


if __name__ == "__main__":
    main()
