# DataSets.py
# -*- coding: utf-8 -*-
"""
中文说明：
本文件仅负责三个公开图像分类数据集的“获取/下载”，不涉及训练环节的任何超参数（如 epoch、batch 等）。
按分类难度从易到难依次包含：CIFAR-10、STL-10（仅使用有标签部分）、CIFAR-100。
提供统一入口 get_dataset(...) 与三个专用函数，默认只做 ToTensor() 转换，方便后续在训练脚本中自定义增强。
"""

from typing import Optional, Literal, Dict, Any
from torchvision import datasets, transforms

# —— 数据集的基础元信息（中文注释）—————————————
# 说明：仅供查询显示，不影响获取逻辑
DATASET_META: Dict[str, Dict[str, Any]] = {
    "cifar10":   {"num_classes": 10,  "input_size": (3, 32, 32)},
    "stl10":     {"num_classes": 10,  "input_size": (3, 96, 96)},  # 官方原始分辨率为 96×96
    "cifar100":  {"num_classes": 100, "input_size": (3, 32, 32)},
}

# —— 默认的基础变换（中文注释）—————————————
# 说明：为了保持本文件“只负责获取数据集”，此处仅提供 ToTensor()，不包含归一化/增强/缩放等训练相关配置
def _default_transform():
    """中文：返回默认的基础变换，仅将 PIL 图或 ndarray 转为张量"""
    return transforms.ToTensor()


# —— 专用获取函数（中文注释）—————————————
def get_cifar10(
    root: str = "./data",
    split: Literal["train", "test"] = "train",
    download: bool = True,
    transform = None,
):
    """中文：获取 CIFAR-10 数据集。
    参数：
      - root: 数据存放目录
      - split: "train" 或 "test"
      - download: 如本地不存在则下载
      - transform: 可自定义变换，默认仅 ToTensor()
    """
    if transform is None:
        transform = _default_transform()
    train_flag = (split == "train")
    return datasets.CIFAR10(root=root, train=train_flag, download=download, transform=transform)


def get_stl10(
    root: str = "./data",
    split: Literal["train", "test"] = "train",
    download: bool = True,
    transform = None,
):
    """中文：获取 STL-10 的“有标签部分”（仅允许 train/test）。
    注意：
      - STL-10 额外存在 "unlabeled"（无标签）与 "train+unlabeled" 等 split，本项目刻意不提供，以确保“仅取有标签的部分”。
    参数含义同上。
    """
    if transform is None:
        transform = _default_transform()
    if split not in ("train", "test"):
        raise ValueError("STL10 仅支持有标签的 split: 'train' 或 'test'，不支持 'unlabeled' 等。")
    # torchvision 的 STL10 使用 split="train"/"test"/"unlabeled"/"train+unlabeled"
    return datasets.STL10(root=root, split=split, download=download, transform=transform)


def get_cifar100(
    root: str = "./data",
    split: Literal["train", "test"] = "train",
    download: bool = True,
    transform = None,
):
    """中文：获取 CIFAR-100 数据集。参数含义同上。"""
    if transform is None:
        transform = _default_transform()
    train_flag = (split == "train")
    return datasets.CIFAR100(root=root, train=train_flag, download=download, transform=transform)


# —— 统一入口（中文注释）—————————————
def get_dataset(
    name: Literal["cifar10", "stl10", "cifar100"],
    split: Literal["train", "test"] = "train",
    root: str = "./data",
    download: bool = True,
    transform = None,
):
    """中文：统一的数据集获取接口。
    使用示例：
        ds_train = get_dataset("cifar10", split="train", root="./data")
        ds_test  = get_dataset("stl10",   split="test",  root="./data")

    说明：
      - 为保持本文件“只负责数据集获取”，transform 默认仅 ToTensor()。
      - 若后续需要统一尺寸（例如将 STL-10 从 96×96 缩放到 32×32 以对齐 CIFAR），请在训练/预处理代码中自行传入 transform。
    """
    name_l = name.lower()
    if name_l not in DATASET_META:
        raise ValueError(f"Unsupported dataset name: {name}. Choose from: 'cifar10', 'stl10', 'cifar100'.")

    if transform is None:
        transform = _default_transform()

    if name_l == "cifar10":
        return get_cifar10(root=root, split=split, download=download, transform=transform)
    elif name_l == "stl10":
        return get_stl10(root=root, split=split, download=download, transform=transform)
    elif name_l == "cifar100":
        return get_cifar100(root=root, split=split, download=download, transform=transform)

    # 理论上不会到达此处
    raise RuntimeError("Unknown dataset branch reached.")


# —— 实用的元信息查询函数（中文注释）—————————————
def dataset_meta(name: Literal["cifar10", "stl10", "cifar100"]) -> Dict[str, Any]:
    """中文：返回指定数据集的元信息（类别数、原始输入尺寸等）"""
    key = name.lower()
    if key not in DATASET_META:
        raise ValueError(f"Unsupported dataset name: {name}.")
    return DATASET_META[key]


__all__ = [
    "get_dataset",
    "get_cifar10",
    "get_stl10",
    "get_cifar100",
    "dataset_meta",
    "DATASET_META",
]


# —— 最小可运行示例（中文注释）—————————————
# 说明：直接运行本文件，将在 ./data 目录下下载三个数据集，并打印样本数。
if __name__ == "__main__":
    # 按难度顺序：CIFAR-10 -> STL-10(有标签) -> CIFAR-100
    d1_train = get_dataset("cifar10", split="train", root="./data", download=True)
    d1_test  = get_dataset("cifar10", split="test",  root="./data", download=True)
    print(f"[CIFAR-10] train={len(d1_train)}, test={len(d1_test)}")

    d2_train = get_dataset("stl10", split="train", root="./data", download=True)
    d2_test  = get_dataset("stl10", split="test",  root="./data", download=True)
    print(f"[STL-10]  train={len(d2_train)}, test={len(d2_test)} (labeled only)")

    d3_train = get_dataset("cifar100", split="train", root="./data", download=True)
    d3_test  = get_dataset("cifar100", split="test",  root="./data", download=True)
    print(f"[CIFAR-100] train={len(d3_train)}, test={len(d3_test)}")
