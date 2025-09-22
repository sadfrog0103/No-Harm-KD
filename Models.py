# -*- coding: utf-8 -*-
"""
中文说明：
本文件仅包含“图像分类模型定义 + 统一工厂”，不包含任何训练逻辑。
支持三类模型：
  1) ResNet：复用 ResNet.py（通过 get_resnet 工厂实例化；小图自动 small_input=True）
  2) MobileNetV2：轻量级 CNN；小图时首层 stride=1，避免过早下采样
  3) ViT：简洁实现的 Vision Transformer（Patch Embedding + Encoder）
"""
from typing import Tuple, Union, Optional, Dict, Any
import math
import torch
import torch.nn as nn

# —— 复用 ResNet 工厂 ——
try:
    from ResNet import get_resnet  # get_resnet(name, num_classes, channels=3, small_input=False)
except Exception as e:
    get_resnet = None
    _resnet_import_error = e

# =========================
# 工具
# =========================
def _to_hw(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    assert isinstance(size, tuple) and len(size) == 2
    return size

# =========================
# MobileNetV2
# =========================
class ConvBNAct(nn.Sequential):
    """Conv2d + BN + ReLU6"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差块：expand → depthwise → project"""
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden_dim, kernel_size=1))
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        layers += [nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False), nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class MobileNetV2(nn.Module):
    """轻量级 CNN；small_input=True 时首层 stride=1，更适配 32×32/64×64"""
    def __init__(self,
                 num_classes: int,
                 channels: int = 3,
                 width_mult: float = 1.0,
                 round_nearest: int = 8,
                 last_channel: Optional[int] = None,
                 small_input: bool = False):
        super().__init__()

        interverted_settings = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        input_channel = _make_divisible(32 * width_mult, round_nearest)
        if last_channel is None:
            last_channel = _make_divisible(1280 * width_mult, round_nearest) if width_mult > 1.0 else 1280

        # stem：小图时 stride=1，避免早期信息流失
        stem_stride = 1 if small_input else 2
        features = [ConvBNAct(channels, input_channel, kernel_size=3, stride=stem_stride)]

        # blocks
        for t, c, n, s in interverted_settings:
            out_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, out_channel, stride=stride, expand_ratio=t))
                input_channel = out_channel

        # last conv
        features.append(ConvBNAct(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channel, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =========================
# Vision Transformer (ViT)
# =========================
class PatchEmbed(nn.Module):
    """把图像切成 patch，经 Conv 提升到 embed_dim"""
    def __init__(self, img_size: Union[int, Tuple[int,int]], patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        img_h, img_w = _to_hw(img_size)
        assert img_h % patch_size == 0 and img_w % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_h = img_h // patch_size
        self.grid_w = img_w // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, Gh, Gw = x.shape
        return x.flatten(2).transpose(1, 2)  # B, N, C

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out

class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size: Union[int, Tuple[int,int]],
                 patch_size: int,
                 num_classes: int,
                 in_chans: int = 3,
                 embed_dim: int = 192,
                 depth: int = 9,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0,
                 drop: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop  = nn.Dropout(drop)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, attn_drop=0.0, drop=drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)              # B, N, C
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed[:, :N+1])
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])

# =========================
# 统一工厂
# =========================
def get_model(
    name: str,
    num_classes: int,
    channels: int = 3,
    img_size: Union[int, Tuple[int, int]] = 32,
    **kwargs,
) -> nn.Module:
    n = name.lower()
    h, w = _to_hw(img_size)
    is_small = (max(h, w) <= 64)

    # —— ResNet ——
    if n.startswith("resnet"):
        if get_resnet is None:
            raise ImportError(f"Failed to import ResNet.get_resnet: {_resnet_import_error}")
        small_input = kwargs.pop("small_input", is_small)
        return get_resnet(n, num_classes=num_classes, channels=channels, small_input=small_input)

    # —— MobileNetV2 ——（小图时 stem stride=1）
    if n in ["mobilenetv2", "mbv2", "mobilev2"]:
        width_mult = float(kwargs.pop("width_mult", 1.0))
        round_nearest = int(kwargs.pop("round_nearest", 8))
        last_channel = kwargs.pop("last_channel", None)
        small_input = bool(kwargs.pop("small_input", is_small))
        return MobileNetV2(num_classes=num_classes, channels=channels,
                           width_mult=width_mult, round_nearest=round_nearest,
                           last_channel=last_channel, small_input=small_input)

    # —— ViT ——（小图默认用更小的 patch）
    if n in ["vit_tiny", "vit_small", "vit"]:
        if n == "vit_tiny":
            default = dict(patch_size=4 if is_small else 8, embed_dim=192, depth=9,  num_heads=3, mlp_ratio=4.0)
        elif n == "vit_small":
            default = dict(patch_size=4 if is_small else 8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0)
        else:
            default = dict(patch_size=4 if is_small else 16, embed_dim=256, depth=10, num_heads=4, mlp_ratio=4.0)
        cfg = {**default, **kwargs}
        return VisionTransformer(
            img_size=img_size,
            patch_size=int(cfg["patch_size"]),
            num_classes=num_classes,
            in_chans=channels,
            embed_dim=int(cfg["embed_dim"]),
            depth=int(cfg["depth"]),
            num_heads=int(cfg["num_heads"]),
            mlp_ratio=float(cfg["mlp_ratio"]),
            drop=float(cfg.get("drop", 0.0)),
        )

    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "get_model",
    "MobileNetV2",
    "VisionTransformer",
    "InvertedResidual",
    "TransformerBlock",
    "MultiHeadSelfAttention",
    "MLP",
    "PatchEmbed",
]

if __name__ == "__main__":
    x32 = torch.randn(2, 3, 32, 32)
    x96 = torch.randn(2, 3, 96, 96)
    if get_resnet is not None:
        m_r18 = get_model("resnet18", num_classes=10, channels=3, img_size=32)
        print("ResNet18:", m_r18(x32).shape)
    m_mbv2 = get_model("mobilenetv2", num_classes=100, channels=3, img_size=32)
    print("MobileNetV2:", m_mbv2(x32).shape)
    m_vit_t = get_model("vit_tiny", num_classes=100, channels=3, img_size=32, patch_size=4)
    print("ViT-tiny:", m_vit_t(x32).shape)
