"""实验模型工厂：统一建立 CNN 与 DeiT 分类器。"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained_checkpoint: str | None = None):
    """按配置建立分类模型并返回未迁移设备的 ``nn.Module``。

    Args:
        name: 配置文件中的模型标识。
        num_classes: 数据集类别数，决定最终分类头输出维度。
        pretrained_checkpoint: DeiT 官方预训练权重的本地路径；CNN 实验从头训练。

    ResNet/VGG 的结构修改用于保留 32x32 或 64x64 小图像上的空间信息；
    DeiT 保持 224x224 官方结构，以便严格加载预训练骨干。
    """

    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
        # ImageNet 默认 7x7/stride-2 stem 对小图像下采样过强，改为 CIFAR stem。
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

    if name == "resnet50":
        model = models.resnet50(weights=None, num_classes=num_classes)
        # 与 ResNet-18 使用相同的小图像输入协议，保证模型对比口径一致。
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

    if name == "vgg16":
        model = models.vgg16_bn(weights=None, num_classes=num_classes)
        # 自适应池化将任意实验输入尺寸压到 1x1，分类头输入固定为 512 维。
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        return model

    if name == "deit_tiny_patch16_224":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("DeiT requires timm; install requirements.txt in the project environment.") from exc
        # 先建立目标类别数模型，再显式加载本地权重，保证权重来源可哈希审计。
        model = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        if pretrained_checkpoint:
            checkpoint_path = Path(pretrained_checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"DeiT checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = dict(checkpoint.get("model", checkpoint))
            # 官方分类头为 ImageNet-1K；丢弃它并保留新建的数据集专用分类头。
            state.pop("head.weight", None)
            state.pop("head.bias", None)
            incompatible = model.load_state_dict(state, strict=False)
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            # 除分类头外的任何缺失/多余键都意味着骨干权重与结构不兼容。
            if unexpected or not missing.issubset({"head.weight", "head.bias"}):
                raise RuntimeError(
                    f"Incompatible DeiT checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
        return model

    raise ValueError(f"Unsupported model: {name}")
