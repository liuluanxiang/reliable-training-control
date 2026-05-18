import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int):
    name = name.lower()

    if name == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

    if name == "resnet50":
        model = models.resnet50(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

    if name == "vgg16":
        model = models.vgg16_bn(weights=None, num_classes=num_classes)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        return model

    raise ValueError(f"Unsupported model: {name}")
