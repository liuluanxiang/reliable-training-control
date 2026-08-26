"""数据集读取、预处理及可复现的训练/验证/测试划分。"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


NORMALIZATION = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def image_transform(dataset: str, train: bool, image_size: int | None = None, normalization: str = "dataset"):
    """建立训练增强或确定性评估变换。

    训练集允许随机裁剪和翻转；验证/测试集只做确定性的缩放与中心裁剪。
    这一区分保证模型选择不会受到随机验证增强噪声影响。
    """

    dataset = dataset.lower()
    native_size = 64 if dataset == "tiny_imagenet" else 32
    size = int(image_size or native_size)
    norm_key = dataset if normalization == "dataset" and dataset in NORMALIZATION else normalization
    if norm_key == "dataset":
        norm_key = "imagenet"
    mean, std = NORMALIZATION[norm_key]

    if size > native_size:
        # DeiT 等 224 输入模型：训练用随机面积裁剪，评估用标准中心裁剪。
        spatial = [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ] if train else [
            transforms.Resize(round(size / 0.875)),
            transforms.CenterCrop(size),
        ]
    elif train:
        padding = 8 if native_size == 64 else 4
        spatial = [transforms.RandomCrop(native_size, padding=padding), transforms.RandomHorizontalFlip()]
    else:
        spatial = []
    return transforms.Compose([*spatial, transforms.ToTensor(), transforms.Normalize(mean, std)])


class TinyImageNetValidation(Dataset):
    """按官方 ``val_annotations.txt`` 解析 Tiny ImageNet 验证集。

    官方验证图像不按类别建子目录，不能直接用 ``ImageFolder``；这里复用训练
    集的 ``class_to_idx``，确保训练与测试标签编号完全一致。
    """

    def __init__(self, root: Path, class_to_idx: dict[str, int], transform=None):
        """解析官方标注并缓存 ``(图像路径, 类别索引)`` 列表。"""

        self.root = Path(root)
        self.transform = transform
        annotations = self.root / "val" / "val_annotations.txt"
        image_root = self.root / "val" / "images"
        if not annotations.exists():
            raise FileNotFoundError(f"Tiny ImageNet annotations not found: {annotations}")
        self.samples = []
        for line in annotations.read_text(encoding="utf-8").splitlines():
            filename, wnid, *_ = line.split("\t")
            if wnid not in class_to_idx:
                raise ValueError(f"Unknown Tiny ImageNet class {wnid} in {annotations}")
            self.samples.append((image_root / filename, class_to_idx[wnid]))

    def __len__(self):
        """返回官方验证图像数量。"""

        return len(self.samples)

    def __getitem__(self, index):
        """读取单张图像并返回 ``(C,H,W)`` 张量及整数类别标签。"""

        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _split_indices(length: int, val_ratio: float, seed: int):
    """用局部随机生成器产生固定的训练/验证互斥索引。"""

    val_size = int(length * val_ratio)
    train_size = length - val_size
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[:train_size], indices[train_size:]


def _cifar_datasets(dataset: str, root: str, train_transform, eval_transform, val_ratio: float, seed: int):
    """从官方训练集划出验证集，官方测试集仅留作最终一次评估。"""

    dataset_cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    # 两个 Dataset 共享原始样本但使用不同 transform，避免验证集继承训练增强。
    train_source = dataset_cls(root=root, train=True, transform=train_transform, download=True)
    validation_source = dataset_cls(root=root, train=True, transform=eval_transform, download=True)
    test_set = dataset_cls(root=root, train=False, transform=eval_transform, download=True)
    train_indices, validation_indices = _split_indices(len(train_source), val_ratio, seed)
    return (
        Subset(train_source, train_indices),
        Subset(validation_source, validation_indices),
        test_set,
        10 if dataset == "cifar10" else 100,
    )


def _tiny_imagenet_datasets(root: str, train_transform, eval_transform, val_ratio: float, seed: int):
    """构建 Tiny ImageNet 训练划分、内部验证划分和官方验证测试集。"""

    dataset_root = Path(root) / "tiny-imagenet-200"
    train_root = dataset_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"Tiny ImageNet training directory not found: {train_root}")
    train_source = datasets.ImageFolder(train_root, transform=train_transform)
    validation_source = datasets.ImageFolder(train_root, transform=eval_transform)
    if len(train_source.classes) != 200:
        raise ValueError(f"Tiny ImageNet must contain 200 classes, found {len(train_source.classes)}")
    train_indices, validation_indices = _split_indices(len(train_source), val_ratio, seed)
    test_set = TinyImageNetValidation(dataset_root, train_source.class_to_idx, transform=eval_transform)
    return Subset(train_source, train_indices), Subset(validation_source, validation_indices), test_set, 200


def build_loaders(
    dataset: str,
    root: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    image_size: int | None = None,
    normalization: str = "dataset",
):
    """建立三个数据加载器并返回 ``(train, val, test, num_classes)``。

    ``seed`` 同时固定数据子集划分和训练 DataLoader 的洗牌顺序。验证与测试
    不洗牌，便于复核 logits、targets 与派生指标的一一对应关系。
    """

    dataset = dataset.lower()
    train_transform = image_transform(dataset, True, image_size, normalization)
    eval_transform = image_transform(dataset, False, image_size, normalization)
    if dataset in {"cifar10", "cifar100"}:
        train_set, val_set, test_set, num_classes = _cifar_datasets(
            dataset, root, train_transform, eval_transform, val_ratio, seed
        )
    elif dataset == "tiny_imagenet":
        train_set, val_set, test_set, num_classes = _tiny_imagenet_datasets(
            root, train_transform, eval_transform, val_ratio, seed
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        # CUDA 训练可借助页锁定内存加速 H2D 拷贝；CPU 环境也保持接口一致。
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, num_classes
