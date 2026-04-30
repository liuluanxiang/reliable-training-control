import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def cifar_transform(train: bool):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_loaders(dataset: str, root: str, batch_size: int, num_workers: int, val_ratio: float, seed: int):
    dataset = dataset.lower()

    if dataset == "cifar10":
        train_full = datasets.CIFAR10(root=root, train=True, transform=cifar_transform(True), download=True)
        test_set = datasets.CIFAR10(root=root, train=False, transform=cifar_transform(False), download=True)
        num_classes = 10
    elif dataset == "cifar100":
        train_full = datasets.CIFAR100(root=root, train=True, transform=cifar_transform(True), download=True)
        test_set = datasets.CIFAR100(root=root, train=False, transform=cifar_transform(False), download=True)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes
