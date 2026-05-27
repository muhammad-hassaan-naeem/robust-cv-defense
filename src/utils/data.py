"""
utils/data.py
-------------
Dataset loaders for CIFAR-10, MNIST, and GTSRB (stub).

All loaders return (train_loader, val_loader, test_loader) tuples
with standard PyTorch DataLoader objects.
"""

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split


# ──────────────────────────────────────────────────────────────────────────────
# CIFAR-10
# ──────────────────────────────────────────────────────────────────────────────
def get_cifar10(
    data_dir:   str  = "./data",
    batch_size: int  = 128,
    val_frac:   float = 0.1,
    num_workers: int  = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    CIFAR-10: 50k train / 10k test, 32×32 RGB, 10 classes.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # Normalisation disabled — we keep inputs in [0,1] for attack/defense
    ])
    test_tf = T.Compose([T.ToTensor()])

    train_full = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_ds    = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    val_size   = int(len(train_full) * val_frac)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


# ──────────────────────────────────────────────────────────────────────────────
# MNIST
# ──────────────────────────────────────────────────────────────────────────────
def get_mnist(
    data_dir:   str  = "./data",
    batch_size: int  = 128,
    val_frac:   float = 0.1,
    num_workers: int  = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    MNIST: 60k train / 10k test, 28×28 grayscale, 10 classes.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    tf = T.Compose([T.ToTensor()])

    train_full = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    test_ds    = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=tf)

    val_size   = int(len(train_full) * val_frac)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ──────────────────────────────────────────────────────────────────────────────
DATASETS = {
    "cifar10": get_cifar10,
    "mnist":   get_mnist,
}

def get_dataset(
    name:       str,
    data_dir:   str  = "./data",
    batch_size: int  = 128,
    val_frac:   float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory: returns (train_loader, val_loader, test_loader) for the given dataset.

    Supported: 'cifar10', 'mnist'
    """
    name = name.lower()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    return DATASETS[name](data_dir=data_dir, batch_size=batch_size, val_frac=val_frac)
