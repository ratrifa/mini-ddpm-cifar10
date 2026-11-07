"""Data pipeline CIFAR-10 untuk Mini DDPM.

Gambar dinormalisasi ke rentang [-1, 1] karena proses difusi menambahkan
noise Gaussian N(0, 1); data yang berpusat di nol dengan skala serupa
membuat gambar dan noise sebanding.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_dataloader(root="./data", batch_size=128, train=True, num_workers=2):
    """Mengembalikan DataLoader CIFAR-10 dengan gambar (B, 3, 32, 32) di [-1, 1] dan label (B,)."""
    augment = [transforms.RandomHorizontalFlip()] if train else []  # p=0.5, hanya data train
    transform = transforms.Compose(augment + [
        transforms.ToTensor(),                                   # [0, 255] -> [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0, 1] -> [-1, 1]
    ])
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
    )


if __name__ == "__main__":
    loader = get_dataloader(batch_size=64)
    images, labels = next(iter(loader))
    print("jumlah data :", len(loader.dataset))
    print("shape gambar:", tuple(images.shape))
    print("shape label :", tuple(labels.shape))
    print(f"rentang     : [{images.min():.3f}, {images.max():.3f}]")
    print("contoh label:", [CLASSES[i] for i in labels[:5].tolist()])
