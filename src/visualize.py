"""Visualisasi untuk Mini DDPM: contoh dataset (dan nanti kurva loss serta grid sampel)."""

import os

import matplotlib
matplotlib.use("Agg")  # simpan ke file tanpa membuka jendela (aman di Colab/terminal)
import matplotlib.pyplot as plt

from data import CLASSES


def to_display(x):
    """Mengubah tensor gambar (3, H, W) dari [-1, 1] ke array (H, W, 3) di [0, 1] untuk matplotlib."""
    x = (x + 1) / 2
    return x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def show_dataset_grid(dataset, out_path="outputs/dataset_samples.png", per_class=8):
    """Menyimpan grid 10 baris (satu per kelas) x per_class gambar, nama kelas di kiri tiap baris."""
    # kumpulkan indeks per_class gambar pertama untuk tiap kelas
    picks = {c: [] for c in range(len(CLASSES))}
    for idx, label in enumerate(dataset.targets):
        if len(picks[label]) < per_class:
            picks[label].append(idx)
        if all(len(v) == per_class for v in picks.values()):
            break

    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.1, len(CLASSES) * 1.1))
    for c, indices in picks.items():
        for j, idx in enumerate(indices):
            ax = axes[c, j]
            ax.imshow(to_display(dataset[idx][0]))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[c, 0].set_ylabel(CLASSES[c], rotation=0, ha="right", va="center", fontsize=10)

    fig.suptitle("Contoh dataset CIFAR-10 per kelas")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    from data import get_dataloader

    dataset = get_dataloader(train=False, num_workers=0).dataset  # data test: tanpa flip
    print("tersimpan:", show_dataset_grid(dataset))
