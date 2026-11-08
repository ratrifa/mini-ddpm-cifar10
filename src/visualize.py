"""Visualisasi untuk Mini DDPM: contoh dataset, kurva loss training, dan grid sampel per kelas."""

import csv
import os

import matplotlib
matplotlib.use("Agg")  # simpan ke file tanpa membuka jendela (aman di Colab/terminal)
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def show_forward_process(images, diffusion, timesteps=(0, 100, 200, 300, 400, 500, 700, 999),
                         out_path="outputs/forward_process.png", row_labels=None):
    """Menyimpan grid forward process: tiap baris satu gambar, makin ke kanan makin noisy.

    images     : tensor (N, 3, H, W) di [-1, 1]; kolom pertama gambar asli x0
    diffusion  : objek Diffusion (src/diffusion.py); x_t dibuat dengan diffusion.q_sample
    timesteps  : indeks t (0..T-1) untuk kolom berikutnya
    row_labels : opsional, teks di kiri tiap baris (misal nama kelas)
    Satu noise dipakai untuk semua timestep pada gambar yang sama, jadi perbedaan antarkolom
    hanya berasal dari t (porsi sinyal sqrt(alpha_bar_t) yang ditulis di judul kolom).
    """
    n = images.shape[0]
    ts = torch.tensor(timesteps, dtype=torch.long)
    noise = torch.randn_like(images)

    fig, axes = plt.subplots(n, len(ts) + 1, figsize=((len(ts) + 1) * 1.2, n * 1.2 + 0.4), squeeze=False)
    for i in range(n):
        x0 = images[i:i + 1].repeat(len(ts), 1, 1, 1)
        xt = diffusion.q_sample(x0, ts, noise[i:i + 1].expand(len(ts), -1, -1, -1))
        for j, img in enumerate([images[i]] + list(xt)):
            ax = axes[i, j]
            ax.imshow(to_display(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        if row_labels is not None:
            axes[i, 0].set_ylabel(row_labels[i], rotation=0, ha="right", va="center", fontsize=10)

    axes[0, 0].set_title("x0 (asli)", fontsize=9)
    for j, t in enumerate(timesteps, start=1):
        axes[0, j].set_title(f"t={t}\n√ᾱ={diffusion.sqrt_alpha_bars[t]:.2f}", fontsize=9)

    fig.suptitle("Forward process q(x_t | x_0): gambar makin noisy seiring t")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def read_step_losses(csv_path):
    """Membaca loss_steps.csv dari train.py (kolom step,epoch,loss) menjadi list loss urut per step.

    Bila satu step tercatat dua kali (Colab putus setelah CSV ditulis tetapi sebelum checkpoint
    tersimpan, lalu epoch itu diulang saat resume), nilai yang dipakai adalah yang terakhir.
    """
    by_step = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_step[int(row["step"])] = float(row["loss"])
    return [by_step[s] for s in sorted(by_step)]


def plot_loss_curve(losses, out_path="outputs/loss_curve.png", window=100):
    """Menyimpan kurva loss training (satu nilai per step) beserta rata-rata bergeraknya.

    Loss per step DDPM sangat berisik karena timestep t dan noise diambil acak tiap batch,
    jadi tren dibaca dari garis rata-rata bergerak (window step terakhir), bukan dari nilai mentah.
    Sumbu y logaritmik karena loss turun cepat di awal lalu melandai.
    """
    losses = np.asarray(losses, dtype=float)
    steps = np.arange(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, color="tab:blue", alpha=0.25, linewidth=0.8, label="loss per step")
    if len(losses) >= window:
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smooth, color="tab:blue", linewidth=2,
                label=f"rata-rata {window} step")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss (noise)")
    ax.set_title("Kurva loss training")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_epoch_loss_compare(runs, out_path="outputs/loss_compare.png", max_epoch=None):
    """Menyimpan grafik loss rata-rata per epoch beberapa run dalam satu sumbu.

    runs      : dict {nama run: path loss_log.csv (kolom epoch,loss,seconds dari train.py)}
    max_epoch : bila diisi, hanya epoch 1..max_epoch yang digambar (agar run beda panjang setara)
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, csv_path in runs.items():
        with open(csv_path, newline="") as f:
            rows = [(int(r["epoch"]), float(r["loss"])) for r in csv.DictReader(f)]
        rows = [r for r in rows if max_epoch is None or r[0] <= max_epoch]
        ax.plot([e for e, _ in rows], [l for _, l in rows], marker="o", markersize=3, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss rata-rata per epoch")
    ax.set_title("Perbandingan loss per epoch")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def show_sample_grid(samples, labels, out_path="outputs/samples_per_class.png", title="Sampel hasil generate per kelas"):
    """Menyimpan grid sampel hasil generate: satu baris per kelas, nama kelas di kiri tiap baris.

    samples : tensor (N, 3, H, W) di rentang [-1, 1] (keluaran sampling, belum di-clamp tidak apa-apa)
    labels  : tensor/list (N,) berisi indeks kelas tiap sampel
    Jumlah kolom = jumlah sampel terbanyak di satu kelas; kelas yang sampelnya lebih sedikit
    menyisakan sel kosong.
    """
    labels = [int(c) for c in labels]
    rows = {c: [i for i, lab in enumerate(labels) if lab == c] for c in sorted(set(labels))}
    n_cols = max(len(v) for v in rows.values())

    fig, axes = plt.subplots(len(rows), n_cols, figsize=(n_cols * 1.1, len(rows) * 1.1), squeeze=False)
    for r, (c, indices) in enumerate(rows.items()):
        for j in range(n_cols):
            ax = axes[r, j]
            if j < len(indices):
                ax.imshow(to_display(samples[indices[j]]))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[r, 0].set_ylabel(CLASSES[c], rotation=0, ha="right", va="center", fontsize=10)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys

    from data import get_dataloader
    from diffusion import Diffusion

    root = sys.argv[1] if len(sys.argv) > 1 else "./data"  # lokasi CIFAR-10 bisa dioper lewat argumen
    dataset = get_dataloader(root=root, train=False, num_workers=0).dataset  # data test: tanpa flip
    print("tersimpan:", show_dataset_grid(dataset))

    # forward process: gambar test pertama dari tiap kelas, noise tetap (seed 0) agar hasil bisa diulang
    torch.manual_seed(0)
    first = {}
    for idx, label in enumerate(dataset.targets):
        first.setdefault(label, idx)
        if len(first) == len(CLASSES):
            break
    images = torch.stack([dataset[first[c]][0] for c in range(len(CLASSES))])
    print("tersimpan:", show_forward_process(images, Diffusion(), row_labels=CLASSES))
