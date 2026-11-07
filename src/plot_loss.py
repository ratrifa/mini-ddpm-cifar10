"""Menggambar kurva loss dari loss_steps.csv hasil train.py.

Contoh: python src/plot_loss.py /content/drive/MyDrive/mini-ddpm/run1/loss_steps.csv --out outputs/loss_curve.png
"""

import argparse

from visualize import plot_loss_curve, read_step_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--out", default="outputs/loss_curve.png")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    losses = read_step_losses(args.csv_path)
    print(f"{len(losses)} langkah, loss awal {losses[0]:.4f}, loss akhir {losses[-1]:.4f}")
    print("tersimpan:", plot_loss_curve(losses, args.out, window=args.window))
