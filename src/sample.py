"""Membuat gambar dari checkpoint training: n gambar per kelas, disimpan sebagai satu grid.

Contoh: python src/sample.py --ckpt /content/drive/MyDrive/mini-ddpm/run1/last.pt --out outputs/samples.png
Tiap baris grid = satu kelas (urutan data.CLASSES).
"""

import argparse

import torch
from torchvision.utils import save_image

from diffusion import Diffusion
from model import UNet
from train import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="outputs/samples.png")
    parser.add_argument("--n-per-class", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    model = UNet(**{k: v for k, v in config.items() if k != "timesteps"}).to(device)
    model.load_state_dict(ckpt["ema"])  # bobot EMA menghasilkan sampel lebih halus
    model.eval()
    diffusion = Diffusion(timesteps=config["timesteps"])

    torch.manual_seed(args.seed)
    labels = torch.arange(config["num_classes"], device=device).repeat_interleave(args.n_per_class)
    images = diffusion.sample(model, labels, guidance_scale=args.guidance_scale)
    save_image((images + 1) / 2, args.out, nrow=args.n_per_class)  # [-1, 1] -> [0, 1]
    print(f"epoch checkpoint {ckpt['epoch']}, guidance {args.guidance_scale}: "
          f"{tuple(images.shape)} rentang [{images.min():.2f}, {images.max():.2f}] -> {args.out}")


if __name__ == "__main__":
    main()
