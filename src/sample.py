"""Membuat gambar dari checkpoint training: n gambar per kelas untuk satu atau beberapa guidance scale.

Contoh:
    python src/sample.py --ckpt /content/drive/MyDrive/mini-ddpm/run1/last.pt --out outputs/samples.png
    python src/sample.py --ckpt .../last.pt --out outputs/samples.png --guidance-scales 1 3 5

Untuk tiap guidance scale disimpan dua file:
- <nama>.png : grid mentah, tiap baris = satu kelas (urutan data.CLASSES)
- <nama>.pt  : dict {"images": float32 (N, 3, 32, 32) di [-1, 1], "labels": long (N,),
                     "guidance_scale": float, "epoch": int}, semua di CPU.
               Bisa langsung dipakai: visualize.show_sample_grid(d["images"], d["labels"], ...)
Bila hanya satu guidance scale, <nama> = --out apa adanya. Bila lebih dari satu, ditambah akhiran
_g<scale>, misalnya samples_g1.png, samples_g3.png, samples_g5.png.
Seed di-reset untuk tiap scale, jadi noise awal sama dan hasil antar-scale bisa dibandingkan langsung.
"""

import argparse
import os

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
    # --guidance-scale (bentuk lama) tetap diterima agar cell notebook lama tidak rusak
    parser.add_argument("--guidance-scales", "--guidance-scale", type=float, nargs="+", default=[3.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    model = UNet(**{k: v for k, v in config.items() if k != "timesteps"}).to(device)
    model.load_state_dict(ckpt["ema"])  # bobot EMA menghasilkan sampel lebih halus
    model.eval()
    diffusion = Diffusion(timesteps=config["timesteps"])
    labels = torch.arange(config["num_classes"], device=device).repeat_interleave(args.n_per_class)

    stem, ext = os.path.splitext(args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for scale in args.guidance_scales:
        base = stem if len(args.guidance_scales) == 1 else f"{stem}_g{scale:g}"
        torch.manual_seed(args.seed)
        images = diffusion.sample(model, labels, guidance_scale=scale)
        save_image((images + 1) / 2, base + ext, nrow=args.n_per_class)  # [-1, 1] -> [0, 1]
        torch.save({"images": images.cpu(), "labels": labels.cpu(),
                    "guidance_scale": scale, "epoch": ckpt["epoch"]}, base + ".pt")
        print(f"epoch checkpoint {ckpt['epoch']}, guidance {scale:g}: {tuple(images.shape)} "
              f"rentang [{images.min():.2f}, {images.max():.2f}] -> {base + ext}, {base}.pt")


if __name__ == "__main__":
    main()
