"""Membuat grid sampel berlabel (nama kelas di kiri tiap baris) dari file .pt hasil sample.py.

Contoh: python src/label_grid.py outputs/run1/samples_g3.pt --out outputs/samples_g3_labeled.png
"""

import argparse

import torch

from visualize import show_sample_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_path")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    d = torch.load(args.pt_path)
    title = f"Sampel per kelas, guidance w={d['guidance_scale']:g}, epoch {d['epoch']}"
    print("tersimpan:", show_sample_grid(d["images"], d["labels"], args.out, title))
