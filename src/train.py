"""Training loop Mini DDPM (class-conditional, dengan classifier-free guidance dan EMA).

Contoh: python src/train.py --epochs 100 --ckpt-dir /content/drive/MyDrive/mini-ddpm/run1

Isi --ckpt-dir:
- last.pt       : checkpoint terakhir {model, ema, optimizer, epoch, step, config}; otomatis
                  dilanjutkan (resume) bila file ini sudah ada.
- loss_log.csv  : kolom epoch,loss,seconds (rata-rata loss MSE per epoch).
- loss_steps.csv: kolom step,epoch,loss (loss tiap langkah) untuk visualize.plot_loss_curve.
                  Ditulis di akhir tiap epoch, tepat sebelum checkpoint, agar tetap sinkron saat resume.
"""

import argparse
import copy
import csv
import os
import time

import torch
import torch.nn.functional as F

from data import get_dataloader
from diffusion import Diffusion
from model import UNet


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def update_ema(ema, model, decay):
    """ema = decay * ema + (1 - decay) * model, per parameter."""
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)


def save_checkpoint(state, path):
    """Tulis ke file sementara lalu rename, agar last.pt tidak rusak bila Colab terputus saat menyimpan."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--ch-mult", type=int, nargs="+", default=[1, 2, 2])
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--p-uncond", type=float, default=0.1, help="peluang label diganti token tanpa kelas")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--ckpt-dir", default="./checkpoints")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="batasi langkah per epoch (uji coba)")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    device = get_device()
    config = dict(base_ch=args.base_ch, ch_mult=tuple(args.ch_mult), num_res_blocks=args.num_res_blocks,
                  num_classes=10, dropout=args.dropout, timesteps=args.timesteps)
    model_kwargs = {k: v for k, v in config.items() if k != "timesteps"}

    model = UNet(**model_kwargs).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    diffusion = Diffusion(timesteps=args.timesteps)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "last.pt")
    log_path = os.path.join(args.ckpt_dir, "loss_log.csv")
    steps_path = os.path.join(args.ckpt_dir, "loss_steps.csv")
    start_epoch, global_step = 0, 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt["config"] != config:
            raise ValueError(f"config checkpoint {ckpt['config']} beda dengan argumen {config}; pakai --ckpt-dir lain")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt.get("step", 0)
        print(f"resume dari {ckpt_path}, epoch selesai: {start_epoch}")

    loader = get_dataloader(root=args.data_root, batch_size=args.batch_size, train=True,
                            num_workers=args.num_workers)
    print(f"device={device}, parameter={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, "
          f"langkah/epoch={args.steps_per_epoch or len(loader)}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total, n, start = 0.0, 0, time.time()
        step_losses = []
        for step, (x0, y) in enumerate(loader):
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
            x0, y = x0.to(device), y.to(device)
            t = torch.randint(0, args.timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)
            # classifier-free guidance: sebagian label diganti token tanpa kelas (indeks 10)
            drop = torch.rand(y.shape[0], device=device) < args.p_uncond
            y = torch.where(drop, torch.full_like(y, model.num_classes), y)

            loss = F.mse_loss(model(xt, t, y), noise)  # model memprediksi noise yang ditambahkan
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema, model, args.ema_decay)

            step_losses.append(loss.item())
            total += step_losses[-1]
            n += 1
            if (step + 1) % args.log_every == 0:
                print(f"  epoch {epoch + 1} langkah {step + 1}: loss {total / n:.4f}")

        avg, seconds = total / n, time.time() - start
        new_file = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["epoch", "loss", "seconds"])
            writer.writerow([epoch + 1, f"{avg:.6f}", f"{seconds:.1f}"])
        new_file = not os.path.exists(steps_path)
        with open(steps_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["step", "epoch", "loss"])
            for i, value in enumerate(step_losses, start=global_step + 1):
                writer.writerow([i, epoch + 1, f"{value:.6f}"])
        global_step += len(step_losses)
        save_checkpoint({"model": model.state_dict(), "ema": ema.state_dict(),
                         "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "step": global_step, "config": config}, ckpt_path)
        print(f"epoch {epoch + 1}/{args.epochs}: loss {avg:.4f} ({seconds:.0f} detik), tersimpan {ckpt_path}")


if __name__ == "__main__":
    main()
