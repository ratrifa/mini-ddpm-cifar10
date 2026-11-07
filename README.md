# mini-ddpm-cifar10

Class-conditional Mini DDPM (Denoising Diffusion Probabilistic Model) dibangun dari nol dengan PyTorch, dilatih pada CIFAR-10 (32×32) di Google Colab.

Proyek 1 (Vibe Coding) — CAK4KBB3 Generative AI 2026-1, Kelompok 14.

## Anggota
| Nama | NIM | Peran |
|---|---|---|
| Satria Febry Andanu (Ketua) | 103012300372 | Prompt Engineer & LLM Navigator · Training Loop & Hyperparameter Tuning |
| Gifan Apriadi | 103012300385 | Data Pipeline & Preprocessing · Lead Documentarian & Error Handler |

Rincian tugas: [JOBDESK.md](JOBDESK.md)

## Struktur
```
src/        kode model, diffusion, data pipeline, training
notebooks/  notebook Colab untuk training & sampling
logs/       Log Interaksi LLM per sesi
outputs/    contoh hasil (grid sampel, kurva loss)
```

| File | Isi |
|---|---|
| `src/data.py` | DataLoader CIFAR-10, gambar `(B, 3, 32, 32)` di [-1, 1], flip horizontal untuk data train |
| `src/diffusion.py` | Noise schedule linear (T = 1000), forward process `q_sample`, reverse process `sample` dengan classifier-free guidance |
| `src/model.py` | U-Net 4.60M parameter dengan residual block, timestep embedding sinusoidal, dan class embedding |
| `src/train.py` | Training loop (loss MSE pada noise, Adam, EMA, checkpoint + resume otomatis) |
| `src/sample.py` | Membuat gambar dari checkpoint (PNG + tensor `.pt`) |
| `src/visualize.py` | Grid dataset, visualisasi forward process, kurva loss, grid sampel per kelas |
| `notebooks/train.ipynb` | Training dan sampling di Google Colab |
| `ROADMAP.md`, `JOBDESK.md`, `EXPERIMENTS.md` | Urutan sesi, pembagian tugas, rencana eksperimen |

## Cara Menjalankan

### 1. Lingkungan lokal (laptop)

Dipakai untuk mengembangkan dan menguji kode. Diuji dengan Python 3.12, torch 2.15, torchvision 0.29, matplotlib 3.11, numpy 2.5 di macOS (GPU Apple/MPS); CPU dan CUDA juga didukung.

```bash
git clone https://github.com/ratrifa/mini-ddpm-cifar10.git
cd mini-ddpm-cifar10
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib numpy
```

Uji tiap modul (semua perintah dijalankan dari root repo):

```bash
python src/data.py        # unduh CIFAR-10 ke ./data (± 170 MB, sekali saja), cek shape dan rentang
python src/diffusion.py   # cek forward process di CPU/MPS/CUDA
python src/model.py       # cek shape output U-Net dan jumlah parameter
python src/visualize.py   # simpan outputs/dataset_samples.png dan outputs/forward_process.png
```

Server unduhan CIFAR-10 bisa sangat lambat (pernah ± 150 kB/s di laptop dan 93 kB/s di Colab). Bila sudah punya folder `cifar-10-batches-py/`, letakkan di `./data/` agar tidak diunduh ulang.

### 2. Training di Google Colab

1. Buka `notebooks/train.ipynb` di Colab (File → Open notebook → GitHub → `ratrifa/mini-ddpm-cifar10`).
2. Runtime → Change runtime type → **GPU (T4)**.
3. Jalankan cell setup: cek GPU, mount Google Drive, clone repo.
4. Jalankan cell **uji coba singkat** (2 epoch × 200 langkah, ± 2 menit di T4). Cek `last.pt`, `loss_log.csv`, `loss_steps.csv` muncul di `MyDrive/mini-ddpm/trial`.
5. Jalankan cell **training penuh** (100 epoch, ± 3 jam di T4) ke `MyDrive/mini-ddpm/run1`.

Agar dataset tidak diunduh ulang di setiap runtime (± 30 menit), simpan sekali di Drive lalu tambahkan `--data-root /content/drive/MyDrive/mini-ddpm/data`. Langkahnya ada di [EXPERIMENTS.md](EXPERIMENTS.md#0-persiapan-dataset-di-google-drive-sekali-saja).

Perintah training bisa juga dijalankan langsung (lokal maupun Colab):

```bash
python src/train.py --epochs 100 --data-root ./data --ckpt-dir ./checkpoints/run1
```

Argumen penting: `--epochs`, `--batch-size` (128), `--lr` (2e-4), `--timesteps` (1000), `--base-ch` (64), `--ema-decay` (0.999), `--p-uncond` (0.1), `--steps-per-epoch` (batasi langkah untuk uji coba).

Isi `--ckpt-dir`:
- `last.pt`: checkpoint terakhir (bobot model, bobot EMA, optimizer, epoch, config). Bila Colab terputus, jalankan ulang perintah yang sama dengan `--ckpt-dir` yang sama; training otomatis lanjut dari epoch terakhir.
- `loss_log.csv`: rata-rata loss per epoch (`epoch,loss,seconds`).
- `loss_steps.csv`: loss tiap langkah (`step,epoch,loss`).

Pakai `--ckpt-dir` berbeda untuk setiap eksperimen; resume ditolak bila config model berbeda.

### 3. Sampling

```bash
python src/sample.py --ckpt ./checkpoints/run1/last.pt --out outputs/samples.png
python src/sample.py --ckpt ./checkpoints/run1/last.pt --out outputs/samples.png --guidance-scales 1 3 5
```

- Memakai bobot EMA, membuat `--n-per-class` (8) gambar untuk tiap kelas, 1000 langkah reverse.
- Untuk tiap guidance scale disimpan `<nama>.png` (grid, satu baris per kelas) dan `<nama>.pt` (`{"images", "labels", "guidance_scale", "epoch"}`). Dengan lebih dari satu scale, nama file diberi akhiran `_g<scale>` (misalnya `samples_g3.png`).
- Guidance scale 0 = tanpa kelas, 1 = conditional biasa, > 1 ciri kelas makin tegas.

Di Colab, cell terakhir `notebooks/train.ipynb` menjalankan sampling dari `MyDrive/mini-ddpm/run1/last.pt`.

### 4. Grafik untuk laporan

```python
import csv, sys, torch
sys.path.insert(0, "src")
from visualize import plot_loss_curve, show_sample_grid

losses = [float(r["loss"]) for r in csv.DictReader(open("checkpoints/run1/loss_steps.csv"))]
plot_loss_curve(losses, out_path="outputs/loss_curve.png")

d = torch.load("outputs/samples_g3.pt")
show_sample_grid(d["images"], d["labels"], out_path="outputs/samples_per_class.png")
```
