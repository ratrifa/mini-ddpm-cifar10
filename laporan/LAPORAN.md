# Laporan Proyek 1 — Vibe Coding Generative AI from Scratch

**Class-conditional Mini DDPM pada CIFAR-10 (32×32)**

CAK4KBB3 Generative AI 2026-1 · Kelompok 14
- Satria Febry Andanu (103012300372), Ketua
- Gifan Apriadi (103012300385)

Repositori: https://github.com/ratrifa/mini-ddpm-cifar10
LLM: Claude Opus 5.5 (Claude Code)

> Status: **kerangka**. Bagian bertanda _(isi)_ dilengkapi kelompok; bagian Hasil diisi setelah training penuh selesai.

---

## 1. Pendahuluan

### 1.1 Latar belakang
_(isi: model generatif belajar distribusi data P(x), dan versi conditional belajar P(x | y). Hubungkan dengan materi GEN01.)_

### 1.2 Tujuan
- Membangun Denoising Diffusion Probabilistic Model (DDPM; Ho et al., 2020) skala kecil dari nol dengan PyTorch melalui kolaborasi dengan LLM (vibe coding), tanpa menyalin kode tutorial.
- Model dapat membuat gambar 32×32 untuk kelas CIFAR-10 yang dipilih (class-conditional).
- Mendokumentasikan seluruh interaksi dengan LLM: inisiasi → penanganan eror → optimasi.

### 1.3 Alasan memilih DDPM conditional pada CIFAR-10
_(isi: alasan kelompok. Catatan: Kelompok 4 sudah memakai Fashion-MNIST, sehingga kelompok ini memakai gambar berwarna.)_

### 1.4 Hubungan dengan materi kuliah
_(isi, opsional)_ Jembatan konsep: ELBO pada VAE (GEN06) → bound variasional DDPM; sampling dari prior N(0, I) seperti pada VAE; U-Net sebagai encoder–decoder CNN (02A) dengan residual block seperti ResNet (GEN02).

---

## 2. Dataset dan Data Pipeline

### 2.1 CIFAR-10
- 60.000 gambar RGB 32×32 dalam 10 kelas: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- 50.000 gambar train (dipakai untuk training) dan 10.000 gambar test (dipakai untuk visualisasi).
- Diunduh otomatis lewat `torchvision.datasets.CIFAR10`.

![Contoh dataset per kelas](../outputs/dataset_samples.png)

*Gambar 1. Delapan gambar test pertama dari tiap kelas.*

### 2.2 Preprocessing (`src/data.py`)
| Langkah | Detail | Alasan |
|---|---|---|
| `ToTensor()` | pixel [0, 255] → [0, 1], shape `(3, 32, 32)` | |
| `Normalize(0.5, 0.5)` | [0, 1] → [-1, 1] | Noise yang ditambahkan berdistribusi N(0, 1), sehingga data perlu berpusat di nol dengan skala sebanding |
| `RandomHorizontalFlip()` | p = 0.5, **hanya data train** | Versi cermin tetap realistis. Flip vertikal tidak dipakai karena model generatif akan ikut belajar menghasilkan gambar terbalik |
| `DataLoader` | batch 128; shuffle + `drop_last` untuk train | Ukuran batch tetap selama training |

Keluaran: gambar `(B, 3, 32, 32)` di [-1, 1] dan label `(B,)`. _(isi: tambahan penjelasan bila perlu)_

---

## 3. Arsitektur dan Metode

### 3.1 Forward process (`src/diffusion.py`)
- T = 1000 langkah dengan noise schedule linear β dari 1e-4 sampai 0.02.
- ᾱ_t = ∏(1 − β_s), sehingga x_t dapat dibuat langsung dari x_0 dalam satu langkah:

  x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε,  ε ~ N(0, I)

- Implementasi: `Diffusion.q_sample(x0, t, noise)`. Indeks t berjalan dari 0 sampai T−1.

![Forward process](../outputs/forward_process.png)

*Gambar 2. Forward process pada satu gambar per kelas dengan noise tetap. √ᾱ_t (sisa porsi sinyal asli) turun dari 1.00 ke 0.01. Objek masih jelas di t=100, samar di t=300, dan hilang di t ≥ 700.*

_(isi: interpretasi kelompok atas gambar ini)_

### 3.2 U-Net (`src/model.py`)
- Fungsi: memprediksi noise ε dari input (x_t, t, y).
- Encoder–decoder dengan skip connection. Tiga level resolusi 32 → 16 → 8 dengan channel 64 / 128 / 128, 2 ResBlock per level, dan 2 ResBlock di tengah.
- ResBlock: GroupNorm → SiLU → Conv, ditambah embedding, lalu dropout 0.1 dan koneksi residual.
- Timestep embedding: sinusoidal (seperti positional encoding Transformer) lalu MLP ke dimensi 256.
- Class embedding: `nn.Embedding(11, 256)`, berisi 10 kelas ditambah 1 token "tanpa kelas" untuk classifier-free guidance. Embedding waktu dan kelas dijumlahkan, lalu ditambahkan per channel di tiap ResBlock.
- Jumlah parameter: ±4,6 juta.

_(isi: diagram U-Net bila ada)_

### 3.3 Training (`src/train.py`, `notebooks/train.ipynb`)
| Komponen | Nilai |
|---|---|
| Loss | MSE antara noise asli ε dan prediksi model |
| Optimizer | Adam, lr 2e-4, gradient clipping 1.0 |
| Batch / epoch | 128 / 100 |
| Classifier-free guidance | label diganti token "tanpa kelas" dengan peluang 0,1 |
| EMA bobot | decay 0,999 (bobot EMA dipakai untuk sampling) |
| Checkpoint | `last.pt` di Google Drive, otomatis resume |
| Log | `loss_log.csv` (rata-rata per epoch), `loss_steps.csv` (per langkah) |
| Hardware | Google Colab _(isi: jenis GPU, durasi training)_ |

### 3.4 Sampling / reverse process (`Diffusion.sample`, `src/sample.py`)
- Mulai dari noise murni x_T ~ N(0, I), lalu 1000 langkah dari t = T−1 ke 0.
- Tiap langkah: prediksi noise dengan dan tanpa kelas, digabung dengan classifier-free guidance ε = ε_uncond + w · (ε_cond − ε_uncond), default w = 3.0.
- x̂_0 ditebak dari x_t dan ε, di-clamp ke [-1, 1], lalu x_{t−1} diambil dari posterior q(x_{t−1} | x_t, x̂_0). Langkah terakhir tanpa noise.

---

## 4. Hasil

### 4.1 Kurva loss
_(isi setelah training penuh: jalankan `python src/plot_loss.py <run1>/loss_steps.csv --out outputs/loss_curve.png`)_

<!-- ![Kurva loss](../outputs/loss_curve.png) -->

Kurva menampilkan loss per langkah (garis samar) dan rata-rata bergerak 100 langkah, dengan sumbu y logaritmik. _(isi: nilai loss awal/akhir dan analisis)_

### 4.2 Sampel per kelas
_(isi setelah sampling: grid berlabel dari `show_sample_grid`)_

<!-- ![Sampel per kelas](../outputs/samples_per_class.png) -->

### 4.3 Eksperimen dan tuning
_(isi dari Sesi 8: perbandingan guidance scale, lr, dan lain-lain)_

| Eksperimen | Perubahan | Hasil |
|---|---|---|
| _(isi)_ | | |

### 4.4 Analisis
_(isi: kualitas sampel per kelas, kelas yang paling mudah/sulit, keterbatasan)_

---

## 5. Pembagian Tugas
| Anggota | Peran | Modul / pekerjaan |
|---|---|---|
| Satria Febry Andanu (Ketua) | Prompt Engineer & LLM Navigator · Training Loop & Hyperparameter Tuning | `src/diffusion.py`, `src/model.py`, `src/train.py`, `src/sample.py`, `notebooks/train.ipynb`, tuning |
| Gifan Apriadi | Data Pipeline & Preprocessing · Lead Documentarian & Error Handler | `src/data.py`, `src/visualize.py`, `src/plot_loss.py`, kompilasi log, laporan, slide |

Alur kerja: tiap modul dikerjakan di branch sendiri (git worktree terpisah per anggota), dicek, lalu di-merge ke `main` dengan `--no-ff`. Rincian ada di [JOBDESK.md](../JOBDESK.md) dan [ROADMAP.md](../ROADMAP.md). _(isi: ringkasan jumlah commit per anggota)_

---

## 6. Log Interaksi LLM
| Sesi | Modul | Pelaksana | Log | Ekspor chat |
|---|---|---|---|---|
| 1 | Data pipeline + grid dataset | Gifan | [2026-11-07-data-pipeline.md](../logs/2026-11-07-data-pipeline.md) | _(isi)_ |
| 2 | Noise schedule + forward process | Satria | [2026-11-07-diffusion.md](../logs/2026-11-07-diffusion.md) | _(isi)_ |
| 3, 7 | Visualisasi forward process, kurva loss, grid sampel | Gifan | [2026-11-07-visualize.md](../logs/2026-11-07-visualize.md) | _(isi)_ |
| 4 | U-Net + embedding | Satria | [2026-11-07-model.md](../logs/2026-11-07-model.md) | _(isi)_ |
| 5 | Training loop + notebook | Satria | [2026-11-07-train.md](../logs/2026-11-07-train.md) | _(isi)_ |
| 6 | Sampling | Satria | [2026-11-07-sampling.md](../logs/2026-11-07-sampling.md) | _(isi)_ |
| 8 | Training penuh + tuning | Satria | _(belum)_ | _(isi)_ |

_(isi: ringkasan pola prompt, eror penting, dan pelajaran yang diambil)_

---

## 7. Kesimpulan
_(isi)_

## Referensi
1. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
2. J. Ho, T. Salimans. *Classifier-Free Diffusion Guidance*. NeurIPS 2021 Workshop.
3. A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10), 2009.
4. Materi kuliah CAK4KBB3: GEN01 Introduction, 02A Encoder-Decoder, GEN02 Review RNN/CNN, GEN06 VAE.
