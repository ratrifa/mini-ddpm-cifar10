# Pembagian Tugas — Kelompok 14

Panduan proyek mendefinisikan 4 peran untuk 4 anggota. Kelompok 14 beranggotakan 2 orang, sehingga setiap anggota memegang 2 peran.

| Anggota | Peran (sesuai panduan) |
|---|---|
| **Satria Febry Andanu** (Ketua) | 1. Prompt Engineer & LLM Navigator · 3. Training Loop & Hyperparameter Tuning Engineer |
| **Gifan Apriadi** | 2. Data Pipeline & Preprocessing Engineer · 4. Lead Documentarian & Error Handler |

## Satria Febry Andanu — Ketua

**Peran 1: Prompt Engineer & LLM Navigator**
- Menyusun system prompt / konteks awal untuk LLM (arsitektur, batasan "from scratch", gaya kode).
- Merancang arsitektur: U-Net kecil dengan residual block, timestep embedding (sinusoidal), dan class embedding untuk conditioning.
- Menjaga konsistensi antarmodul (shape tensor, nama fungsi, konvensi rentang [-1, 1]).

**Peran 3: Training Loop & Hyperparameter Tuning**
- Training loop: loss MSE pada noise, optimizer, checkpoint ke Google Drive, resume training.
- Eksperimen tuning: learning rate, batch size, jumlah timestep T, guidance scale; mencatat hasil tiap eksperimen.
- Mencatat metrik (loss per epoch) untuk grafik.

**File utama:** `src/model.py`, `src/diffusion.py`, `src/train.py`, `notebooks/train.ipynb`

## Gifan Apriadi

**Peran 2: Data Pipeline & Preprocessing**
- Memuat CIFAR-10 via `torchvision`, normalisasi ke [-1, 1], augmentasi (random horizontal flip), DataLoader.
- Memastikan shape input sesuai arsitektur: `(B, 3, 32, 32)` + label `(B,)`.
- Visualisasi dataset (contoh gambar per kelas) dan visualisasi forward process (gambar makin noisy per timestep).

**Peran 4: Lead Documentarian & Error Handler**
- Mengompilasi Log Interaksi LLM di `logs/` (satu file per sesi, pakai `logs/TEMPLATE.md`), termasuk sesi milik Satria.
- Menangkap pesan eror dari Colab dan merumuskannya menjadi prompt perbaikan ke LLM; mendokumentasikan analisis solusinya.
- Membuat skrip visualisasi hasil: kurva loss dan grid sampel per kelas.
- Menyusun laporan akhir dan slide presentasi.

**File utama:** `src/data.py`, `src/visualize.py`, `logs/`, laporan akhir

## Dikerjakan Bersama
- **Code review silang:** setiap modul dikerjakan di branch sendiri lalu digabung lewat Pull Request yang di-review anggota lain.
- **Sampling & evaluasi akhir:** memilih checkpoint terbaik, membuat grid sampel 10 kelas.
- **Presentasi:** kedua anggota wajib memahami seluruh modul, bukan hanya bagiannya sendiri.

## Urutan Pengerjaan

| # | Modul | Penanggung jawab | Bergantung pada |
|---|---|---|---|
| 1 | Data pipeline (`data.py`) + visualisasi dataset | Gifan | — |
| 2 | Noise schedule + forward process (`diffusion.py`) | Satria | 1 |
| 3 | Visualisasi forward process | Gifan | 2 |
| 4 | U-Net + timestep & class embedding (`model.py`) | Satria | — |
| 5 | Training loop + checkpoint (`train.py`, notebook) | Satria | 1, 2, 4 |
| 6 | Sampling (reverse process) | Satria | 2, 4 |
| 7 | Kurva loss + grid sampel (`visualize.py`) | Gifan | 5, 6 |
| 8 | Tuning hyperparameter | Satria | 5 |
| 9 | Kompilasi log + laporan + slide | Gifan (berjalan sejak #1) | semua |
