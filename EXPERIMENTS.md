# Rencana Eksperimen Sesi 8 — Tuning Hyperparameter

Target: selesai dalam **satu malam di Colab gratis (GPU T4)**. Semua angka waktu di bawah berasal dari uji coba Sesi 5 di Colab (± 3,6 langkah/detik, satu epoch penuh 390 langkah ± 105 detik), kecuali yang ditandai *perkiraan*.

## 0. Persiapan: dataset di Google Drive (sekali saja)

Uji coba Sesi 5 menunjukkan unduhan CIFAR-10 di Colab memakan **30 menit** (93 kB/s), dan `/content/data` hilang setiap runtime baru. Solusinya: simpan dataset di Drive.

`torchvision` hanya mengunduh bila file di `<data-root>/cifar-10-batches-py/` tidak ada atau checksum MD5-nya tidak cocok. Jadi cukup sediakan folder itu di Drive:

1. Upload folder `data/cifar-10-batches-py/` dari laptop (178 MB, sudah ada sejak Sesi 1) ke Google Drive sehingga menjadi `MyDrive/mini-ddpm/data/cifar-10-batches-py/`.
   File `cifar-10-python.tar.gz` (163 MB) **tidak perlu** di-upload.
2. Di setiap perintah training, pakai `--data-root /content/drive/MyDrive/mini-ddpm/data`.

Bila upload dari laptop tidak memungkinkan, jalankan sekali training apa pun dengan `--data-root` di atas; unduhan 30 menit terjadi satu kali lalu tersimpan di Drive.

Tidak perlu mengubah kode: `--data-root` sudah ada di `train.py`.

## 1. Baseline (sedang berjalan)

| | Nilai |
|---|---|
| Folder | `MyDrive/mini-ddpm/run1` |
| Hyperparameter | 100 epoch, batch 128, lr 2e-4, T = 1000 linear, EMA 0.999, p_uncond 0.1, model 4.60M |
| Waktu | ± 3 jam |

Semua eksperimen dibandingkan dengan run ini.

## 2. Daftar eksperimen (urut prioritas)

### E1 — Guidance scale 0, 1, 3, 5 (sampling saja, tanpa training ulang)
- **Alasan:** Guidance scale hanya dipakai saat sampling, jadi bisa dibandingkan dari checkpoint baseline tanpa biaya training. Ini parameter yang paling terlihat pengaruhnya di gambar: makin besar, ciri kelas makin tegas tetapi variasi turun dan warna bisa berlebihan. Scale 0 = sampling tanpa kelas (unconditional), berguna untuk melihat apakah class conditioning benar-benar bekerja.
- **Perintah:**
  ```
  !python src/sample.py --ckpt /content/drive/MyDrive/mini-ddpm/run1/last.pt \
      --out /content/drive/MyDrive/mini-ddpm/run1/samples.png --guidance-scales 0 1 3 5
  ```
  Menghasilkan `samples_g0.png/.pt` … `samples_g5.png/.pt`; noise awal sama untuk semua scale (seed sama), jadi perbedaannya murni dari guidance.
- **Waktu:** *perkiraan* 2–4 menit per scale (1000 langkah × batch 160), total ≤ 15 menit.
- **Yang dicatat:** grid per scale (pakai `visualize.show_sample_grid`), pengamatan kualitatif per kelas.

### E2 — Learning rate 4e-4 (training pendek, 30 epoch)
- **Alasan:** Kuota Colab gratis terbatas, jadi pertanyaan yang relevan adalah apakah lr lebih besar membuat model konvergen lebih cepat dalam jumlah epoch yang sama, atau justru tidak stabil. Model kecil (4.60M) biasanya toleran terhadap lr lebih besar dibanding model paper (35.7M, lr 2e-4).
- **Perintah:**
  ```
  !python src/train.py --epochs 30 --lr 4e-4 \
      --data-root /content/drive/MyDrive/mini-ddpm/data --ckpt-dir /content/drive/MyDrive/mini-ddpm/e2_lr4e-4
  ```
- **Waktu:** 30 × 105 detik ± 55 menit, ditambah sampling ± 3 menit.
- **Perbandingan:** loss per epoch E2 vs baseline **pada epoch yang sama** (kolom `loss` di `loss_log.csv` epoch 1–30), plus grid sampel guidance 3. Catatan: grid E2 (30 epoch) tidak sebanding langsung dengan grid baseline (100 epoch), karena checkpoint baseline epoch 30 tidak disimpan; perbandingan utama memakai loss.

### E3 (opsional, bila kuota masih ada) — T = 500
- **Alasan:** Sampling 1000 langkah adalah bagian paling lambat saat membuat gambar. T = 500 membuat sampling 2× lebih cepat; eksperimen ini mengukur berapa kualitas yang hilang.
- **Catatan penting:** dengan β tetap 1e-4 → 0.02, T = 500 menyisakan sinyal di langkah terakhir: ᾱ_T = 6.35e-3 (√ᾱ_T ≈ 0.080, ± 8% gambar asli), dibanding ᾱ_T = 4.04e-5 (√ᾱ_T ≈ 0.006) pada T = 1000 (dihitung dengan `Diffusion`). Sampling selalu mulai dari noise murni, jadi ada ketidakcocokan antara training dan sampling. Perbaikan yang benar adalah menaikkan `beta_end` (± 0.04) agar total noise setara, tetapi `train.py` belum punya argumen `--beta-end` dan kode inti dibekukan selama run berjalan. Karena itu E3 hanya dijalankan bila kuota sisa, dan hasilnya dibaca dengan catatan ini — atau ditunda sampai `--beta-end` ditambahkan setelah run selesai.
- **Perintah:**
  ```
  !python src/train.py --epochs 30 --timesteps 500 \
      --data-root /content/drive/MyDrive/mini-ddpm/data --ckpt-dir /content/drive/MyDrive/mini-ddpm/e3_T500
  ```
- **Waktu:** ± 55 menit training + sampling ± 2 menit.
- **Perbandingan:** dengan E2 tidak bisa (lr berbeda); bandingkan dengan baseline pada epoch 30 (loss) dan secara visual. Loss antar-T tidak sepenuhnya sebanding karena distribusi timestep berbeda, jadi pengamatan visual lebih penting.

### Tidak dilakukan malam ini (dan alasannya)
- **Model lebih besar (`--base-ch 96/128`):** ± 2–3× lebih lambat per epoch; tidak muat bersama eksperimen lain dalam satu malam.
- **Cosine schedule / self-attention:** butuh perubahan `diffusion.py` / `model.py`, sedangkan kode inti dibekukan selama run berjalan.
- **FID:** butuh ribuan sampel (puluhan menit sampling) dan Inception network; evaluasi malam ini kualitatif (grid) + loss.

## 3. Jadwal satu malam

| Urutan | Eksperimen | Waktu | Kumulatif |
|---|---|---|---|
| — | Baseline `run1` (berjalan) | ± 3 jam | ± 3 jam |
| 1 | E1 guidance sweep | ≤ 15 menit | ± 3,25 jam |
| 2 | E2 lr 4e-4 | ± 1 jam | ± 4,25 jam |
| 3 | E3 T = 500 (opsional) | ± 1 jam | ± 5,25 jam |

Colab gratis tidak menjamin durasi sesi dan bisa memutus runtime. Bila terputus: jalankan ulang cell setup lalu **perintah yang sama dengan `--ckpt-dir` yang sama**; `train.py` otomatis melanjutkan dari epoch terakhir di `last.pt`. Gunakan folder berbeda untuk tiap eksperimen agar checkpoint tidak tertimpa (resume ditolak bila config berbeda).

## 4. Tabel hasil (diisi setelah eksperimen)

| ID | Perubahan dari baseline | Epoch | Loss epoch terakhir | Loss epoch 30 | Waktu | Catatan visual |
|---|---|---|---|---|---|---|
| run1 | — | 100 | | | | |
| E1 | guidance 0 / 1 / 3 / 5 | 100 | — | — | 10 menit 10 detik, 4 skala × 80 gambar, di laptop (MPS) karena kuota GPU Colab habis | **0:** objek terbentuk tapi kelas acak · **1:** sebagian besar sesuai kelas, ada yang meleset (objek merah di baris kuda, kucing kabur) · **3:** semua baris jelas sesuai kelas; kucing/anjing paling lemah · **5:** komposisi sama dengan 3 (seed sama), warna lebih jenuh, variasi turun (anjing hampir semua wajah putih). Default 3 dipertahankan. |
| E2 | lr 4e-4 | 30 | | | | |
| E3 | T = 500 | 30 | | | | |
