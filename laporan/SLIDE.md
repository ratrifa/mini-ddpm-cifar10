# Kerangka Slide Presentasi — Kelompok 14

Mini DDPM class-conditional pada CIFAR-10. Target 12–13 slide (10b opsional bila waktu cukup), sekitar 10–12 menit. Kolom "Pembawa" hanya usulan.

| # | Judul | Isi pokok | Visual | Pembawa |
|---|---|---|---|---|
| 1 | Judul | Class-conditional Mini DDPM pada CIFAR-10 · Kelompok 14 · nama + NIM · LLM: Claude Opus 5.5 | — | Satria |
| 2 | Latar belakang & tujuan | Model generatif P(x) vs P(x\|y); tujuan: membangun DDPM dari nol dengan vibe coding; alasan memilih CIFAR-10 berwarna | — | Satria |
| 3 | Cara kerja vibe coding kami | Pembagian peran, alur sesi (prompt → tes → commit → merge), satu worktree per anggota, log tiap sesi | Diagram alur / tabel ROADMAP | Gifan |
| 4 | Dataset & data pipeline | CIFAR-10: 50k train / 10k test, 10 kelas; normalisasi ke [-1, 1]; flip horizontal hanya untuk train; shape `(B, 3, 32, 32)` | `outputs/dataset_samples.png` | Gifan |
| 5 | Forward process | Schedule linear β 1e-4 → 0.02, T = 1000; rumus x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε | `outputs/forward_process.png` | Gifan |
| 6 | U-Net | Encoder–decoder 32→16→8, ResBlock + skip connection, embedding waktu (sinusoidal) + kelas, ±4,6 juta parameter | Diagram U-Net | Satria |
| 7 | Training | Loss MSE pada noise, Adam 2e-4, batch 128, 100 epoch (39.000 langkah) di Colab T4 ± 3,1 jam, EMA 0,999, label dropout 10% untuk CFG, checkpoint + resume di Drive | Tabel hyperparameter | Satria |
| 8 | Sampling & classifier-free guidance | 1000 langkah dari noise murni; ε = ε_uncond + w(ε_cond − ε_uncond) | Strip gambar noise → hasil _(opsional)_ | Satria |
| 9 | Hasil: kurva loss | Loss per langkah + rata-rata 100 langkah, sumbu log. Loss rata-rata per epoch 0,1106 (epoch 1) → 0,0309 (epoch 100), terendah 0,0305 di epoch 79; turun tajam di ± 1.000 langkah pertama lalu melandai | `outputs/loss_curve.png` | Gifan |
| 10 | Hasil: sampel per kelas + guidance (E1) | w = 3: semua baris sesuai kelas; pesawat, kapal, katak, kuda, truk paling jelas, kucing/anjing paling lemah. w = 0 kelas acak, w = 1 ada yang meleset, w = 5 warna lebih jenuh dan variasi turun. Sampling 4 skala 10 menit 10 detik di laptop (MPS) | `outputs/samples_g3_labeled.png` + `samples_g{0,1,5}_labeled.png` (kecil) | Gifan / Satria |
| 10b | Hasil: learning rate 4e-4 (E2) | 30 epoch, ± 59 menit. Loss lebih cepat turun sampai ± epoch 15, lalu menyatu dengan run1; epoch 30: 0,0325 vs 0,0327. Grid E2 (epoch 30) tidak setara dengan run1 (epoch 100) karena checkpoint run1 epoch 30 tidak ada. E3 (T = 500) tidak dijalankan: kuota GPU habis | `outputs/loss_compare_lr.png` + `outputs/e2_samples_g3_labeled.png` | Satria |
| 11 | Log LLM: eror & pelajaran | 2–3 contoh prompt → eror → perbaikan dari log | Cuplikan log | Gifan |
| 12 | Kesimpulan & keterbatasan | Ringkasan, keterbatasan (resolusi, waktu training), pengembangan berikutnya | — | Satria |

---

## Pertanyaan yang Mungkin Ditanya Dosen (bagian Gifan)

Jawaban sengaja tidak disertakan. Siapkan jawabanmu sendiri, lalu cek ke kode atau log terkait.

### Data pipeline (`src/data.py`)
1. Kenapa gambar dinormalisasi ke [-1, 1], bukan [0, 1]? Apa yang terjadi kalau tetap [0, 1]?
2. Apa yang dilakukan `Normalize((0.5,)*3, (0.5,)*3)` secara matematis?
3. Kenapa augmentasi flip hanya untuk data train? Kenapa tidak memakai flip vertikal atau rotasi?
4. Apa fungsi `shuffle` dan `drop_last` di DataLoader train?
5. Shape satu batch berapa, dan kenapa harus `(B, 3, 32, 32)`?
6. Label kelas dipakai di mana dalam model? Bagaimana label diubah jadi input U-Net?
7. Kenapa grid dataset memakai data test, bukan data train?
8. Berapa banyak data yang dipakai training? Apakah data test ikut dilatih?

### Visualisasi (`src/visualize.py`, `src/plot_loss.py`)
9. Apa yang dilakukan `to_display`? Kenapa perlu clamp?
10. Jelaskan gambar forward process: kenapa di t=999 gambar jadi noise murni? Apa arti angka √ᾱ di atas tiap kolom?
11. Kenapa satu noise yang sama dipakai untuk semua kolom pada satu gambar?
12. Kenapa kolom t=0 terlihat hampir sama dengan gambar asli? (petunjuk: konvensi indeks t di `diffusion.py`)
13. Kenapa gambar sudah hampir tidak terlihat sekitar t=300–500 padahal T=1000? Apa hubungannya dengan schedule linear?
14. Kenapa kurva loss memakai rata-rata bergerak? Kenapa loss per langkah sangat berisik?
15. Kenapa sumbu y kurva loss logaritmik?
16. Apakah loss yang rendah menjamin gambar hasil bagus? Kenapa?
17. Bagaimana `read_step_losses` menangani Colab yang terputus lalu training di-resume?

### Peran Lead Documentarian
18. Tunjukkan satu contoh eror nyata dari log, prompt perbaikannya, dan pelajarannya.
19. Bagaimana kalian memastikan log berisi prompt asli, bukan dibuat ulang?
20. Bagaimana pembagian kerja terlihat di Git (author commit, branch, merge)?
