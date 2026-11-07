# Kerangka Slide Presentasi — Kelompok 14

Mini DDPM class-conditional pada CIFAR-10. Target 12 slide, sekitar 10–12 menit. Kolom "Pembawa" hanya usulan.

| # | Judul | Isi pokok | Visual | Pembawa |
|---|---|---|---|---|
| 1 | Judul | Class-conditional Mini DDPM pada CIFAR-10 · Kelompok 14 · nama + NIM · LLM: Claude Opus 5.5 | — | Satria |
| 2 | Latar belakang & tujuan | Model generatif P(x) vs P(x\|y); tujuan: membangun DDPM dari nol dengan vibe coding; alasan memilih CIFAR-10 berwarna | — | Satria |
| 3 | Cara kerja vibe coding kami | Pembagian peran, alur sesi (prompt → tes → commit → merge), satu worktree per anggota, log tiap sesi | Diagram alur / tabel ROADMAP | Gifan |
| 4 | Dataset & data pipeline | CIFAR-10: 50k train / 10k test, 10 kelas; normalisasi ke [-1, 1]; flip horizontal hanya untuk train; shape `(B, 3, 32, 32)` | `outputs/dataset_samples.png` | Gifan |
| 5 | Forward process | Schedule linear β 1e-4 → 0.02, T = 1000; rumus x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε | `outputs/forward_process.png` | Gifan |
| 6 | U-Net | Encoder–decoder 32→16→8, ResBlock + skip connection, embedding waktu (sinusoidal) + kelas, ±4,6 juta parameter | Diagram U-Net | Satria |
| 7 | Training | Loss MSE pada noise, Adam 2e-4, 100 epoch di Colab, EMA, label dropout 10% untuk CFG, checkpoint + resume di Drive | Tabel hyperparameter | Satria |
| 8 | Sampling & classifier-free guidance | 1000 langkah dari noise murni; ε = ε_uncond + w(ε_cond − ε_uncond) | Strip gambar noise → hasil _(opsional)_ | Satria |
| 9 | Hasil: kurva loss | Loss per langkah + rata-rata bergerak, sumbu log; nilai awal/akhir | `outputs/loss_curve.png` _(belum ada)_ | Gifan |
| 10 | Hasil: sampel per kelas | Grid 10 kelas berlabel; kelas yang bagus dan yang sulit | grid sampel _(belum ada)_ | Gifan / Satria |
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
