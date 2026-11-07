# Roadmap Sesi Pengerjaan

Acuan kerja Proyek 1 (Mini DDPM CIFAR-10). Satria dan Gifan bekerja **paralel** di sesi Claude masing-masing. Nomor sesi hanya urutan yang disarankan; yang wajib dipatuhi adalah kolom **Butuh**. Perbarui kolom **Status** setiap sesi selesai.

| Sesi | Siapa | Modul | Butuh | Kriteria selesai | Tempat | Status |
|---|---|---|---|---|---|---|
| 1 | Gifan | `src/data.py` + visualisasi contoh dataset | — | Shape batch `(B, 3, 32, 32)`, nilai pixel di rentang [-1, 1], grid 10 kelas tampil | Laptop | ✅ |
| 2 | Satria | `src/diffusion.py`: noise schedule + forward process | 1 | Fungsi pembuat gambar noisy pada timestep t berjalan | Laptop | ⬜ |
| 3 | Gifan | Visualisasi forward process | 2 (atau kesepakatan antarmuka) | Satu gambar ditampilkan makin noisy dari t=0 sampai T | Laptop | ⬜ |
| 4 | Satria | `src/model.py`: U-Net + timestep & class embedding | — | Shape output sama dengan input | Laptop | ⬜ |
| 5 | Satria | `src/train.py` + `notebooks/train.ipynb`, uji coba training singkat | 1, 2, 4 | Loss turun, checkpoint tersimpan ke Google Drive | Colab | ⬜ |
| 6 | Satria | Sampling (reverse process) | 2, 4 | Gambar hasil generate keluar | Colab | ⬜ |
| 7 | Gifan | `src/visualize.py`: kurva loss + grid sampel per kelas | Kode: — (uji dengan data palsu) · Hasil akhir: 5, 6 | Grafik + grid siap dipakai di laporan | Laptop/Colab | ⬜ |
| 8 | Satria | Training penuh + tuning hyperparameter | 5, 6 | Perbandingan hasil eksperimen tercatat | Colab | ⬜ |
| 9 | Gifan | Kompilasi log + laporan + slide | Dicicil sejak awal; final setelah semua | Siap dikumpulkan | Laptop | ⬜ |

Status: ⬜ belum · 🟨 berjalan · ✅ selesai

## Urutan yang Disarankan (Paralel)
- **Satria:** Sesi 2 → Sesi 4 → Sesi 5 → Sesi 6 → Sesi 8
- **Gifan:** lengkapi log Sesi 1 → kode Sesi 7 (data palsu) → Sesi 3 → hasil akhir Sesi 7 → Sesi 9

## Kesepakatan Antarmuka
Modul yang dipakai anggota lain harus disepakati nama fungsi dan parameternya sebelum dikerjakan paralel. Jika berubah, kabari anggota lain dan perbarui tabel ini.

| Fungsi | Modul | Dipakai oleh | Status |
|---|---|---|---|
| `get_dataloader(root, batch_size, train, num_workers)` | `data.py` | Sesi 2, 5 | ✅ sudah ada |
| `to_display(x)` | `visualize.py` | Sesi 3, 6, 7 | ✅ sudah ada |
| `q_sample(x0, t, noise=None) -> x_t` | `diffusion.py` | Sesi 3, 5 | ⬜ usulan, dikonfirmasi di Sesi 2 |

## Aturan Setiap Sesi
1. Di awal sesi, sebutkan siapa yang mengerjakan. Commit Gifan memakai `--author="Gifan Apriadi <141617477+gifann@users.noreply.github.com>"`.
2. Satria dan Gifan memakai folder kerja (git worktree) terpisah agar perpindahan branch di satu sesi tidak mengganggu sesi lain.
3. Pemilik modul menulis prompt sendiri ke LLM.
4. Pesan eror dari terminal/Colab ditempel utuh ke LLM.
5. Di akhir sesi: jalankan `/export`, lalu isi log di `logs/` berdasarkan `logs/TEMPLATE.md` (nama file: `YYYY-MM-DD-nama-modul.md`).
6. Merge branch ke `main` secara lokal (`git merge --no-ff`), perbarui kolom Status, lalu push. Sebelum merge, `git pull` dulu karena anggota lain mungkin sudah push.
