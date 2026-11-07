# Roadmap Sesi Pengerjaan

Acuan urutan kerja Proyek 1 (Mini DDPM CIFAR-10). Sesi dikerjakan bergantian karena memakai satu laptop. Perbarui kolom **Status** setiap sesi selesai.

| Sesi | Siapa | Modul | Kriteria selesai | Tempat | Status |
|---|---|---|---|---|---|
| 1 | Gifan | `src/data.py` + visualisasi contoh dataset | Shape batch `(B, 3, 32, 32)`, nilai pixel di rentang [-1, 1], grid 10 kelas tampil | Laptop | ✅ |
| 2 | Satria | `src/diffusion.py`: noise schedule + forward process | Fungsi pembuat gambar noisy pada timestep t berjalan | Laptop | ✅ |
| 3 | Gifan | Visualisasi forward process | Satu gambar ditampilkan makin noisy dari t=0 sampai T | Laptop | ⬜ |
| 4 | Satria | `src/model.py`: U-Net + timestep & class embedding | Shape output sama dengan input | Laptop | ✅ |
| 5 | Satria | `src/train.py` + `notebooks/train.ipynb`, uji coba training singkat | Loss turun, checkpoint tersimpan ke Google Drive | Colab | 🟨 |
| 6 | Satria | Sampling (reverse process) | Gambar hasil generate keluar | Colab | ⬜ |
| 7 | Gifan | `src/visualize.py`: kurva loss + grid sampel per kelas | Grafik + grid siap dipakai di laporan | Laptop/Colab | ⬜ |
| 8 | Satria | Training penuh + tuning hyperparameter | Perbandingan hasil eksperimen tercatat | Colab | ⬜ |
| 9 | Gifan | Kompilasi log + laporan + slide | Siap dikumpulkan | Laptop | ⬜ |

Status: ⬜ belum · 🟨 berjalan · ✅ selesai

## Aturan Setiap Sesi
1. Di awal sesi, sebutkan siapa yang mengerjakan. Commit Gifan memakai `--author="Gifan Apriadi <141617477+gifann@users.noreply.github.com>"`.
2. Pemilik modul menulis prompt sendiri ke LLM.
3. Pesan eror dari terminal/Colab ditempel utuh ke LLM.
4. Di akhir sesi: jalankan `/export`, lalu isi log di `logs/` berdasarkan `logs/TEMPLATE.md` (nama file: `YYYY-MM-DD-nama-modul.md`).
5. Perbarui kolom Status di tabel ini, lalu commit.
