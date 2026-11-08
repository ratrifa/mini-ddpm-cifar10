# LOG INTERAKSI LLM - KELOMPOK 14

- **Tanggal Sesi:** 07/11/2026
- **Pelaksana:** Satria Febry Andanu (Prompt Engineer & LLM Navigator · Training Loop & Hyperparameter Tuning)
- **LLM:** Claude Opus 5.5 (Claude Code)
- **Target Modul:** Noise schedule + forward process (`src/diffusion.py`)
- **Tautan Sesi Chat / Ekspor:** _(lampirkan file hasil `/export`)_
- **Commit terkait:** `f38385d`, `a5c79b8` (merge `ef50bb6`)

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)

#### Prompt 1 — Membuka sesi
- **Prompt Kelompok:** "Gua Satria, ketua Kelompok 14. Baca dulu memory proyek ini, terutama session-handoff.md, midterm-group-context.md, sama no-pull-requests.md, terus project-1/ROADMAP.md dan JOBDESK.md. Di sesi ini kita ngerjain Sesi 2: src/diffusion.py. Commit atas nama gua, pakai alur branch → merge lokal → push."
- **Alasan Khusus:** _(dirumuskan bersama LLM, disetujui Satria)_ LLM diminta membaca konteks proyek (pembagian tugas, roadmap, aturan commit) terlebih dahulu agar kode baru konsisten dengan modul Sesi 1 (`data.py`: rentang [-1, 1], shape `(B, 3, 32, 32)`) dan alur kerja kelompok.
- **Output Model:** LLM merangkum status proyek, lalu sebelum menulis kode mengajukan dua pertanyaan desain: jenis noise schedule dan jumlah timestep T.

#### Prompt 2 — Keputusan desain
- **Pilihan Kelompok:**
  - Noise schedule: **Linear** (β dari 1e-4 ke 0.02), dibanding opsi cosine (Improved DDPM) atau keduanya.
  - Jumlah timestep: **T = 1000**, dibanding opsi T = 500 atau T = 300.
- **Alasan Khusus:** _(dirumuskan bersama LLM, disetujui Satria)_ Linear dengan T = 1000 adalah pengaturan asli paper DDPM (Ho et al., 2020) untuk CIFAR-10, sehingga mudah dijelaskan dan menjadi baseline. T tetap berupa argumen, jadi bisa diturunkan saat tuning di Sesi 8 bila sampling di Colab terlalu lambat.
- **Output Model:** Kelas `Diffusion(timesteps=1000, beta_start=1e-4, beta_end=0.02)` yang menyimpan β, α = 1 − β, ᾱ (cumprod α), √ᾱ, dan √(1 − ᾱ), serta fungsi forward process:
  ```python
  def q_sample(self, x0, t, noise=None):
      # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
      if noise is None:
          noise = torch.randn_like(x0)
      return (self._extract(self.sqrt_alpha_bars, t, x0) * x0
              + self._extract(self.sqrt_one_minus_alpha_bars, t, x0) * noise)
  ```
  Keputusan antarmuka yang dicatat di docstring (untuk Sesi 3, 5, 6):
  - Indeks timestep `t` berjalan dari 0 sampai T−1; indeks 0 setara t=1 di paper. Gambar bersih adalah `x0` itu sendiri.
  - `q_sample` hanya mengembalikan `x_t`. Training (Sesi 5) membuat `noise` sendiri lalu mengopernya, karena noise itu target yang diprediksi model.
  - Contoh untuk Sesi 3 (satu gambar di beberapa timestep dengan noise yang sama) ditulis di docstring.
- **Hasil Uji** (`python src/diffusion.py`):
  ```
  T=1000, beta: 1.0e-04 -> 2.0e-02
  alpha_bar[0]=0.99990, alpha_bar[T-1]=4.04e-05

  [cpu]
    shape x_t            : (64, 3, 32, 32) (sama dengan x0)
    selisih vs rumus     : 8.34e-07
    t=0   maks |x_t - x0|: 0.0459
    t=T-1 mean / std     : -0.0030 / 1.0008  (target ~0 / ~1)

  [mps]
    shape x_t            : (64, 3, 32, 32) (sama dengan x0)
    selisih vs rumus     : 1.61e-06
    t=0   maks |x_t - x0|: 0.0421
    t=T-1 mean / std     : -0.0002 / 0.9976  (target ~0 / ~1)
  ```
  Uji tambahan dengan satu gambar test CIFAR-10 asli memakai contoh di docstring: korelasi x_t dengan x0 turun dari 1.000 (t=0) → 0.337 (t=250) → 0.085 (t=500) → 0.001 (t=750) → −0.018 (t=999), sedangkan std naik ke ~1. Artinya pada t = T−1 gambar sudah praktis menjadi noise N(0, 1).

### 2. TAHAP PENANGANAN EROR / DEBUGGING

Tidak ada eror saat kode dijalankan. Hal yang diantisipasi sejak awal:

#### Antisipasi 1 — Perbedaan device (laptop MPS, Colab CUDA)
- **Masalah potensial:** Tabel schedule disimpan di CPU, sedangkan `t` dan gambar bisa berada di MPS/CUDA. Mengindeks tensor CPU dengan indeks di device lain akan eror.
- **Solusi:** Fungsi `_extract` memindahkan tabel dan `t` ke device gambar sebelum mengambil nilai, lalu mengubah shape ke `(B, 1, 1, 1)` agar bisa dikalikan dengan gambar `(B, 3, H, W)`. Self-test dijalankan di CPU dan MPS.

#### Antisipasi 2 — MPS tidak mendukung float64
- **Masalah potensial:** ᾱ adalah hasil perkalian 1000 bilangan, sehingga lebih presisi bila dihitung di float64, tetapi MPS tidak bisa memakai float64.
- **Solusi:** Schedule dihitung di float64 lalu disimpan sebagai float32.
- **Pelajaran:** _(dirumuskan bersama LLM, disetujui Satria)_ Perbedaan device memengaruhi kode, bukan hanya kecepatan: fitur yang jalan di CPU belum tentu jalan di MPS atau CUDA. Karena itu setiap modul diuji di CPU dan MPS sejak awal, sebelum dipakai di Colab.

## Ringkasan Sesi
| Kriteria (ROADMAP Sesi 2) | Status |
|---|---|
| Noise schedule linear β, α, ᾱ (T = 1000) | ✅ |
| Fungsi pembuat gambar noisy pada timestep t (`q_sample`) berjalan | ✅ |
| Shape output sama dengan input, berjalan di CPU dan MPS | ✅ |
