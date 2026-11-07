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
