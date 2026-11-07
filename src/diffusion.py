"""Noise schedule dan forward process (q) untuk Mini DDPM (Ho et al., 2020).

Konvensi timestep: indeks t berjalan dari 0 sampai T-1. Indeks 0 setara t=1 di paper
(sudah sedikit bernoise); gambar bersih adalah x0 itu sendiri, bukan q_sample(x0, 0).
"""

import torch


class Diffusion:
    """Menyimpan noise schedule linear dan menyediakan forward process q(x_t | x_0)."""

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        # beta_t: varians noise yang ditambahkan di tiap langkah, naik linear (paper DDPM)
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        self.alphas = 1.0 - self.betas
        # alpha_bar_t = prod(alpha_1..alpha_t): sisa porsi sinyal asli setelah t langkah
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # dihitung di float64 lalu disimpan float32 (MPS tidak mendukung float64)
        self.sqrt_alpha_bars = self.alpha_bars.sqrt().float()
        self.sqrt_one_minus_alpha_bars = (1.0 - self.alpha_bars).sqrt().float()
        self.betas = self.betas.float()
        self.alphas = self.alphas.float()
        self.alpha_bars = self.alpha_bars.float()

    @staticmethod
    def _extract(values, t, x):
        """Mengambil values[t] per sampel dan mengubahnya ke shape (B, 1, 1, 1) di device x."""
        return values.to(x.device)[t.to(x.device)].view(-1, 1, 1, 1)

    def q_sample(self, x0, t, noise=None):
        """Membuat gambar noisy x_t langsung dari x0 dalam satu langkah.

        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        x0    : gambar bersih (B, 3, H, W) di rentang [-1, 1]
        t     : tensor long (B,), indeks timestep di [0, T-1]
        noise : opsional, noise N(0, 1) dengan shape sama seperti x0; dibuat acak bila None.
                Training perlu membuat noise sendiri dan mengopernya, karena noise itu
                target yang diprediksi model.
        Mengembalikan x_t dengan shape sama seperti x0.

        Contoh satu gambar pada beberapa timestep dengan noise yang sama:
            ts = torch.tensor([0, 250, 500, 750, 999])
            xs = diffusion.q_sample(img.repeat(len(ts), 1, 1, 1), ts, noise.expand(len(ts), -1, -1, -1))
        """
        if noise is None:
            noise = torch.randn_like(x0)
        return (self._extract(self.sqrt_alpha_bars, t, x0) * x0
                + self._extract(self.sqrt_one_minus_alpha_bars, t, x0) * noise)


if __name__ == "__main__":
    diffusion = Diffusion()
    T = diffusion.timesteps
    print(f"T={T}, beta: {diffusion.betas[0]:.1e} -> {diffusion.betas[-1]:.1e}")
    print(f"alpha_bar[0]={diffusion.alpha_bars[0]:.5f}, alpha_bar[T-1]={diffusion.alpha_bars[-1]:.2e}")

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
        + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        torch.manual_seed(0)
        x0 = torch.rand(64, 3, 32, 32, device=device) * 2 - 1  # gambar acak di [-1, 1]
        noise = torch.randn_like(x0)
        t = torch.randint(0, T, (64,), device=device)
        xt = diffusion.q_sample(x0, t, noise)

        manual = (diffusion.alpha_bars[t.cpu()].sqrt().view(-1, 1, 1, 1) * x0.cpu()
                  + (1 - diffusion.alpha_bars[t.cpu()]).sqrt().view(-1, 1, 1, 1) * noise.cpu())
        x_first = diffusion.q_sample(x0, torch.zeros(64, dtype=torch.long, device=device), noise)
        x_last = diffusion.q_sample(x0, torch.full((64,), T - 1, device=device), noise)

        print(f"\n[{device}]")
        print("  shape x_t            :", tuple(xt.shape), "(sama dengan x0)" if xt.shape == x0.shape else "(BEDA!)")
        print(f"  selisih vs rumus     : {(xt.cpu() - manual).abs().max():.2e}")
        print(f"  t=0   maks |x_t - x0|: {(x_first - x0).abs().max():.4f}")
        print(f"  t=T-1 mean / std     : {x_last.mean():.4f} / {x_last.std():.4f}  (target ~0 / ~1)")
