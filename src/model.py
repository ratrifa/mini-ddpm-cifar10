"""U-Net kecil untuk Mini DDPM: memprediksi noise dari gambar noisy x_t, timestep t, dan kelas y.

Konvensi input (harus sama dengan diffusion.py dan train.py):
- x : gambar noisy (B, 3, 32, 32)
- t : tensor long (B,), indeks timestep di [0, T-1]
- y : tensor long (B,), label kelas di [0, num_classes-1]. Indeks num_classes (=10)
      adalah token "tanpa kelas" untuk classifier-free guidance; y=None berarti semua tanpa kelas.
Output: prediksi noise dengan shape sama seperti x.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = 8  # jumlah grup GroupNorm; semua jumlah channel harus habis dibagi angka ini


def timestep_embedding(t, dim):
    """Embedding sinusoidal (seperti positional encoding Transformer): (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    """Residual block (seperti ResNet) yang menerima embedding waktu + kelas."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(GROUPS, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # skip connection perlu conv 1x1 bila jumlah channel berubah
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]  # info t dan y ditambahkan per channel
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, ch_mult=(1, 2, 2), num_res_blocks=2,
                 num_classes=10, dropout=0.1):
        super().__init__()
        self.base_ch = base_ch
        self.num_classes = num_classes
        emb_dim = base_ch * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)  # +1 untuk token tanpa kelas

        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # jalur turun (encoder): tiap level beberapa ResBlock, lalu resolusi dibagi 2 (kecuali level terakhir)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chs = []  # jumlah channel tiap skip connection
        ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, emb_dim, dropout))
                ch = out_ch
            self.down_blocks.append(blocks)
            chs.append(ch)
            last = i == len(ch_mult) - 1
            self.downsamples.append(nn.Identity() if last else nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), ResBlock(ch, ch, emb_dim, dropout)])

        # jalur naik (decoder): gabung skip dari encoder, ResBlock, lalu resolusi dikali 2
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch_block = ch + chs[i] if j == 0 else out_ch  # blok pertama menerima hasil concat skip
                blocks.append(ResBlock(in_ch_block, out_ch, emb_dim, dropout))
                ch = out_ch
            self.up_blocks.append(blocks)
            self.upsamples.append(nn.Identity() if i == 0 else nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch, ch, 3, padding=1),
            ))

        self.norm_out = nn.GroupNorm(GROUPS, ch)
        self.conv_out = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x, t, y=None):
        if y is None:
            y = torch.full((x.shape[0],), self.num_classes, dtype=torch.long, device=x.device)
        emb = self.time_mlp(timestep_embedding(t, self.base_ch)) + self.class_emb(y)

        h = self.conv_in(x)
        skips = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h, emb)
            skips.append(h)
            h = down(h)

        for block in self.mid:
            h = block(h, emb)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, emb)
            h = up(h)

        return self.conv_out(F.silu(self.norm_out(h)))


if __name__ == "__main__":
    model = UNet()
    print(f"jumlah parameter: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
        + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        model = model.to(device)
        x = torch.randn(8, 3, 32, 32, device=device)
        t = torch.randint(0, 1000, (8,), device=device)
        y = torch.tensor([0, 1, 2, 3, 4, 5, 9, 10], device=device)  # 10 = token tanpa kelas
        out = model(x, t, y)
        out_null = model(x, t)  # y=None
        print(f"[{device}] input {tuple(x.shape)} -> output {tuple(out.shape)}, "
              f"y=None -> {tuple(out_null.shape)}, sama: {out.shape == x.shape == out_null.shape}")
