"""Part 3-4 — per-modality encoders. All obey the same contract:
    out_dim: int ;  forward(x) -> (B, out_dim)
so fusion can be built against dummy encoders before real ones exist.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """18 -> 64 -> 64. BatchNorm, dropout, GELU."""
    out_dim = 64

    def __init__(self, n_feat=18, h=64, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p),
        )

    def forward(self, x):
        return self.net(x)


class AudioCNN(nn.Module):
    """4x (Conv-BN-GELU-MaxPool) on log-mel (B,1,128,T) -> 128-d."""
    out_dim = 128

    def __init__(self, p=0.3):
        super().__init__()
        chans = [1, 32, 64, 128, 128]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1),
                nn.BatchNorm2d(chans[i + 1]), nn.GELU(), nn.MaxPool2d(2),
            ]
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.pool(self.conv(x)).flatten(1)
        return self.drop(z)


class FaceCNN(nn.Module):
    """4 conv blocks (32/64/128/256) on (B,1,48,48) -> 128-d.
    Exposes emotion head (7-class) alongside the shared embedding."""
    out_dim = 128

    def __init__(self, p=0.3, n_emotion=7):
        super().__init__()
        chans = [1, 32, 64, 128, 256]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1),
                nn.BatchNorm2d(chans[i + 1]), nn.GELU(), nn.MaxPool2d(2),
            ]
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(p))
        self.emotion_head = nn.Linear(128, n_emotion)   # 7-class FER, for conflict flag at inference

    def forward(self, x, return_emotion=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.proj(self.pool(self.conv(x)).flatten(1))
        if return_emotion:
            return z, self.emotion_head(z)
        return z


class DummyEncoder(nn.Module):
    """Returns random tensors of the right shape — lets the training loop be
    tested at hour 2 before any real encoder exists (plan Part 8)."""
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self._p = nn.Parameter(torch.zeros(1))  # so optimiser has something

    def forward(self, x):
        b = x.shape[0]
        return torch.randn(b, self.out_dim, device=self._p.device) + self._p
