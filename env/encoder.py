"""
Observation encoder for the car racing PPO agent.

Input
-----
img     : (B, 3, 64, 64)  float32, pixels normalised to 0..1
scalars : (B, 4)           float32, [speed, on_track, sin_angle, cos_angle]

Output
------
(B, 288)  flat feature vector → feed directly into actor / critic heads.

Architecture
------------
ImpalaCNN  (Espeholt et al., IMPALA 2018)
  3 blocks × (Conv → MaxPool → ResBlock → ResBlock)
  channels : 16 → 32 → 32
  64×64 input shrinks to 8×8 after 3 stride-2 MaxPools  →  32×8×8 = 2048 → FC(256)

  Key difference vs Nature CNN: each block adds two residual (skip) connections.
  Gradients flow straight back through the shortcuts, so early conv filters keep
  updating throughout training.  Empirically 3-5× more sample-efficient on
  visual RL tasks at identical inference cost.

Scalar MLP
  4 → 32 → 32  (speed, on_track, sin/cos angle)

Combined
  cat([img_features, scalar_features])  →  288-d vector
"""

import torch
import torch.nn as nn


# ── Building blocks ───────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)          # skip connection


class _ImpalaBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = _ResBlock(out_ch)
        self.res2 = _ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv(x))
        x = self.res1(x)
        x = self.res2(x)
        return x


# ── Encoders ──────────────────────────────────────────────────────────────────

class ImpalaCNN(nn.Module):
    """
    Encodes a (B, 3, 64, 64) image to a (B, 256) feature vector.

    Block channels [16, 32, 32]:
        input  64×64
        block1 32×32  (16 ch)
        block2 16×16  (32 ch)
        block3  8×8   (32 ch)  →  flatten 2048  →  FC 256
    """

    CHANNELS = [16, 32, 32]

    def __init__(self, in_channels: int = 3, out_features: int = 256):
        super().__init__()
        blocks, ch = [], in_channels
        for out_ch in self.CHANNELS:
            blocks.append(_ImpalaBlock(ch, out_ch))
            ch = out_ch
        self.cnn = nn.Sequential(*blocks, nn.ReLU())
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 8 * 8, out_features),
            nn.ReLU(),
        )
        self.out_features = out_features

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(img))


class RaceEncoder(nn.Module):
    """
    Full encoder: ImpalaCNN for image + small MLP for scalars, outputs
    concatenated feature vector for actor / critic heads.

    out_features = img_features (256) + scalar_features (32) = 288
    """

    def __init__(self, img_features: int = 256, scalar_features: int = 32):
        super().__init__()
        self.cnn = ImpalaCNN(out_features=img_features)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(4, scalar_features),
            nn.ReLU(),
            nn.Linear(scalar_features, scalar_features),
            nn.ReLU(),
        )
        self.out_features = img_features + scalar_features

    def forward(self, img: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        img     : (B, 3, 64, 64)  float32  pixels / 255
        scalars : (B, 4)           float32  [speed, on_track, sin_angle, cos_angle]
        returns : (B, out_features)
        """
        return torch.cat([self.cnn(img), self.scalar_mlp(scalars)], dim=-1)
