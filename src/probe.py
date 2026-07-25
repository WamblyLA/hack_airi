import torch
import torch.nn as nn


class LatentProbe(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, channels, 3, padding=1),
        )
        if sum(p.numel() for p in self.parameters()) > 2_000_000:
            raise ValueError("Probe parameter limit exceeded")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + self.net(latent)
