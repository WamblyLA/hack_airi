import json
import math
import struct
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


CHANNEL_NAMES = [
    "t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc",
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Z1000", "Z925", "Z850", "Z700",
    "Q1000", "Q925", "Q850", "Q700",
]
SEMANTIC_GROUPS = [
    [0, 5, 8, 9, 10, 11],
    [2, 3, 12, 13, 14, 15, 16, 17, 18, 19],
    [1, 20, 21, 22, 23],
    [4, 6, 7, 24, 25, 26, 27],
]
MAGIC = b"SPRC1"


def pad_geo(x: torch.Tensor, p: int) -> torch.Tensor:
    x = F.pad(x, (p, p, 0, 0), mode="circular")
    return F.pad(x, (0, 0, p, p), mode="reflect")


class GeoConv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1):
        super().__init__()
        self.p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(pad_geo(x, self.p))


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.n1 = nn.GroupNorm(groups, channels)
        self.n2 = nn.GroupNorm(groups, channels)
        self.c1 = GeoConv(channels, channels)
        self.c2 = GeoConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(GeoConv(cin, cout, stride=2), ResBlock(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.proj = GeoConv(cin, cout)
        self.res = ResBlock(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.res(self.proj(x))


class WeatherTransform(nn.Module):
    def __init__(self, width: int = 40, latent: int = 32, stem: int = 8):
        super().__init__()
        self.stems = nn.ModuleList([
            nn.Sequential(GeoConv(len(group), stem), nn.SiLU(), ResBlock(stem))
            for group in SEMANTIC_GROUPS
        ])
        self.static_stem = nn.Sequential(GeoConv(4, stem), nn.SiLU(), ResBlock(stem))
        self.fuse = GeoConv(stem * 5, width, k=1)
        self.d1 = Down(width, width * 2)
        self.d2 = Down(width * 2, width * 3)
        self.d3 = Down(width * 3, width * 4)
        self.detail_head = GeoConv(width * 4, latent, k=1)
        self.d4 = Down(width * 4, width * 5)
        self.base_head = GeoConv(width * 5, latent, k=1)
        self.base_proj = GeoConv(latent, width * 5, k=1)
        self.u4 = Up(width * 5, width * 4)
        self.detail_proj = GeoConv(latent, width * 4, k=1)
        self.u3 = Up(width * 4, width * 3)
        self.u2 = Up(width * 3, width * 2)
        self.u1 = Up(width * 2, width)
        self.static_decode = GeoConv(4, width // 2)
        self.out = nn.Sequential(
            GeoConv(width + width // 2, width),
            nn.SiLU(),
            GeoConv(width, 28, k=1),
        )

    def encode(self, x: torch.Tensor, static: torch.Tensor):
        features = [stem(x[:, group]) for stem, group in zip(self.stems, SEMANTIC_GROUPS)]
        h = self.fuse(torch.cat([*features, self.static_stem(static)], dim=1))
        h = self.d3(self.d2(self.d1(h)))
        detail = self.detail_head(h)
        base = self.base_head(self.d4(h))
        return base, detail

    def decode(self, base: torch.Tensor, detail: torch.Tensor | None, static: torch.Tensor):
        h = self.u4(self.base_proj(base))
        if detail is not None:
            h = h + self.detail_proj(detail)
        h = self.u1(self.u2(self.u3(h)))
        s = self.static_decode(static)
        if h.shape[-2:] != s.shape[-2:]:
            h = F.interpolate(h, s.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([h, s], dim=1))


class HyperStream(nn.Module):
    def __init__(self, latent: int, conditional: bool):
        super().__init__()
        hw, hz = 48, 24
        self.ha = nn.Sequential(
            nn.Conv2d(latent, hw, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hw, hw, 5, 2, 2),
            nn.SiLU(),
            nn.Conv2d(hw, hz, 5, 2, 2),
        )
        self.hs = nn.Sequential(
            nn.ConvTranspose2d(hz, hw, 5, 2, 2, output_padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hw, hw, 5, 2, 2, output_padding=1),
            nn.SiLU(),
            nn.Conv2d(hw, 2 * latent, 3, padding=1),
        )
        self.context = nn.Conv2d(latent, 2 * latent, 3, padding=1) if conditional else None
        table = torch.exp(torch.linspace(math.log(0.11), math.log(256.0), 64))
        self.eb = EntropyBottleneck(hz)
        self.gc = GaussianConditional(table)

    def params(self, zhat: torch.Tensor, shape: Sequence[int], context=None):
        p = self.hs(zhat)
        if p.shape[-2:] != tuple(shape):
            p = F.interpolate(p, tuple(shape), mode="bilinear", align_corners=False)
        if self.context is not None:
            c = F.interpolate(context, tuple(shape), mode="bilinear", align_corners=False)
            p = p + self.context(c)
        raw, means = p.chunk(2, 1)
        return F.softplus(raw) + 0.11, means

    def forward(self, y: torch.Tensor, context=None):
        zhat, zlik = self.eb(self.ha(torch.abs(y)))
        scales, means = self.params(zhat, y.shape[-2:], context)
        yhat, ylik = self.gc(y, scales, means=means)
        bits = -torch.log2(ylik.clamp_min(1e-9)).flatten(1).sum(1)
        bits += -torch.log2(zlik.clamp_min(1e-9)).flatten(1).sum(1)
        return yhat, bits

    def update(self):
        self.eb.update(force=True)
        self.gc.update_scale_table(self.gc.scale_table, force=True)

    @torch.no_grad()
    def compress(self, y: torch.Tensor, context=None):
        z = self.ha(torch.abs(y))
        zs = self.eb.compress(z)
        zhat = self.eb.decompress(zs, z.shape[-2:])
        scales, means = self.params(zhat, y.shape[-2:], context)
        indexes = self.gc.build_indexes(scales)
        ys = self.gc.compress(y, indexes, means=means)
        return {"strings": [ys, zs], "z_shape": list(z.shape[-2:]), "y_shape": list(y.shape[-2:])}

    @torch.no_grad()
    def decompress(self, payload, context=None):
        ys, zs = payload["strings"]
        zhat = self.eb.decompress(zs, tuple(payload["z_shape"]))
        scales, means = self.params(zhat, tuple(payload["y_shape"]), context)
        indexes = self.gc.build_indexes(scales)
        return self.gc.decompress(ys, indexes, means=means)


class SPARCCodec(nn.Module):
    def __init__(self, width: int = 40, latent: int = 32):
        super().__init__()
        self.width = width
        self.latent = latent
        self.transform = WeatherTransform(width, latent)
        self.base_stream = HyperStream(latent, conditional=False)
        self.detail_stream = HyperStream(latent, conditional=True)

    def forward(self, x: torch.Tensor, static: torch.Tensor):
        by, dy = self.transform.encode(x, static)
        bh, bb = self.base_stream(by)
        dh, db = self.detail_stream(dy, bh)
        return {
            "base_reconstruction": self.transform.decode(bh, None, static),
            "full_reconstruction": self.transform.decode(bh, dh, static),
            "base_latent": bh,
            "detail_latent": dh,
            "base_bits": bb,
            "detail_bits": db,
        }

    def aux_loss(self):
        return self.base_stream.eb.loss() + self.detail_stream.eb.loss()

    def update(self):
        self.base_stream.update()
        self.detail_stream.update()

    @torch.no_grad()
    def encode(self, x: torch.Tensor, static: torch.Tensor, detail: bool):
        self.eval()
        by, dy = self.transform.encode(x, static)
        bp = self.base_stream.compress(by)
        bh = self.base_stream.decompress(bp)
        dp = self.detail_stream.compress(dy, bh) if detail else None
        payloads = [bp] + ([dp] if dp else [])
        strings = []
        for payload in payloads:
            strings.extend([payload["strings"][0][0], payload["strings"][1][0]])
        header = {
            "detail": detail,
            "shape": list(x.shape),
            "streams": [{"z_shape": p["z_shape"], "y_shape": p["y_shape"]} for p in payloads],
            "lengths": [len(s) for s in strings],
        }
        hb = json.dumps(header, separators=(",", ":")).encode()
        return MAGIC + struct.pack(">I", len(hb)) + hb + b"".join(strings)

    @torch.no_grad()
    def decode(self, stream: bytes, static: torch.Tensor, return_latents=False):
        if stream[:5] != MAGIC:
            raise ValueError("Invalid bitstream")
        n = struct.unpack(">I", stream[5:9])[0]
        header = json.loads(stream[9:9 + n])
        cursor = 9 + n
        strings = []
        for length in header["lengths"]:
            strings.append(stream[cursor:cursor + length])
            cursor += length
        payloads = []
        for i, meta in enumerate(header["streams"]):
            payloads.append({
                "strings": [[strings[2 * i]], [strings[2 * i + 1]]],
                "z_shape": meta["z_shape"],
                "y_shape": meta["y_shape"],
            })
        bh = self.base_stream.decompress(payloads[0])
        dh = self.detail_stream.decompress(payloads[1], bh) if header["detail"] else None
        reconstruction = self.transform.decode(bh, dh, static)
        if return_latents:
            return reconstruction, {"base": bh, "detail": dh}
        return reconstruction
