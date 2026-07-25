from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


CHANNEL_NAMES = [
    "t2m", "mslp", "u10", "v10", "tp6h", "sst", "tcwv", "tcc",
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Z1000", "Z925", "Z850", "Z700",
    "Q1000", "Q925", "Q850", "Q700",
]


def wrapped(array, prefix, lat_slice, lon_start, width):
    total = array.shape[-1]
    end = lon_start + width
    key = (*prefix, lat_slice)
    if end <= total:
        return np.asarray(array[(*key, slice(lon_start, end))])
    a = np.asarray(array[(*key, slice(lon_start, total))])
    b = np.asarray(array[(*key, slice(0, end - total))])
    return np.concatenate([a, b], axis=-1)


class PatchDataset(Dataset):
    def __init__(self, cache: Path, stats: Path, patch: int, length: int, num_times: int, seed: int):
        self.root = zarr.open_group(str(cache), mode="r")
        self.weather = self.root["weather"]
        self.static = self.root["static"]
        self.ocean = self.root["ocean_weight"]
        self.latitude = np.asarray(self.root["latitude"][:], dtype=np.float32)
        values = np.load(stats)
        self.mean = np.asarray(values["mean"], dtype=np.float32)
        self.std = np.maximum(np.asarray(values["std"], dtype=np.float32), 1e-6)
        self.patch = patch
        self.length = length
        self.num_times = min(num_times, self.weather.shape[0])
        self.seed = seed
        if self.weather.shape[1] != 28:
            raise ValueError("Cache must contain 28 channels")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        t = int(rng.integers(self.num_times))
        h, w = self.weather.shape[-2:]
        sampled = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0)))
        center = int(np.argmin(np.abs(self.latitude - sampled)))
        y = int(np.clip(center - self.patch // 2, 0, h - self.patch))
        x = int(rng.integers(w))
        ls = slice(y, y + self.patch)
        weather = wrapped(self.weather, (t, slice(None)), ls, x, self.patch).astype(np.float32)
        static = wrapped(self.static, (slice(None),), ls, x, self.patch).astype(np.float32)
        ocean = wrapped(self.ocean, tuple(), ls, x, self.patch).astype(np.float32)
        weather = (weather - self.mean[:, None, None]) / self.std[:, None, None]
        weather[5] = np.where(ocean > 0.5, weather[5], 0.0)
        return {
            "x": torch.from_numpy(weather),
            "static": torch.from_numpy(static),
            "ocean": torch.from_numpy(ocean),
            "latitude": torch.from_numpy(self.latitude[ls]),
            "time_index": torch.tensor(t),
        }
