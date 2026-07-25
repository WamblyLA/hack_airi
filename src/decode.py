import argparse
from pathlib import Path

import numpy as np
import torch

from .model import SPARCCodec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SPARCCodec(state["width"], state["latent"]).to(device)
    excluded = ("_quantized_cdf", "_offset", "_cdf_length", "scale_table")
    weights = {
        key: value
        for key, value in state["model"].items()
        if not key.endswith(excluded)
    }
    model.load_state_dict(weights, strict=False)
    model.update()
    static = torch.from_numpy(np.load(args.static)["static"]).to(device)
    reconstruction = model.decode(args.input.read_bytes(), static)
    np.save(args.output, reconstruction.cpu().numpy())


if __name__ == "__main__":
    main()
