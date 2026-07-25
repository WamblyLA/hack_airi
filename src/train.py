import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data import CHANNEL_NAMES, PatchDataset
from .model import SPARCCodec


def distortion(pred, target, latitude, ocean):
    w = torch.cos(torch.deg2rad(latitude)).clamp_min(1e-4).unsqueeze(1).unsqueeze(-1)
    error = (pred - target).square()
    num = (error * w).sum((-2, -1))
    den = (w.sum((-2, -1)) * target.shape[-1]).expand_as(num)
    mse = num / den.clamp_min(1e-8)
    sst_num = (error[:, 5] * w[:, 0] * ocean).sum((-2, -1))
    sst_den = (w[:, 0] * ocean).sum((-2, -1)).clamp_min(1e-8)
    mse[:, 5] = sst_num / sst_den
    return 0.5 * mse[:, :8].mean(1) + 0.5 * mse[:, 8:].mean(1)


def loss_fn(output, batch, cfg):
    base_d = distortion(output["base_reconstruction"], batch["x"], batch["latitude"], batch["ocean"]).mean()
    full_d = distortion(output["full_reconstruction"], batch["x"], batch["latitude"], batch["ocean"]).mean()
    values = float(np.prod(batch["x"].shape[1:]))
    base_r = (output["base_bits"] / values).mean()
    full_r = ((output["base_bits"] + output["detail_bits"]) / values).mean()
    spec = F.l1_loss(
        torch.log1p(torch.fft.rfft2(output["full_reconstruction"].float(), norm="ortho").abs().square()),
        torch.log1p(torch.fft.rfft2(batch["x"].float(), norm="ortho").abs().square()),
    )
    target_tp = batch["x"][:, 4]
    pred_tp = output["full_reconstruction"][:, 4]
    mask = (target_tp > 2.0).float()
    precip = ((pred_tp - target_tp).abs() * mask).sum() / mask.sum().clamp_min(1.0)
    sparse = output["detail_latent"].abs().mean()
    total = (
        base_d
        + cfg.full_distortion_weight * full_d
        + cfg.lambda_base_rate * base_r
        + cfg.lambda_full_rate * full_r
        + cfg.detail_sparsity_weight * sparse
        + cfg.spectral_weight * spec
        + cfg.precipitation_weight * precip
    )
    return total, {
        "loss": float(total.detach()),
        "base_distortion": float(base_d.detach()),
        "full_distortion": float(full_d.detach()),
        "base_bpv": float(base_r.detach()),
        "full_bpv": float(full_r.detach()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--num-times", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    args = parser.parse_args()
    cfg = TrainConfig()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.num_times is not None:
        cfg.num_times = args.num_times
    if args.samples_per_epoch is not None:
        cfg.samples_per_epoch = args.samples_per_epoch
    checkpoints_dir = args.out / "checkpoints"
    bitstreams_dir = args.out / "bitstreams"
    metrics_dir = args.out / "metrics"
    plots_dir = args.out / "plots"
    examples_dir = args.out / "examples"
    manifests_dir = args.out / "manifests"
    for directory in (
        checkpoints_dir,
        bitstreams_dir,
        metrics_dir,
        plots_dir,
        examples_dir,
        manifests_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PatchDataset(
        args.cache, args.stats, cfg.patch_size, cfg.samples_per_epoch,
        cfg.num_times, cfg.seed,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=device.type == "cuda",
        persistent_workers=cfg.workers > 0,
    )
    model = SPARCCodec(cfg.base_width, cfg.latent_channels).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count > 20_000_000:
        raise ValueError("Parameter limit exceeded")
    named = dict(model.named_parameters())
    aux_names = {name for name in named if name.endswith(".quantiles")}
    optimizer = torch.optim.AdamW(
        [p for name, p in named.items() if name not in aux_names],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    aux_optimizer = torch.optim.Adam(
        [named[name] for name in aux_names], lr=cfg.aux_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader)
    history = []
    started = time.perf_counter()
    progress = tqdm(range(cfg.steps), desc="Train SPARC-ERA5")
    for step in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch["x"], batch["static"])
            loss, metrics = loss_fn(output, batch, cfg)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        aux = model.aux_loss()
        aux.backward()
        aux_optimizer.step()
        scaler.update()
        if step % 25 == 0 or step + 1 == cfg.steps:
            metrics["step"] = step + 1
            history.append(metrics)
            progress.set_postfix(loss=f"{metrics['loss']:.3f}", bpv=f"{metrics['full_bpv']:.3f}")
    elapsed = time.perf_counter() - started
    model.update()
    model.eval()
    sample = dataset[0]
    x = sample["x"].unsqueeze(0).to(device)
    static = sample["static"].unsqueeze(0).to(device)
    streams = {}
    reconstructions = {}
    roundtrip = {}
    for mode, detail in (("base_64x", False), ("full_32x", True)):
        stream = model.encode(x, static, detail)
        reconstruction, latents_a = model.decode(stream, static, return_latents=True)
        reconstruction_b, latents_b = model.decode(stream, static, return_latents=True)
        exact = torch.equal(latents_a["base"], latents_b["base"])
        if detail:
            exact = exact and torch.equal(latents_a["detail"], latents_b["detail"])
        streams[mode] = stream
        reconstructions[mode] = reconstruction.cpu().numpy()
        roundtrip[mode] = exact
        (bitstreams_dir / f"sample_{mode}.bin").write_bytes(stream)
    raw_bytes = x.numel() * 4
    metrics = {
        "actual_base_cr": raw_bytes / len(streams["base_64x"]),
        "actual_full_cr": raw_bytes / len(streams["full_32x"]),
        "exact_roundtrip": roundtrip,
        "train_seconds": elapsed,
        "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.0,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
        "parameters": count,
        "steps": cfg.steps,
        "unique_train_times": dataset.num_times,
        "last": history[-1],
    }
    checkpoint = {
        "model": model.state_dict(),
        "width": cfg.base_width,
        "latent": cfg.latent_channels,
        "config": asdict(cfg),
        "channel_names": CHANNEL_NAMES,
    }
    checkpoint_path = checkpoints_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    np.savez(
        examples_dir / "sample_input.npz",
        x=x.cpu().numpy(),
        static=static.cpu().numpy(),
        base_reconstruction=reconstructions["base_64x"],
        full_reconstruction=reconstructions["full_32x"],
    )
    (metrics_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (metrics_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (metrics_dir / "exact_roundtrip.json").write_text(
        json.dumps(roundtrip, indent=2), encoding="utf-8"
    )
    resources = {
        "train_seconds": metrics["train_seconds"],
        "gpu_hours": metrics["gpu_hours"],
        "peak_vram_gib": metrics["peak_vram_gib"],
        "parameters": metrics["parameters"],
        "steps": metrics["steps"],
        "unique_train_times": metrics["unique_train_times"],
    }
    (metrics_dir / "resources.json").write_text(
        json.dumps(resources, indent=2), encoding="utf-8"
    )
    manifest = {
        "name": "SPARC-ERA5",
        "resolution": "0.5 degree",
        "channels": CHANNEL_NAMES,
        "train_cache": str(args.cache),
        "stats": str(args.stats),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    (manifests_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_np = x.cpu().numpy()[0]
    base_np = reconstructions["base_64x"][0]
    full_np = reconstructions["full_32x"][0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for row, channel in enumerate((0, 4)):
        for column, (array, title) in enumerate(
            ((x_np, "Original"), (base_np, "Base"), (full_np, "Base + detail"))
        ):
            image = axes[row, column].imshow(array[channel], cmap="RdBu_r")
            axes[row, column].set_title(f"{CHANNEL_NAMES[channel]} — {title}")
            axes[row, column].axis("off")
            fig.colorbar(image, ax=axes[row, column], fraction=0.046)
    fig.tight_layout()
    fig.savefig(plots_dir / "reconstruction_comparison.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([record["step"] for record in history], [record["loss"] for record in history])
    ax.set(xlabel="Optimizer step", ylabel="Loss", title="Training curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "training_curve.png", dpi=140)
    plt.close(fig)

    base_error = float(np.sqrt(np.mean((base_np - x_np) ** 2)))
    full_error = float(np.sqrt(np.mean((full_np - x_np) ** 2)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        [metrics["actual_base_cr"], metrics["actual_full_cr"]],
        [base_error, full_error],
        marker="o",
    )
    ax.set(
        xlabel="Actual compression ratio",
        ylabel="Normalized RMSE",
        title="Quality–bitrate",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_vs_bitrate.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([dataset.num_times], [full_error], s=70)
    ax.set(
        xlabel="Unique training frames",
        ylabel="Normalized RMSE",
        title="Quality–data",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_vs_data.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for array, label in (
        (x_np[0], "Original"),
        (base_np[0], "Base"),
        (full_np[0], "Base + detail"),
    ):
        spectrum = np.abs(np.fft.rfft2(array)) ** 2
        ax.loglog(spectrum.mean(axis=0)[1:], label=label)
    ax.set(xlabel="Zonal wavenumber", ylabel="Power", title="t2m spectrum")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "spectra.png", dpi=140)
    plt.close(fig)

    summary = {
        "architecture": "SPARC-ERA5 semantic progressive conditional hyperprior",
        "base_mode": {
            "bitstream": "bitstreams/sample_base_64x.bin",
            "actual_cr": metrics["actual_base_cr"],
            "normalized_rmse": base_error,
        },
        "full_mode": {
            "bitstream": "bitstreams/sample_full_32x.bin",
            "actual_cr": metrics["actual_full_cr"],
            "normalized_rmse": full_error,
        },
    }
    (metrics_dir / "submission_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
