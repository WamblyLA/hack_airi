from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    num_times: int = 8192
    steps: int = 20_000
    patch_size: int = 64
    batch_size: int = 4
    samples_per_epoch: int = 2048
    workers: int = 2
    learning_rate: float = 2e-4
    aux_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    base_width: int = 64
    latent_channels: int = 48
    lambda_base_rate: float = 0.05
    lambda_full_rate: float = 0.025
    full_distortion_weight: float = 0.75
    detail_sparsity_weight: float = 2e-4
    spectral_weight: float = 0.002
    precipitation_weight: float = 0.02
