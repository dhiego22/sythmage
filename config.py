from dataclasses import dataclass
from typing import Tuple

dir_base = ""
dir_target = ""
checkpoints_dir = 'checkpoints'

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 1
    lr: float = 2e-4
    lambda_l1: int = 100
    patch_size: tuple = (64,64,64)
    patches_per_volume: int = 32
