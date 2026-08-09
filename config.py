from dataclasses import dataclass
from pathlib import Path
@dataclass
class TrainConfig:
    dir_3t:str=''
    dir_7t:str=''
    epochs:int=100
    batch_size:int=1
    lr:float=2e-4
    lambda_l1:float=100.0
    patch_size:tuple=(64,64,64)
    patches_per_volume:int=32
    checkpoint_dir:str='checkpoints'
