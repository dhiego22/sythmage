import os
import glob
import random
from typing import List, Tuple
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import minmax_norm

class PairedMRI3DPatches(Dataset):
    def __init__(self, dir_3t:str, dir_7t:str, 
                 patch_size: Tuple[int,int,int]=(64,64,64),
                 patches_per_volume: int=16):

        files_3t = sorted(glob.glob(os.path.join(dir_3t, '*.nii.gz'))) # or *.nii
        files_7t = sorted(glob.glob(os.path.join(dir_7t, '*.nii.gz'))) # or *.nii
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume

        def pairing_key(p:str) -> str:
            return os.path.splitext(os.path.basename(p))[0]

        map_3t, map_7t = {},  {}
        for p in files_3t:
            map_3t[pairing_key(p)] = p
        for p in files_7t:
            map_7t[pairing_key(p)] = p
        common = sorted(set(map_3t) & set(map_7t))
        if len(common) == 0:
            raise FileNotFoundError(f"No paired files; check directories. 3T={dir_3t} 7T={dir_7t}")
        self.pairs: List[Tuple[str,str]] = [(map_3t[k], map_7t[k]) for k in common]#Build a list of paired file paths (tuples (3T_path, 7T_path)), aligned by the same pairing key.
        
        self.cache_3t: List[np.ndarray] = [None] * len(self.pairs) # caches for loaded volumes (numpy arrays).
        self.cache_7t: List[np.ndarray] = [None] * len(self.pairs) #volumes are only loaded when first used, then kept in memory for re-use during the epoch (reducing disk I/O)

    def __len__(self):
            return len(self.pairs) * self.patches_per_volume

    def _get_volumes(self, idx_pair:int):
            v3, v7 = self.cache_3t[idx_pair], self.cache_7t[idx_pair]
            if v3 is None:
                p3, p7 = self.pairs[idx_pair]
                v3 = nib.load(p3).get_fdata().astype(np.float32)
                v7 = nib.load(p7).get_fdata().astype(np.float32)
                v3 = minmax_norm(v3)
                v7 = minmax_norm(v7) 
                self.cache_3t[idx_pair] = v3
                self.cache_7t[idx_pair] = v7
            return v3, v7

    def __getitem__(self, idx:int):
            pair_idx = idx // self.patches_per_volume# //  floor division assignment operator. Choosing which pair to use
            v3, v7 = self._get_volumes(pair_idx)
            H, W, D = v3.shape
            ph, pw, pd = self.patch_size
            sh = random.randint(0, max(0, H - ph)) if H >= ph else 0 #  patch extraction
            sw = random.randint(0, max(0, W - pw)) if W >= pw else 0
            sd = random.randint(0, max(0, D - pd)) if D >= pd else 0
            patch3 = v3[sh:sh+ph, sw:sw+pw, sd:sd+pd]
            patch7 = v7[sh:sh+ph, sw:sw+pw, sd:sd+pd]
            def pad_to_shape(x, target):
                pad_h = max(0, target[0] - x.shape[0])
                pad_w = max(0, target[1] - x.shape[1])
                pad_d = max(0, target[2] - x.shape[2])
                if pad_h or pad_w or pad_d:
                    x = np.pad(x, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='reflect')
                return x
            patch3 = pad_to_shape(patch3, self.patch_size)#Ensure both patches are exactly (ph, pw, pd)
            patch7 = pad_to_shape(patch7, self.patch_size)
            t3 = torch.from_numpy(patch3).permute(2,0,1).unsqueeze(0).float()# Convert numpy arrays → PyTorch tensors (float32)
            t7 = torch.from_numpy(patch7).permute(2,0,1).unsqueeze(0).float()#final shapes: (1, pd, ph, pw) applying .unsqueeze(0)
            return t3, t7
