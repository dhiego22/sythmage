import os
import glob
import random
from typing import List, Tuple
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import minmax_norm

class PairedMRI3DPatches(Dataset):
    def __init__(self, dir_base:str, dir_target:str, 
                 patch_size: Tuple[int,int,int]=(64,64,64),
                 patches_per_volume: int=16):

        files_base = sorted(glob.glob(os.path.join(dir_base, '*.nii.gz'))) # or *.nii
        files_target = sorted(glob.glob(os.path.join(dir_target, '*.nii.gz'))) # or *.nii
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume

        def pairing_key(p:str) -> str:
            return os.path.splitext(os.path.basename(p))[0]

        map_base, map_target = {},  {}
        for p in files_base:
            map_base[pairing_key(p)] = p
        for p in files_target:
            map_target[pairing_key(p)] = p
        common = sorted(set(map_base) & set(map_target))
        if len(common) == 0:
            raise FileNotFoundError(f"No paired files; check directories. base={dir_base} target={dir_target}")
        self.pairs: List[Tuple[str,str]] = [(map_base[k], map_target[k]) for k in common]#Build a list of paired file paths (tuples (3T_path, 7T_path)), aligned by the same pairing key.
        
        self.cache_base: List[np.ndarray] = [None] * len(self.pairs) # caches for loaded volumes (numpy arrays).
        self.cache_target: List[np.ndarray] = [None] * len(self.pairs) #volumes are only loaded when first used, then kept in memory for re-use during the epoch (reducing disk I/O)

    def __len__(self):
            return len(self.pairs) * self.patches_per_volume

    def _get_volumes(self, idx_pair:int):
            v=vb, vt = self.cache_base[idx_pair], self.cache_target[idx_pair]
            if vb is None:
                pb, pt = self.pairs[idx_pair]
                vb = nib.load(pb).get_fdata().astype(np.float32)
                vt = nib.load(pt).get_fdata().astype(np.float32)
                vb = minmax_norm(vb)
                vt = minmax_norm(vt) 
                self.cache_base[idx_pair] = vb
                self.cache_target[idx_pair] = vt
            return vb, vt

    def __getitem__(self, idx:int):
            pair_idx = idx // self.patches_per_volume# //  floor division assignment operator. Choosing which pair to use
            vb, vt = self._get_volumes(pair_idx)
            H, W, D = vb.shape
            ph, pw, pd = self.patch_size
            sh = random.randint(0, max(0, H - ph)) if H >= ph else 0 #  patch extraction
            sw = random.randint(0, max(0, W - pw)) if W >= pw else 0
            sd = random.randint(0, max(0, D - pd)) if D >= pd else 0
            patchb = vb[sh:sh+ph, sw:sw+pw, sd:sd+pd]
            patcht = vt[sh:sh+ph, sw:sw+pw, sd:sd+pd]
            def pad_to_shape(x, target):
                pad_h = max(0, target[0] - x.shape[0])
                pad_w = max(0, target[1] - x.shape[1])
                pad_d = max(0, target[2] - x.shape[2])
                if pad_h or pad_w or pad_d:
                    x = np.pad(x, ((0, pad_h), (0, pad_w), (0, pad_d)), mode='reflect')
                return x
            patchb = pad_to_shape(patchb, self.patch_size)#Ensure both patches are exactly (ph, pw, pd)
            patcht = pad_to_shape(patcht, self.patch_size)
            tb = torch.from_numpy(patchb).permute(2,0,1).unsqueeze(0).float()# Convert numpy arrays → PyTorch tensors (float32)
            tt = torch.from_numpy(patcht).permute(2,0,1).unsqueeze(0).float()#final shapes: (1, pd, ph, pw) applying .unsqueeze(0)
            return tb, tt
