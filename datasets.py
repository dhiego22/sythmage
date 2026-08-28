import os
import glob
from typing import List, Tuple, Optional
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import minmax_norm

class PairedMRI3DPatches(Dataset):
    def __init__(
        self,
        dir_base: str,
        dir_target: str,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patches_per_volume: int = 16,
    ):

        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume

        files_base = sorted(glob.glob(os.path.join(dir_base, "*")))
        files_target = sorted(glob.glob(os.path.join(dir_target, "*")))

        files_base = [f for f in files_base if os.path.isfile(f)]
        files_target = [f for f in files_target if os.path.isfile(f)]

        map_base = {os.path.basename(f): f for f in files_base}
        map_target = {os.path.basename(f): f for f in files_target}

        common_files = sorted(set(map_base.keys()) & set(map_target.keys()))

        if not common_files:
            raise FileNotFoundError(
                f"No matching filenames found.\n"
                f"Base folder: {dir_base}\n"
                f"Target folder: {dir_target}\n\n"
                f"Files must have identical names/extensions."
            )

        self.pairs: List[Tuple[str, str]] = [
            (map_base[name], map_target[name])
            for name in common_files
        ]

        self.shapes: List[Tuple[int, int, int]] = []
        print(f"Found {len(self.pairs)} paired volumes.")

        for pb, pt in self.pairs:
            shape_b = nib.load(pb).shape
            shape_t = nib.load(pt).shape

            if shape_b != shape_t:
                raise ValueError(
                    f"Shape mismatch detected:\n"
                    f"{pb}: {shape_b}\n"
                    f"{pt}: {shape_t}"
                )

            self.shapes.append(shape_b)

        self.cache_base: List[Optional[np.ndarray]] = [None] * len(self.pairs)
        self.cache_target: List[Optional[np.ndarray]] = [None] * len(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_volume

    @staticmethod
    def pad_to_shape(
        x: np.ndarray,
        target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        pad_h = max(0, target_shape[0] - x.shape[0])
        pad_w = max(0, target_shape[1] - x.shape[1])
        pad_d = max(0, target_shape[2] - x.shape[2])

        if pad_h or pad_w or pad_d:
            x = np.pad(
                x,
                (
                    (0, pad_h),
                    (0, pad_w),
                    (0, pad_d),
                ),
                mode="reflect",
            )

        return x

    def _get_volumes(self, idx_pair: int):
        vb = self.cache_base[idx_pair]
        vt = self.cache_target[idx_pair]

        if vb is None:

            pb, pt = self.pairs[idx_pair]

            img_b = nib.load(pb)
            img_t = nib.load(pt)

            # More memory efficient than get_fdata()
            vb = np.asarray(img_b.dataobj, dtype=np.float32)
            vt = np.asarray(img_t.dataobj, dtype=np.float32)

            vb = minmax_norm(vb)
            vt = minmax_norm(vt)

            self.cache_base[idx_pair] = vb
            self.cache_target[idx_pair] = vt

        return vb, vt

    def __getitem__(self, idx: int):
        pair_idx = idx // self.patches_per_volume
        vb, vt = self._get_volumes(pair_idx)
        H, W, D = vb.shape
        ph, pw, pd = self.patch_size

        sh = np.random.randint(H - ph + 1) if H >= ph else 0
        sw = np.random.randint(W - pw + 1) if W >= pw else 0
        sd = np.random.randint(D - pd + 1) if D >= pd else 0

        patch_b = vb[
            sh : sh + ph,
            sw : sw + pw,
            sd : sd + pd,
        ]

        patch_t = vt[
            sh : sh + ph,
            sw : sw + pw,
            sd : sd + pd,
        ]

        patch_b = self.pad_to_shape(patch_b, self.patch_size)
        patch_t = self.pad_to_shape(patch_t, self.patch_size)

        tensor_b = (
            torch.from_numpy(patch_b)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )

        tensor_t = (
            torch.from_numpy(patch_t)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )

        return tensor_b, tensor_t
