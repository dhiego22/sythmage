import os
import glob
import nibabel as nib
import numpy as np
import torch
from models import UNetGenerator3D
from utils import minmax_norm

def make_3d_window(patch_size, kind='hamming', power=1.0, floor=1e-3):
    ph, pw, pd = patch_size

    if kind == 'hann':
        wx = np.hanning(ph); wy = np.hanning(pw); wz = np.hanning(pd)
    elif kind == 'hamming':
        wx = np.hamming(ph); wy = np.hamming(pw); wz = np.hamming(pd)
    elif kind == 'gaussian':
        # Centered Gaussian per axis; sigma ~ size/6 so ~3σ spans patch
        def gauss(n, sigma=None):
            if sigma is None:
                sigma = n / 6.0
            x = np.arange(n) - (n - 1) / 2.0
            return np.exp(-0.5 * (x / sigma) ** 2)
        wx = gauss(ph); wy = gauss(pw); wz = gauss(pd)
    elif kind == 'ones':
        wx = np.ones(ph); wy = np.ones(pw); wz = np.ones(pd)
    else:
        raise ValueError(f"Unsupported window kind: {kind}")

    # Separable outer product → (H, W, D)
    w = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    # Normalize peak to 1, then apply power and floor
    w = w / (w.max() + 1e-8)
    if power != 1.0:
        w = np.power(w, power)
    if floor is not None and floor > 0:
        w = np.clip(w, floor, None)

    return w.astype(np.float32)


def infer_full_volume_patches(
    netG,
    vol3: np.ndarray,
    device,
    patch_size=(64, 64, 64),
    overlap=(32, 32, 32),
    window_kind='hamming',
    window_power=1.0,
    window_floor=1e-3,
) -> np.ndarray:
    """Reconstruct synthetic volume using overlapping patches with windowed blending.
    vol3: numpy array (H,W,D) normalized to [-1,1]
    Returns numpy array (H,W,D) in [-1,1]
    """
    H, W, D = vol3.shape
    ph, pw, pd = patch_size
    oh, ow, od = overlap

    # Output accumulators
    out = np.zeros((H, W, D), dtype=np.float32)
    cnt = np.zeros((H, W, D), dtype=np.float32)

    # Precompute blending window for full patch
    win = make_3d_window(patch_size, kind=window_kind, power=window_power, floor=window_floor)

    def gen_starts(L, P, O):
        starts = list(range(0, max(1, L - P + 1), max(1, P - O)))
        if len(starts) == 0 or starts[-1] != max(0, L - P):
            starts.append(max(0, L - P))
        return starts

    sh_list = gen_starts(H, ph, oh)
    sw_list = gen_starts(W, pw, ow)
    sd_list = gen_starts(D, pd, od)

    netG.eval()
    with torch.no_grad():
        for sh in sh_list:
            for sw in sw_list:
                for sd in sd_list:
                    # Extract patch (may be smaller at borders)
                    patch = vol3[sh:sh+ph, sw:sw+pw, sd:sd+pd]
                    ph2, pw2, pd2 = patch.shape

                    # Pad to full patch size if needed
                    if patch.shape != (ph, pw, pd):
                        patch = np.pad(
                            patch,
                            ((0, ph - ph2), (0, pw - pw2), (0, pd - pd2)),
                            mode='reflect'
                        )

                    # (H,W,D) → (D,H,W), add batch+channel → (1,1,D,H,W)
                    t = (
                        torch.from_numpy(patch)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                        .float()
                    )

                    # Forward and bring back to NumPy
                    g = netG(t)[0, 0].cpu().numpy()  # (D,H,W)

                    # Crop prediction back to unpadded extents
                    g = g[:pd2, :ph2, :pw2]          # (D,H,W)
                    g_hw_d = np.transpose(g, (1, 2, 0))  # → (H,W,D)

                    # Corresponding window crop (H,W,D)
                    w_crop = win[:ph2, :pw2, :pd2]

                    # Weighted accumulation
                    out[sh:sh+ph2, sw:sw+pw2, sd:sd+pd2] += w_crop * g_hw_d
                    cnt[sh:sh+ph2, sw:sw+pw2, sd:sd+pd2] += w_crop

    # Safety: avoid division by zero (shouldn’t happen with floor > 0)
    cnt[cnt == 0] = 1.0
    return out / cnt


def reconstruct(checkpoint_path: str,
                  dir_base: str,
                  out_dir: str="",
                  patch_size=(64,64,64),
                  overlap=(32,32,32),
                  zero_eps: float = 0.0  # set e.g. 1e-6 if you want "near-zero" treated as zero
                 ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netG = UNetGenerator3D(in_ch=1, out_ch=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(ckpt['netG'])
    netG.eval()

    os.makedirs(out_dir, exist_ok=True)

    files_3t = sorted(glob.glob(os.path.join(dir_3t, '*.nii*')))
    print(f'Found {len(files_3t)} 3T files in {dir_3t}')

    for src3t_path in files_3t:
        src_nii = nib.load(src3t_path)
        vol3 = src_nii.get_fdata().astype(np.float32)

        # ---- (1) Build mask from ORIGINAL 3T (unnormalized) ----
        # exact zeros:
        if zero_eps == 0.0:
            mask = (vol3 != 0).astype(np.float32)
        else:
            mask = (np.abs(vol3) > zero_eps).astype(np.float32)

        # ---- (2) Your existing pipeline ----
        vol3n = minmax_norm(vol3)
        synth_norm = infer_full_volume_patches(
            netG, vol3n, device,
            patch_size=patch_size,
            overlap=overlap
        )
        synth_01 = (synth_norm + 1.0) / 2.0

        # ---- (3) Apply mask AFTER reconstruction ----
        # ensure shape compatibility
        #if synth_01.shape != mask.shape:
        #    raise ValueError(f"Shape mismatch: synth={synth_01.shape}, mask={mask.shape}")

        synth_01_masked = synth_01 * mask
        #synth_01_masked = synth_01

        # ---- (4) Save ----
        base = os.path.basename(src3t_path)
        stem = base[:-7] if base.endswith('.nii.gz') else os.path.splitext(base)[0]
        out_name = f'{stem}_synthetic_7T.nii.gz'
        out_path = os.path.join(out_dir, out_name)

        synth_img = nib.Nifti1Image(synth_01_masked.astype(np.float32), src_nii.affine, src_nii.header)
        synth_img.header.set_data_dtype(np.float32)
        synth_img.header['descrip'] = 'pix2pix3D synthetic 7T-like (masked by 3T zeros)'
        nib.save(synth_img, out_path)
        print(f'[Saved] {out_path}')
