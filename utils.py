import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
from torchmetrics.image.fid import (
    FrechetInceptionDistance,
)


def plot_metrics_over_epochs(*y_series, #*y_series (list of lists): One or more lists of metric values (e.g., loss, val_loss, accuracy)
                             labels=None, # labels (list of str, optional): Labels for each series. Length must match number of y_series
                             title="Training Metrics Over Epochs", # title (str): Plot title
                             y_label="Metric", # y_label (str): Label for the Y-axis
                             start_epoch=1, # start_epoch (int): Starting epoch number (default 1)
                             epoch_step=1, # epoch_step (int): Step between epochs (default 1)
                             styles=None, # styles (list of dict, optional): Matplotlib style dicts per series, e.g., {'color':'r','linestyle':'--','marker':'o'}
                             save_path="training_metrics.png", # save_path (str, optional): If provided, saves the figure to this path (e.g., 'plot.png')
                             show=False): # show (bool): Whether to call plt.show(). Set False if you only want to save

    if len(y_series) < 2:
        raise ValueError("Provide at least two series to plot.")

    # Validate lengths and prepare epoch axis
    lengths = [len(s) for s in y_series]
    if len(set(lengths)) != 1:
        raise ValueError(f"All series must have the same length, got lengths: {lengths}")

    n_points = lengths[0]
    epochs = list(range(start_epoch, start_epoch + n_points * epoch_step, epoch_step))

    # Validate labels and styles
    if labels is not None and len(labels) != len(y_series):
        raise ValueError("labels length must match the number of series.")
    if styles is not None and len(styles) != len(y_series):
        raise ValueError("styles length must match the number of series.")

    plt.figure(figsize=(9, 5.5))
    for i, series in enumerate(y_series):
        label = labels[i] if labels else f"Series {i+1}"
        style = styles[i] if styles else {}
        # Apply defaults, overridden by any provided style keys
        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        marker = style.get("marker", "o")
        linewidth = style.get("linewidth", 2)

        plt.plot(epochs, series, label=label,
                 color=color, linestyle=linestyle, marker=marker, linewidth=linewidth)

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        #os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()

def minmax_norm(arr: np.ndarray) -> np.ndarray: #min–max normalize each volume to range [-1, 1]
    amin, amax = float(arr.min()), float(arr.max())
    if amax <= amin:
        return np.zeros_like(arr, dtype=np.float32)
    norm = (arr - amin) / (amax - amin)
    return norm.astype(np.float32) * 2.0 - 1.0

def mse(x, y):
    return torch.mean((x - y) ** 2)

@torch.no_grad()
def compute_fid_3D(netG, loader, fid_metric, device, num_slices=8):
    """
    Computes FID for 3D MRI patches by slicing them into 2D images.
    TorchMetrics FID works only on 2D (HxW), so we extract slices from D dimension.
    """
    netG.eval()
    fid_metric.reset()

    for src3t, tgt7t in loader:
        src3t = src3t.to(device)
        tgt7t = tgt7t.to(device)

        fake7t = netG(src3t)

        B, C, D, H, W = tgt7t.shape

        # pick slices evenly from the depth dimension
        slice_idxs = torch.linspace(0, D - 1, num_slices).long()

        for idx in slice_idxs:
            real_slice = tgt7t[:, :, idx, :, :]    # shape: [B, 1, H, W]
            fake_slice = fake7t[:, :, idx, :, :]   # shape: [B, 1, H, W]

            # convert [-1,1] → uint8 [0,255]
            real_uint8 = ((real_slice.clamp(-1,1) + 1) * 127.5).to(torch.uint8)
            fake_uint8 = ((fake_slice.clamp(-1,1) + 1) * 127.5).to(torch.uint8)

        real_rgb = real_uint8.repeat(1, 3, 1, 1)
        fake_rgb = fake_uint8.repeat(1, 3, 1, 1)

        fid_metric.update(real_rgb, real=True)
        fid_metric.update(fake_rgb, real=False)

    return fid_metric.compute().item()
  
def r1_penalty(d_out, x_in):
    grad = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()

def normalize(x):
    min_x = min(x)
    max_x = max(x)
    # avoid division by zero if list has constant values
    if max_x == min_x:
        return [0.0 for _ in x]
    return [(v - min_x) / (max_x - min_x) for v in x] 
