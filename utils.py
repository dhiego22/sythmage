import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


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
    epochs = np.arange(start_epoch, start_epoch + n_points * epoch_step, epoch_step)

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
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def znorm(arr: np.ndarray) -> np.ndarray:
    mean = float(arr.mean())
    std = float(arr.std())
    if std <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    norm = (arr - mean) / std
    norm = np.clip(norm, -3.0, 3.0) / 3.0
    return norm.astype(np.float32)

def mse(x, y):
    return F.mse_loss(x, y)
 
def r1_penalty(d_out, x_in):
    grad = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()
