import numpy as np


import numpy as np
import torch


def get_rgb_stats(*tensors, bands=(2, 1, 0), p_low=2, p_high=98):
    """
    Calcula percentiles RGB comunes para varios tensores [C,H,W].
    Sirve para mostrar Cloudy / Prediction / Ground Truth con la misma escala.
    """
    imgs = []

    for tensor in tensors:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        img = tensor[[bands[0], bands[1], bands[2]]]
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        imgs.append(img)

    stacked = np.concatenate(
        [img.reshape(-1, 3) for img in imgs],
        axis=0
    )

    p_min = np.percentile(stacked, p_low, axis=0)
    p_max = np.percentile(stacked, p_high, axis=0)

    return p_min, p_max


def to_rgb(tensor, bands=(2, 1, 0), stats=None):
    """
    Convierte tensor S2 [C,H,W] a RGB.
    Si stats está definido, usa la misma escala para varias imágenes.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    img = tensor[[bands[0], bands[1], bands[2]]]
    img = img.permute(1, 2, 0).detach().cpu().numpy()

    if stats is not None:
        p_min, p_max = stats
        img = (img - p_min) / (p_max - p_min + 1e-8)
    else:
        img = np.clip(img, 0, 1)

    return np.clip(img, 0, 1)

# def to_rgb(tensor, bands=(2, 1, 0)):
#     img = tensor[[bands[0], bands[1], bands[2]]].permute(1, 2, 0).cpu().numpy()
#     img = np.clip(img, 0, 1)
#     out = np.zeros_like(img)
#     for c in range(3):
#         p2  = np.percentile(img[:, :, c], 2)
#         p98 = np.percentile(img[:, :, c], 98)
#         out[:, :, c] = np.clip((img[:, :, c] - p2) / (p98 - p2 + 1e-8), 0, 1)
#     return out

def to_gray(tensor):
    """Tensor [1, H, W] o [H, W] → array [H, W] para imshow."""
    if tensor.ndim == 3:
        return tensor[0].numpy()
    return tensor.numpy()

def to_sar(tensor, band=0):
    """Tensor SAR [2, H, W] → array [H, W] normalizado para visualizar."""
    img = tensor[band].numpy()
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    return np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)