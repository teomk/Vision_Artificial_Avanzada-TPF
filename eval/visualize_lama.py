# eval/visualize_lama.py
from pathlib import Path
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml
import argparse
from torch.utils.data import DataLoader

ROOT     = Path(__file__).resolve().parent.parent
LAMA_DIR = ROOT / "external" / "lama"
DATA_DIR = ROOT / "dataset"
sys.path.append(str(LAMA_DIR))
sys.path.append(str(DATA_DIR))

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from dataset_lama import SEN12MSCRDataset


# ── Modelo ─────────────────────────────────────────────────────────────

def build_model(use_sar: bool) -> FFCResNetGenerator:
    return FFCResNetGenerator(
        input_nc  = 9 if use_sar else 7,
        output_nc = 6,
        ngf       = 64,
        n_downsampling = 3,
        n_blocks   = 18,
        init_conv_kwargs      ={"ratio_gin": 0,    "ratio_gout": 0},
        downsample_conv_kwargs={"ratio_gin": 0,    "ratio_gout": 0},
        resnet_conv_kwargs    ={"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
    )


# ── Helpers de visualización ───────────────────────────────────────────

def to_rgb(tensor, bands=(2, 1, 0)):
    """
    Convierte un tensor [C, H, W] a imagen RGB [H, W, 3] para mostrar.
    Por defecto usa bandas (B4, B3, B2) → índices (2, 1, 0) en nuestro orden
    [B2, B3, B4, B8, B11, B12].
    """
    img = tensor[[bands[0], bands[1], bands[2]]].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    # stretch por canal independiente
    out = np.zeros_like(img)
    for c in range(3):
        p2, p98 = np.percentile(img[:, :, c], 2), np.percentile(img[:, :, c], 98)
        out[:, :, c] = np.clip((img[:, :, c] - p2) / (p98 - p2 + 1e-8), 0, 1)
    return out

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


# ── Visualización ──────────────────────────────────────────────────────

def visualize_samples(model, dataset, use_sar, device, n_samples=4, save_path=None):
    model.eval()

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    # Columnas: cloudy | sar (opcional) | mask | output | clear
    n_cols = 5 if use_sar else 4
    col_labels = (
        ["Nubosa (RGB)", "SAR (VV)", "Máscara", "Salida modelo", "Clear (GT)"]
        if use_sar else
        ["Nubosa (RGB)", "Máscara", "Salida modelo", "Clear (GT)"]
    )

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_samples))
    gs  = gridspec.GridSpec(n_samples, n_cols, figure=fig,
                            hspace=0.05, wspace=0.05)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]

            if use_sar:
                s1, cloudy, mask, clear = sample
                x = torch.cat([
                    cloudy * (1 - mask),
                    mask,
                    s1
                ], dim=0).unsqueeze(0).float().to(device)
            else:
                cloudy, mask, clear = sample
                x = torch.cat([
                    cloudy * (1 - mask),
                    mask
                ], dim=0).unsqueeze(0).float().to(device)

            output = model(x).squeeze(0).clamp(0, 1).cpu()

            # Armar lista de imágenes y colormaps para esta fila
            images = []
            cmaps  = []

            images.append(to_rgb(cloudy));   cmaps.append(None)
            if use_sar:
                images.append(to_sar(s1));   cmaps.append("gray")
            images.append(to_gray(mask));    cmaps.append("gray")
            images.append(to_rgb(output));   cmaps.append(None)
            images.append(to_rgb(clear));    cmaps.append(None)

            for col, (img, cmap) in enumerate(zip(images, cmaps)):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_labels[col], fontsize=11, pad=6)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()

    plt.close()


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # python eval/visualize_lama.py --config configs/lama_no_sar.yaml
    # python eval/visualize_lama.py --config configs/lama_sar.yaml --n_samples 6 --save

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Cantidad de imágenes a visualizar")
    parser.add_argument("--save",      action="store_true",
                        help="Guardar figura en vez de mostrarla")
    parser.add_argument("--split",     type=str, default="test",
                        choices=["train", "test"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    use_sar    = cfg.get("sar", False)
    model_path = ROOT / cfg["eval"]["model_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {device} | SAR: {use_sar} | split: {args.split}")

    model = build_model(use_sar=use_sar)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model = model.float().to(device)

    ds = SEN12MSCRDataset(split=args.split, include_s1=use_sar)

    save_path = None
    if args.save:
        out_dir = ROOT / "eval" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "sar" if use_sar else "no_sar"
        save_path = out_dir / f"visualize_lama_{tag}_{args.split}.png"

    visualize_samples(model, ds, use_sar=use_sar, device=device,
                      n_samples=args.n_samples, save_path=save_path)