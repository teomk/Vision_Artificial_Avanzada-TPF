import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import sys
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dataset import SEN12MSCRDataset
from dbcr_simple import DBCRSimple
from hf_utils import download_model
from dbcr_utils import inference
from visualize_utils import to_rgb, get_rgb_stats, to_sar

def visualize_samples(model, dataset, device, sar_mode="None", n_samples=4, T=1000, steps = 10, sigmoid_k=10.0, save_path=None, seed=17):
    model.eval()
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    has_sar = sar_mode in ("Concat", "ControlNet")
    if has_sar:
        col_labels = ["Cloudy", "SAR", "Prediction", "Ground Truth"]
        n_cols = 4
    else:
        col_labels = ["Cloudy", "Prediction", "Ground Truth"]
        n_cols = 3

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_samples))
    gs  = gridspec.GridSpec(n_samples, n_cols, figure=fig, hspace=0.05, wspace=0.05)
 
    with torch.no_grad():
        for row, idx in enumerate(tqdm(indices, desc="Samples", unit="sample")):
            if sar_mode == "None":
                cloudy, clear = dataset[idx]
                cloudy_b  = cloudy.unsqueeze(0).float().to(device)
                condition = cloudy_b
                sar       = None
 
            elif sar_mode == "Concat":
                s1, cloudy, clear = dataset[idx]
                cloudy_b  = cloudy.unsqueeze(0).float().to(device)
                s1_b      = s1.unsqueeze(0).float().to(device)
                condition = torch.cat([cloudy_b, s1_b], dim=1)  # [1, 8, H, W]
                sar       = None
 
            elif sar_mode == "ControlNet":
                s1, cloudy, clear = dataset[idx]
                cloudy_b  = cloudy.unsqueeze(0).float().to(device)
                s1_b      = s1.unsqueeze(0).float().to(device)
                condition = cloudy_b                             # [1, 6, H, W]
                sar       = s1_b                                 # [1, 2, H, W]
 
            else:
                raise ValueError(f"sar_mode desconocido: '{sar_mode}'")
 
            pred = inference(model, cloudy_b, condition, device, T=T, steps=steps, sar=sar, sigmoid_k=sigmoid_k, show_progress=(steps>1))
            pred = pred.squeeze(0).clamp(0, 1).cpu()

            stats = get_rgb_stats(cloudy)

            if has_sar:
                imgs = [cloudy, to_sar(s1_b.squeeze(0)), pred, clear]
            else:
                imgs = [cloudy, pred, clear]

            for col, img in enumerate(imgs):
                ax = fig.add_subplot(gs[row, col])
                if has_sar and col == 1:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(to_rgb(img, stats=stats))
                ax.axis("off")
                if row == 0:
                    ax.set_title(col_labels[col], fontsize=11, pad=6)
 
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figura guardada en: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # python visualize/visualize_dbcr.py --config configs/dbcr_no_sar.yaml
    # python visualize/visualize_dbcr.py --config configs/dbcr_concat.yaml
    # python visualize/visualize_dbcr.py --config configs/dbcr_controlnet.yaml
    parser = argparse.ArgumentParser(description="Visualizar predicciones DBCR")
    parser.add_argument("--config", type=str, required=True, help="Ruta al config YAML")
    parser.add_argument("--steps", type=int, default=10, help="Pasos de inferencia")
    parser.add_argument("--n_samples", type=int, default=4, help="Cantidad de muestras")
    parser.add_argument("--save_path", type=str, default=None, help="Path")
    parser.add_argument("--seed", type=int, default=17, help="Seed")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sar_mode      = cfg["sar_mode"]
    repo_id       = cfg["huggingface"]["repo_id"]
    save_filename = cfg["huggingface"]["save_filename"]
    T             = cfg["train"]["T"]
    sigmoid_k = cfg["train"].get("sigmoid_k", 10.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SAR: {sar_mode} | Steps: {args.steps}")

    loaded = download_model(repo_id=repo_id, filename=save_filename, map_location=device)
    checkpoint = (loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded)

    image_channels = 6
    condition_channels = 8 if sar_mode == "Concat" else 6

    model = DBCRSimple(image_channels=image_channels, condition_channels=condition_channels, base_channels=64, time_dim=128, control_net=(sar_mode == "ControlNet"))
    model.load_state_dict(checkpoint, strict=False)
    model = model.float().to(device)

    ds = SEN12MSCRDataset(split="test", include_s1=(sar_mode != "None"), include_mask=False)

    visualize_samples(model=model, dataset=ds, device=device, sar_mode=sar_mode, n_samples=args.n_samples, T=T, steps=args.steps, sigmoid_k=sigmoid_k, save_path=args.save_path, seed=args.seed)