import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import sys
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


def to_rgb(tensor, bands=(2, 1, 0)):
    img = tensor[[bands[0], bands[1], bands[2]]].permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    out = np.zeros_like(img)
    for c in range(3):
        p2  = np.percentile(img[:, :, c], 2)
        p98 = np.percentile(img[:, :, c], 98)
        out[:, :, c] = np.clip((img[:, :, c] - p2) / (p98 - p2 + 1e-8), 0, 1)
    return out


def sigmoid_scheduler(T, sigmoid_k, t, device):
    tau   = torch.clamp(t.float() / T, 0.0, 1.0)
    s     = torch.sigmoid((tau - 0.5) * sigmoid_k)
    s_min = torch.sigmoid(torch.tensor(-0.5 * sigmoid_k, device=device))
    s_max = torch.sigmoid(torch.tensor( 0.5 * sigmoid_k, device=device))
    alpha = torch.clamp((s - s_min) / (s_max - s_min), 0.0, 1.0)
    return alpha[:, None, None, None]


def inference(model, cloudy_b, condition, device, T=1000, steps=10):
    """
    cloudy_b:  [1, 6, H, W] — solo S2, usado para el bridge (x_t)
    condition: [1, 6, H, W] o [1, 8, H, W] — lo que recibe el modelo como s2_cloudy
               (6ch para No-SAR, 8ch para SAR-concat)
    """
    x_t = cloudy_b.clone()
    timesteps = torch.linspace(T, 1, steps).long().to(device)
    with torch.no_grad():
        for t_val in timesteps:
            t = t_val.repeat(x_t.shape[0])
            pred_clean = model(x_t=x_t, t=t, s2_cloudy=condition)
            alpha_prev = sigmoid_scheduler(T, sigmoid_k=10.0, t=(t_val - T // steps).clamp(min=1).repeat(x_t.shape[0]), device=device)
            x_t = (1.0 - alpha_prev) * pred_clean + alpha_prev * cloudy_b
    return pred_clean


def visualize_samples(model, dataset, device, use_sar=False, n_samples=4, T=1000, steps=10, save_path=None, seed=17):
    model.eval()
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    col_labels = ["Cloudy", "Prediction", "Ground Truth"]
    fig = plt.figure(figsize=(4 * 3, 4 * n_samples))
    gs  = gridspec.GridSpec(n_samples, 3, figure=fig, hspace=0.05, wspace=0.05)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            if use_sar:
                cloudy, s1, clear = dataset[idx]
                cloudy_b  = cloudy.unsqueeze(0).float().to(device)
                s1_b      = s1.unsqueeze(0).float().to(device)
                condition = torch.cat([cloudy_b, s1_b], dim=1)  # [1, 8, H, W]
            else:
                cloudy, clear = dataset[idx]
                cloudy_b  = cloudy.unsqueeze(0).float().to(device)
                condition = cloudy_b  # [1, 6, H, W]

            pred = inference(model, cloudy_b, condition, device, T=T, steps=steps)
            pred = pred.squeeze(0).clamp(0, 1).cpu()

            for col, img in enumerate([to_rgb(cloudy), to_rgb(pred), to_rgb(clear)]):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(img)
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
    # python eval/visualize_dbcr.py --config configs/dbcr_no_sar.yaml



    parser = argparse.ArgumentParser(description="Visualizar predicciones DBCR")
    parser.add_argument("--config", type=str, required=True, help="Ruta al config YAML")
    parser.add_argument("--steps", type=int, default=10, help="Pasos de inferencia iterativa (default: 10)")
    parser.add_argument("--n_samples", type=int, default=4, help="Cantidad de muestras (default: 4)")
    parser.add_argument("--save_path", type=str, default=None, help="Ruta para guardar la figura")
    parser.add_argument("--seed", type=int, default=17, help="Semilla (default: 17)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    use_sar       = cfg["sar"]
    repo_id       = cfg["huggingface"]["repo_id"]
    save_filename = cfg["huggingface"]["save_filename"]
    T             = cfg["train"]["T"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SAR: {use_sar} | Steps: {args.steps}")

    checkpoint = download_model(repo_id=repo_id, filename=save_filename, map_location=device)

    image_channels     = 6
    condition_channels = 8 if use_sar else 6

    model = DBCRSimple(image_channels=image_channels, condition_channels=condition_channels, base_channels=64, time_dim=128)
    model.load_state_dict(checkpoint)
    model = model.float().to(device)

    ds = SEN12MSCRDataset(split="test", include_s1=use_sar, include_mask=False)

    visualize_samples(model=model, dataset=ds, device=device, use_sar=use_sar, n_samples=args.n_samples, T=T, steps=args.steps,
                        save_path=args.save_path, seed=args.seed)