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
from ddpm_utils import build_sigmoid_ddpm_scheduler
from dbcr_utils import make_bridge_sample
from visualize_utils import to_rgb, get_rgb_stats

def _visualization_timesteps(T, frame_every, device):
    timesteps = torch.arange(0, T, frame_every, device=device).long()
    last_t = torch.tensor(T - 1, device=device).long()

    if timesteps.numel() == 0 or timesteps[-1].item() != last_t.item():
        timesteps = torch.cat([timesteps, last_t.view(1)])

    return timesteps

def forward_ddpm(clean, scheduler, T, frame_every=100):
    device = clean.device

    clean_b = clean.unsqueeze(0)  # [1, 6, H, W]

    noise = torch.randn_like(clean_b)

    timesteps = _visualization_timesteps(T, frame_every, device)

    frames = []

    for t_val in timesteps:
        if t_val.item() == 0:
            x_t = clean_b.clone()
        else:
            t = t_val.repeat(clean_b.shape[0])
            x_t = scheduler.add_noise(clean_b, noise, t)

        frames.append((t_val.item(), x_t.squeeze(0).clamp(0, 1).detach().cpu()))

    return frames

def forward_bridge(clean, cloudy, T, frame_every=100, sigmoid_k=10.0):
    device = clean.device

    clean_b = clean.unsqueeze(0)    # [1, 6, H, W]
    cloudy_b = cloudy.unsqueeze(0)  # [1, 6, H, W]

    timesteps = _visualization_timesteps(T, frame_every, device)

    frames = []

    for t_val in timesteps:
        t = t_val.view(1)  # [1]

        if t_val.item() == 0:
            x_t_b = clean_b.clone()

        else:
            x_t_b = make_bridge_sample(s2_clean=clean_b, s2_cloudy=cloudy_b, t=t, T=T, sigmoid_k=sigmoid_k, device=device)

        x_t = x_t_b.squeeze(0).clamp(0, 1).detach().cpu()

        frames.append((t_val.item(), x_t))

    return frames

def build_figure(rows):
    n_cols = max(len(f) for _, _, f, _ in rows)
    n_rows = len(rows)

    fig = plt.figure(figsize=(4 * n_cols, 5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.05)

    for row_idx, (title, row_label, frames, stats) in enumerate(rows):

        padded = frames + [(None, None)] * (n_cols - len(frames))
        for col, (t_val, img) in enumerate(padded):
            ax = fig.add_subplot(gs[row_idx, col])
            if img is not None:
                ax.imshow(to_rgb(img, stats=stats))
                ax.set_title(f"t={t_val}", fontsize=20, pad=3)
            else:
                ax.set_visible(False)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9)

    return fig

def load_sample(dataset, idx, sar_mode, device):
    if sar_mode == "None":
        cloudy, clear = dataset[idx]
        cloudy_b  = cloudy.unsqueeze(0).float().to(device)

    elif sar_mode == "Concat":
        s1, cloudy, clear = dataset[idx]
        cloudy_b = cloudy.unsqueeze(0).float().to(device)

    elif sar_mode == "ControlNet":
        s1, cloudy, clear = dataset[idx]
        cloudy_b = cloudy.unsqueeze(0).float().to(device)

    else:
        raise ValueError(f"sar_mode desconocido: '{sar_mode}'")

    clear_b = clear.unsqueeze(0).float().to(device) if not isinstance(clear, torch.Tensor) else clear.unsqueeze(0).to(device)
    return cloudy_b, clear_b

if __name__ == "__main__":
    # python visualize/visualize_noise.py --config configs/ddpm_none.yaml
    # python visualize/visualize_noise.py --config configs/dbcr_no_sar.yaml
    # python visualize/visualize_noise.py --config configs/ddpm_none.yaml --idx 100 --save_path imgs/ddpm_noise.png

    parser = argparse.ArgumentParser(description="Visualizar proceso de ruido")
    parser.add_argument("--config", type=str, required=True, help="Ruta al config YAML")
    parser.add_argument("--frame_every", type=int, default=100, help="Frames cada n frames")
    parser.add_argument("--idx", type=int, default=0, help="Índice de muestra del dataset")
    parser.add_argument("--save_path", type=str, default=None, help="Ruta para guardar la figura")
    parser.add_argument("--seed", type=int, default=17, help="Seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_type = cfg["model"]
    sar_mode = cfg["sar_mode"]
    T = cfg["train"]["T"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {model_type} | SAR: {sar_mode}")

    ds = SEN12MSCRDataset(split="test", include_s1=(sar_mode != "None"), include_mask=False)
    cloudy_b, clear_b = load_sample(ds, args.idx, sar_mode, device)

    sigmoid_k = cfg["train"].get("sigmoid_k", 10.0)
    alpha_min = cfg["train"].get("alpha_min", 0.0001)

    scheduler = build_sigmoid_ddpm_scheduler(T=T, sigmoid_k=sigmoid_k, alpha_min=alpha_min)

    rows = []

    stats = get_rgb_stats(cloudy_b.squeeze(0))

    if model_type == "ddpm":
        fwd_frames = forward_ddpm(clear_b.squeeze(0), scheduler, T, args.frame_every)
        rows.append(("DDPM — Forward", "x_t", fwd_frames, stats))

    elif model_type == "dbcr":
        fwd_frames = forward_bridge(clear_b.squeeze(0), cloudy_b.squeeze(0), T, args.frame_every)
        rows.append(("DBCR — Forward", "x_t", fwd_frames, stats))

    fig = build_figure(rows)

    if args.save_path is not None:
        if not Path(args.save_path).parent.exists():
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_path, bbox_inches="tight", dpi=300)
        print(f"Figura guardada en: {args.save_path}")
    else:
        plt.show()
    plt.close()