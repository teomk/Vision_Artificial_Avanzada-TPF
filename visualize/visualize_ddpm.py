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
from ddpm import ConditionalDDPMUNet
from hf_utils import download_model
from ddpm_utils import inference, build_sigmoid_ddpm_scheduler
from visualize_utils import to_rgb, get_rgb_stats

# ── Visualización ──────────────────────────────────────────────────────

def visualize_samples(model, dataset, device, scheduler, sar_mode="None", n_samples=4, steps=50, save_path=None, seed=17):
    model.eval()
    np.random.seed(seed)
    indices    = np.random.choice(len(dataset), n_samples, replace=False)
    col_labels = ["Cloudy", "Prediction", "Ground Truth"]

    fig = plt.figure(figsize=(4 * 3, 4 * n_samples))
    gs  = gridspec.GridSpec(n_samples, 3, figure=fig, hspace=0.05, wspace=0.05)

    with torch.no_grad():
        for row, idx in enumerate(indices):
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

            pred = inference(model, condition, device, scheduler, steps=steps, sar=sar)
            pred = pred.squeeze(0).clamp(0, 1).cpu()

            stats = get_rgb_stats(cloudy, pred, clear)

            for col, img in enumerate([cloudy, pred, clear]):
                ax = fig.add_subplot(gs[row, col])
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

# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # python visualize/visualize_ddpm.py --config configs/ddpm_none.yaml
    # python visualize/visualize_ddpm.py --config configs/ddpm_concat.yaml
    # python visualize/visualize_ddpm.py --config configs/ddpm_controlnet.yaml

    parser = argparse.ArgumentParser(description="Visualizar predicciones ConditionalDDPMUNet")
    parser.add_argument("--config",    type=str, required=True,  help="Ruta al config YAML")
    parser.add_argument("--steps",     type=int, default=50,     help="Pasos de inferencia DDPM (default: 50)")
    parser.add_argument("--n_samples", type=int, default=4,      help="Cantidad de muestras (default: 4)")
    parser.add_argument("--save_path", type=str, default=None,   help="Ruta para guardar la figura")
    parser.add_argument("--seed",      type=int, default=17,     help="Semilla (default: 17)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sar_mode      = cfg["sar_mode"]
    repo_id       = cfg["huggingface"]["repo_id"]
    save_filename = cfg["huggingface"]["save_filename"]
    T             = cfg["train"]["T"]
    sigmoid_k =cfg["train"].get("sigmoid_k", 25.0),
    alpha_min =cfg["train"].get("alpha_min", 1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SAR: {sar_mode} | Steps: {args.steps}")

    checkpoint = download_model(repo_id=repo_id, filename=save_filename, map_location=device)

    image_channels     = 6
    condition_channels = 8 if sar_mode == "Concat" else 6

    model = ConditionalDDPMUNet(
        image_channels=image_channels,
        condition_channels=condition_channels,
        base_channels=64,
        time_dim=256,
    )
    model.load_state_dict(checkpoint)
    model = model.float().to(device)

    scheduler = build_sigmoid_ddpm_scheduler(
        T=T,
        sigmoid_k=sigmoid_k,
        alpha_min=alpha_min
    )

    ds = SEN12MSCRDataset(split="test", include_s1=(sar_mode != "None"), include_mask=False)

    visualize_samples(
        model=model,
        dataset=ds,
        device=device,
        scheduler=scheduler,
        sar_mode=sar_mode,
        n_samples=args.n_samples,
        steps=args.steps,
        save_path=args.save_path,
        seed=args.seed,
    )