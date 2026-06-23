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
from dbcr_complex import DBCR
from hf_utils import download_model
from dbcr_utils import inference
from dataset_utils import unpack_batch
from visualize_utils import to_rgb, get_rgb_stats, to_sar

def parse_window(value):
    if value is None:
        return None
    value = str(value).lower()
    if value in ["none", "null", "-1"]:
        return None
    return int(value)

def visualize_samples(model, dataset, device, sar_mode="None", n_samples=4, T=1000, steps=10, sigmoid_k=10.0, save_path=None, seed=17):
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
            sample = dataset[idx]

            if sar_mode == "None":
                cloudy, clear = sample
                s1 = None
            else:
                s1, cloudy, clear = sample

            # unpack_batch espera un batch con dimensión de batch
            # lo simulamos agregando y sacando la dim 0
            fake_batch = (
                (s1.unsqueeze(0) if s1 is not None else None),
                cloudy.unsqueeze(0),
                clear.unsqueeze(0),
            )
            s2_cloudy, s2_clean, condition, sar = unpack_batch(fake_batch, sar_mode, device)

            pred = inference(
                model, s2_cloudy, condition, device,
                T=T, steps=steps, sar=sar, sigmoid_k=sigmoid_k,
                show_progress=(steps > 1)
            )
            pred = pred.squeeze(0).clamp(0, 1).cpu()

            stats = get_rgb_stats(cloudy)

            if has_sar:
                imgs = [cloudy, to_sar(s1.unsqueeze(0).squeeze(0)), pred, clear]
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
    # python visualize/visualize_dbcr_complex.py --config configs/dbcr_complex.yaml
    parser = argparse.ArgumentParser(description="Visualizar predicciones DBCR Complex")
    parser.add_argument("--config",    type=str, required=True)
    parser.add_argument("--steps",     type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed",      type=int, default=17)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sar_mode  = cfg["sar_mode"]
    repo_id   = cfg["huggingface"]["repo_id"]
    filename  = cfg["huggingface"]["save_filename"]
    T         = cfg["train"]["T"]
    sigmoid_k = cfg["train"].get("sigmoid_k", 10.0)

    cfg_model = cfg.get("model_args", {})
    image_channels     = int(cfg_model.get("image_channels", 6))
    condition_channels = int(cfg_model.get("condition_channels", 6))
    sar_channels       = int(cfg_model.get("sar_channels", 2))
    base_channels      = int(cfg_model.get("base_channels", 64))
    time_dim           = int(cfg_model.get("time_dim", 128))
    num_heads          = int(cfg_model.get("num_heads", 1))
    window_size_sf0     = parse_window(cfg_model.get("window_size_sf0", 8))
    window_size_not_sf0 = parse_window(cfg_model.get("window_size_not_sf0", None))
    use_checkpoint     = cfg_model.get("use_checkpoint", True)
    include_encoder_4  = cfg_model.get("include_encoder_4", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SAR: {sar_mode} | Steps: {args.steps}")

    ckpt = download_model(repo_id=repo_id, filename=filename, map_location=device)

    model = DBCR(
        image_channels=image_channels,
        condition_channels=condition_channels,
        sar_channels=sar_channels,
        base_channels=base_channels,
        time_dim=time_dim,
        num_heads=num_heads,
        window_size_sf0=window_size_sf0,
        window_size_not_sf0=window_size_not_sf0,
        use_checkpoint=use_checkpoint,
        include_encoder_4=include_encoder_4,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ds = SEN12MSCRDataset(
        split="test",
        include_s1=(sar_mode != "None"),
        include_mask=False,
        total_bands=(image_channels == 13)
    )

    visualize_samples(
        model=model, dataset=ds, device=device,
        sar_mode=sar_mode, n_samples=args.n_samples,
        T=T, steps=args.steps, sigmoid_k=sigmoid_k,
        save_path=args.save_path, seed=args.seed,
    )