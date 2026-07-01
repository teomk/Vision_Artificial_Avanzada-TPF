from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"
RANKING_DEFAULT = ROOT / "tools" / "dbcr_complex_v7_ranking" / "outputs" / "dbcr_complex_v7_test_psnr_ranking.json"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dbcr_complex import DBCR
from dbcr_simple import DBCRSimple
from dbcr_utils import inference
from hf_utils import download_model
from evaluate_utils import psnr
from visualize_utils import get_rgb_stats, to_rgb


S1_MEAN = np.array([-8.999908447265625, -14.78221321105957], dtype=np.float32)
S1_STD = np.array([2.413282871246338, 2.3029115200042725], dtype=np.float32)
REPO_ID = "LucioLuque/lama"
T = 1000
STEPS = 10
SIGMOID_K = 10.0

#800 esta bien, 790 tambien

MODELS_CFG = {
    "DBCR-S": {
        "filename": "dbcr_no_sar_naf_v2.pth",
        "sar_mode": "None",
        "type": "simple",
    },
    "DBCR-SC (sin TL)": {
        "filename": "dbcr_concat_v2.pth",
        "sar_mode": "Concat",
        "type": "simple",
    },
    "DBCR": {
        "filename": "dbcr_complex_v7.pth",
        "sar_mode": "ControlNet",
        "type": "complex",
    },
}

COL_LABELS = ["SAR", "Nublada", "DBCR-S", "DBCR-SC (sin TL)", "DBCR", "Original"]


def normalize_s2(s2_raw):
    return np.clip(s2_raw / 10000.0, 0, 1).astype(np.float32)


def normalize_s1(s1_raw):
    mean = S1_MEAN[:, None, None]
    std = S1_STD[:, None, None]
    return ((s1_raw - mean) / (std + 1e-6)).astype(np.float32)


def to_rgb_np(arr_chw, bands=(2, 1, 0)):
    if arr_chw.ndim == 3 and arr_chw.shape[0] >= 3:
        arr = arr_chw[list(bands)].transpose(1, 2, 0)
    else:
        arr = arr_chw
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)
    return (arr * 255).astype(np.uint8)


def to_sar_np(s1_raw, band=0):
    img = s1_raw[band]
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
    return (img * 255).astype(np.uint8)


def load_ranking_entry(ranking_path: Path, rank_number: int) -> dict:
    data = json.loads(ranking_path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    for sample in samples:
        if int(sample.get("rank", -1)) == rank_number:
            return sample
    raise ValueError(f"No se encontró rank={rank_number} en {ranking_path}")


def load_sample(entry: dict):
    paths = entry["paths"]

    with rasterio.open(paths["s1"]) as src:
        s1_raw = src.read().astype(np.float32)

    with rasterio.open(paths["s2_cloudy"]) as src:
        s2_cloudy_raw = src.read(indexes=[2, 3, 4, 8, 12, 13]).astype(np.float32)

    with rasterio.open(paths["s2"]) as src:
        s2_clean_raw = src.read(indexes=[2, 3, 4, 8, 12, 13]).astype(np.float32)

    s1_norm = normalize_s1(s1_raw)
    cloudy_norm = normalize_s2(s2_cloudy_raw)
    clean_norm = normalize_s2(s2_clean_raw)

    return s1_raw, s1_norm, cloudy_norm, clean_norm


def load_models(device: torch.device):
    models = {}

    for name, cfg in MODELS_CFG.items():
        ckpt = download_model(repo_id=REPO_ID, filename=cfg["filename"], map_location=device)

        if cfg["type"] == "simple":
            condition_channels = 8 if cfg["sar_mode"] == "Concat" else 6
            model = DBCRSimple(
                image_channels=6,
                condition_channels=condition_channels,
                base_channels=64,
                time_dim=128,
                control_net=False,
            )
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
        else:
            model = DBCR(
                image_channels=6,
                condition_channels=6,
                sar_channels=2,
                base_channels=64,
                time_dim=128,
                num_heads=1,
                window_size_sf0=8,
                window_size_not_sf0=None,
                use_checkpoint=True,
                include_encoder_4=False,
            )
            model.load_state_dict(ckpt["model_state_dict"], strict=True)

        models[name] = model.float().to(device).eval()

    return models


def run_inference(model, model_type: str, sar_mode: str, cloudy_norm, s1_norm, device):
    cloudy_t = torch.from_numpy(cloudy_norm).unsqueeze(0).float().to(device)
    s1_t = torch.from_numpy(s1_norm).unsqueeze(0).float().to(device)
    if model_type == "simple":
        if sar_mode == "None":
            condition = cloudy_t
            sar = None
        elif sar_mode == "Concat":
            condition = torch.cat([cloudy_t, s1_t], dim=1)
            sar = None
        else:
            condition = cloudy_t
            sar = s1_t
        pred = inference(
            model,
            cloudy_t,
            condition,
            device,
            T=T,
            steps=STEPS,
            sar=sar,
            sigmoid_k=SIGMOID_K,
            show_progress=False,
        )
    else:
        pred = inference(
            model,
            cloudy_t,
            cloudy_t,
            device,
            T=T,
            steps=STEPS,
            sar=s1_t,
            sigmoid_k=SIGMOID_K,
            show_progress=False,
        )
    return pred.squeeze(0).clamp(0, 1).cpu().numpy()


def plot_sample(entry: dict, models: dict, show: bool = True):
    s1_raw, s1_norm, cloudy_norm, clean_norm = load_sample(entry)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cloudy_t = torch.from_numpy(cloudy_norm).float()
    clean_t = torch.from_numpy(clean_norm).float()
    stats = get_rgb_stats(cloudy_t, clean_t)

    preds = []
    psnr_values = []
    for name, cfg in MODELS_CFG.items():
        pred = run_inference(
            model=models[name],
            model_type=cfg["type"],
            sar_mode=cfg["sar_mode"],
            cloudy_norm=cloudy_norm,
            s1_norm=s1_norm,
            device=device,
        )
        pred_t = torch.from_numpy(pred).float()
        preds.append(pred_t)
        psnr_values.append(float(psnr(pred_t.unsqueeze(0), clean_t.unsqueeze(0))))

    cloudy_rgb = to_rgb(cloudy_t, stats=stats)
    clean_rgb = to_rgb(clean_t, stats=stats)
    sar_img = to_sar_np(s1_raw, band=0)

    fig, axes = plt.subplots(1, len(COL_LABELS), figsize=(4 * len(COL_LABELS), 5))
    panels = [
        (sar_img, "SAR"),
        (cloudy_rgb, "Nublada"),
        (to_rgb(preds[0], stats=stats), f"DBCR-S\nPSNR={psnr_values[0]:.2f} dB"),
        (to_rgb(preds[1], stats=stats), f"DBCR-SC (sin TL)\nPSNR={psnr_values[1]:.2f} dB"),
        (to_rgb(preds[2], stats=stats), f"DBCR\nPSNR={psnr_values[2]:.2f} dB"),
        (clean_rgb, "Original"),
    ]

    for ax, (img, title) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize one ranked sample from DBCR model comparison using the v7 ranking.")
    parser.add_argument("--ranking-file", default=str(RANKING_DEFAULT))
    parser.add_argument("--rank", type=int, required=True, help="Rank 1-based to visualize from the JSON ranking file.")
    args = parser.parse_args()

    def resolve_default_save_path(rank_number: int) -> Path:
        default_dir = ROOT / "visualize" / "outputs"
        return default_dir / f"dbcr_complex_v7_rank{rank_number}.png"

    ranking_path = Path(args.ranking_file)
    entry = load_ranking_entry(ranking_path, args.rank)

    print(f"Rank {entry['rank']} | PSNR={entry['psnr']:.4f}")
    print(f"S1        : {entry['paths']['s1']}")
    print(f"S2_CLOUDY : {entry['paths']['s2_cloudy']}")
    print(f"S2_CLEAN  : {entry['paths']['s2']}")
    print(f"MASK      : {entry['paths']['mask']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    fig = plot_sample(entry, models=models, show=True)

    save_answer = input("Guardar figura? [s/n]: ").strip().lower()
    if save_answer in {"s", "si", "sí", "y", "yes"}:
        output_path = resolve_default_save_path(args.rank)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Figura guardada en {output_path}")
    else:
        print("Figura no guardada.")

    plt.close(fig)


if __name__ == "__main__":
    main()