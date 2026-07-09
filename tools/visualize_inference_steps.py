import argparse
import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"
RANKING_DEFAULT = ROOT / "tools" / "dbcr_complex_v7_ranking" / "outputs" / "dbcr_complex_v7_test_psnr_ranking.json"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dataset import SEN12MSCRDataset
from dbcr_complex import DBCR
from dbcr_utils import inference
from evaluate_utils import psnr
from hf_utils import download_model
from visualize_utils import get_rgb_stats, to_rgb, to_sar

REPO_ID = "LucioLuque/lama"
FILENAME = "dbcr_complex_v7.pth"
SEED = 0
N_SAMPLES = 2
T = 1000
SIGMOID_K = 10.0
STEPS_LIST = [1, 5, 10, 15, 20]

COL_LABELS = ["SAR", "Nublada"] + [f"{s} steps" for s in STEPS_LIST] + ["Original"]

def normalize_s2(s2_raw):
    return np.clip(s2_raw / 10000.0, 0, 1).astype(np.float32)

def normalize_s1(s1_raw):
    mean = np.array([-8.999908447265625, -14.78221321105957], dtype=np.float32)[:, None, None]
    std = np.array([2.413282871246338, 2.3029115200042725], dtype=np.float32)[:, None, None]
    return ((s1_raw - mean) / (std + 1e-6)).astype(np.float32)

def load_complex(filename, device):
    ckpt = download_model(repo_id=REPO_ID, filename=filename, map_location=device)
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
    return model.float().to(device).eval()

def run_inference(model, cloudy, s1, device, steps):
    cloudy_b = cloudy.unsqueeze(0).float().to(device)
    s1_b = s1.unsqueeze(0).float().to(device)
    pred = inference(
        model,
        cloudy_b,
        cloudy_b,
        device,
        T=T,
        steps=steps,
        sar=s1_b,
        sigmoid_k=SIGMOID_K,
    )
    return pred.squeeze(0).clamp(0, 1).cpu()

def load_ranking_entry(ranking_path: Path, rank_number: int) -> dict:
    data = json.loads(ranking_path.read_text(encoding="utf-8"))
    for sample in data.get("samples", []):
        if int(sample.get("rank", -1)) == rank_number:
            return sample
    raise ValueError(f"No se encontró rank={rank_number} en {ranking_path}")

def load_sample_from_paths(paths: dict):
    with rasterio.open(paths["s1"]) as src:
        s1_raw = src.read().astype(np.float32)
    with rasterio.open(paths["s2_cloudy"]) as src:
        cloudy_raw = src.read(indexes=[2, 3, 4, 8, 12, 13]).astype(np.float32)
    with rasterio.open(paths["s2"]) as src:
        clear_raw = src.read(indexes=[2, 3, 4, 8, 12, 13]).astype(np.float32)

    s1 = torch.from_numpy(normalize_s1(s1_raw))
    cloudy = torch.from_numpy(normalize_s2(cloudy_raw))
    clear = torch.from_numpy(normalize_s2(clear_raw))
    return s1, cloudy, clear

def make_figure(model, dataset, device, rank=None, ranking_file=None):
    np.random.seed(SEED)

    if rank is not None:
        ranking_path = Path(ranking_file or RANKING_DEFAULT)
        entry = load_ranking_entry(ranking_path, rank)
        samples = [(rank, entry)]
        n_rows = 1
    else:
        indices = np.random.choice(len(dataset), N_SAMPLES, replace=False)
        samples = [(idx, None) for idx in indices]
        n_rows = N_SAMPLES

    n_cols = len(COL_LABELS)
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.08, wspace=0.05)

    with torch.no_grad():
        for row, (idx_or_rank, entry) in enumerate(tqdm(samples, desc="Muestras")):
            if entry is not None:
                s1, cloudy, clear = load_sample_from_paths(entry["paths"])
                print(f"\n[rank={idx_or_rank}] {entry.get('key', '')}")
                print(f"  cloudy : {Path(entry['paths']['s2_cloudy']).name}")
                print(f"  clear  : {Path(entry['paths']['s2']).name}")
            else:
                triple = dataset.triples[idx_or_rank]
                key = triple["key"]
                print(f"\n[idx={idx_or_rank}] roi={key[0]} season={key[1]} num={key[2]} patch={key[3]}")
                s1, cloudy, clear = dataset[idx_or_rank]

            stats = get_rgb_stats(cloudy, clear)

            preds = []
            psnr_values = []
            for steps in tqdm(STEPS_LIST, desc=f"  Inferencia {idx_or_rank}", leave=False):
                pred = run_inference(model, cloudy, s1, device, steps=steps)
                psnr_val = psnr(pred.unsqueeze(0), clear.unsqueeze(0))
                preds.append(pred)
                psnr_values.append(psnr_val)

            sar_img = to_sar(s1, band=0)
            cols = [sar_img, cloudy] + preds + [clear]

            for col, img in enumerate(cols):
                ax = fig.add_subplot(gs[row, col])

                if col == 0:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(to_rgb(img, stats=stats))
                ax.axis("off")

                if row == 0:
                    ax.set_title(COL_LABELS[col], fontsize=16, pad=6)

                pred_col_start = 2
                pred_col_end = 2 + len(STEPS_LIST) - 1
                if pred_col_start <= col <= pred_col_end:
                    psnr_val = psnr_values[col - pred_col_start]
                    ax.text(
                        0.5,
                        0.04,
                        f"PSNR: {psnr_val:.2f} dB",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=16,
                        color="white",
                        bbox=dict(
                            facecolor="black",
                            alpha=0.65,
                            edgecolor="none",
                            boxstyle="round,pad=0.25",
                        ),
                    )

    plt.savefig(f"comparacion_steps_{SEED}.png", bbox_inches="tight", dpi=150)
    print(f"Figura guardada en comparacion_steps_{SEED}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara distintos steps.")
    parser.add_argument("--rank", type=int, default=None, help="Rank 1-based del ranking v7 a visualizar.")
    parser.add_argument("--ranking-file", default=str(RANKING_DEFAULT), help="Archivo JSON del ranking v7.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Cargando modelo...")
    model = load_complex(FILENAME, device)

    ds = SEN12MSCRDataset(split="test", include_s1=True, include_mask=False)
    make_figure(model, ds, device, rank=args.rank, ranking_file=args.ranking_file)