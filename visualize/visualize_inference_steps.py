import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import sys
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR  = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dataset import SEN12MSCRDataset
from dbcr_complex import DBCR
from hf_utils import download_model
from dbcr_utils import inference
from dataset_utils import unpack_batch
from visualize_utils import to_rgb, get_rgb_stats, to_sar
from evaluate_utils import psnr

# ── Config hardcodeada ────────────────────────────────────────────────
REPO_ID   = "LucioLuque/lama"
FILENAME  = "dbcr_complex_v4.pth"
SAR_MODE  = "ControlNet"
SEED      = 2
N_SAMPLES = 2
T         = 1000
SIGMOID_K = 10.0

# STEPS_LIST = [1, 5, 10, 15, 20]
STEPS_LIST = [1, 1, 1, 1, 1]

COL_LABELS = ["SAR", "Nublada"] + [f"{s} steps" for s in STEPS_LIST] + ["Original"]

# ── Carga de modelo ───────────────────────────────────────────────────

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
    fake_batch = (s1.unsqueeze(0).float(), cloudy_b, cloudy_b)
    s2_cloudy, _, condition, sar = unpack_batch(fake_batch, SAR_MODE, device)
    pred = inference(model, s2_cloudy, condition, device,
                     T=T, steps=steps, sar=sar, sigmoid_k=SIGMOID_K)
    return pred.squeeze(0).clamp(0, 1).cpu()

def make_figure(model, dataset, device):
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), N_SAMPLES, replace=False)

    n_cols = len(COL_LABELS)
    fig = plt.figure(figsize=(4 * n_cols, 4 * N_SAMPLES))
    gs  = gridspec.GridSpec(N_SAMPLES, n_cols, figure=fig, hspace=0.08, wspace=0.05)

    with torch.no_grad():
        for row, idx in enumerate(tqdm(indices, desc="Muestras")):
            triple = dataset.triples[idx]
            key = triple["key"]
            print(f"\n[idx={idx}] roi={key[0]} season={key[1]} num={key[2]} patch={key[3]}")

            s1, cloudy, clear = dataset[idx]
            stats = get_rgb_stats(cloudy, clear)

            preds = []
            psnr_values = []
            for steps in tqdm(STEPS_LIST, desc=f"  Inferencia idx={idx}", leave=False):
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

                # PSNR sobre las predicciones
                pred_col_start = 2
                pred_col_end   = 2 + len(STEPS_LIST) - 1
                if pred_col_start <= col <= pred_col_end:
                    psnr_val = psnr_values[col - pred_col_start]
                    ax.text(
                        0.5, 0.04,
                        f"PSNR: {psnr_val:.2f} dB",
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=16, color="white",
                        bbox=dict(facecolor="black", alpha=0.65,
                                  edgecolor="none", boxstyle="round,pad=0.25")
                    )

                # if col == 0:
                #     ax.text(
                #         -0.08, 0.5, f"idx={idx}",
                #         transform=ax.transAxes,
                #         ha="right", va="center",
                #         fontsize=9, rotation=90
                #     )

    plt.savefig(f"comparacion_steps_{SEED}.png", bbox_inches="tight", dpi=150)
    print(f"Figura guardada en comparacion_steps_{SEED}.png")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Cargando modelo...")
    model = load_complex(FILENAME, device)

    ds = SEN12MSCRDataset(split="test", include_s1=True, include_mask=False)
    make_figure(model, ds, device)