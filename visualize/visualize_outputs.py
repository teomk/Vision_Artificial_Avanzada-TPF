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
from dbcr_simple import DBCRSimple
from dbcr_complex import DBCR
from hf_utils import download_model
from dbcr_utils import inference
from dataset_utils import unpack_batch
from visualize_utils import to_rgb, get_rgb_stats, to_sar

REPO_ID = "LucioLuque/lama"
SEED = 2
N_SAMPLES = 2
STEPS = 1
T = 1000
SIGMOID_K = 10.0

MODELS_CFG = {
    "DBCR-S":          {"filename": "dbcr_no_sar_naf_v2.pth", "sar_mode": "None",   "type": "simple"},
    "DBCR-SC (sin TL)":{"filename": "dbcr_concat_v2.pth", "sar_mode": "Concat", "type": "simple"},
    "DBCR":            {"filename": "dbcr_complex_v7.pth", "sar_mode": "ControlNet", "type": "complex"},
}

COL_LABELS = ["SAR", "Nublada", "DBCR-S", "DBCR-SC (sin TL)", "DBCR", "Original"]


def load_simple(filename, sar_mode, device):
    condition_channels = 8 if sar_mode == "Concat" else 6
    ckpt = download_model(repo_id=REPO_ID, filename=filename, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = DBCRSimple(
        image_channels=6,
        condition_channels=condition_channels,
        base_channels=64,
        time_dim=128,
        control_net=(sar_mode == "ControlNet"),
    )
    model.load_state_dict(state, strict=False)
    return model.float().to(device).eval()

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

def run_inference(model, model_type, sar_mode, cloudy, s1, device):
    cloudy_b = cloudy.unsqueeze(0).float().to(device)

    if model_type == "simple":
        if sar_mode == "None":
            condition = cloudy_b
            sar = None
        elif sar_mode == "Concat":
            s1_b = s1.unsqueeze(0).float().to(device)
            condition = torch.cat([cloudy_b, s1_b], dim=1)
            sar = None
        elif sar_mode == "ControlNet":
            s1_b = s1.unsqueeze(0).float().to(device)
            condition = cloudy_b
            sar = s1_b

        pred = inference(model, cloudy_b, condition, device,
                         T=T, steps=STEPS, sar=sar, sigmoid_k=SIGMOID_K,
                         show_progress=False)

    else:
        fake_batch = (s1.unsqueeze(0).float(), cloudy_b, cloudy_b)
        s2_cloudy, _, condition, sar = unpack_batch(fake_batch, sar_mode, device)
        pred = inference(model, s2_cloudy, condition, device,
                         T=T, steps=STEPS, sar=sar, sigmoid_k=SIGMOID_K)

    return pred.squeeze(0).clamp(0, 1).cpu()

def make_figure(models, dataset, device):
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), N_SAMPLES, replace=False)

    n_cols = len(COL_LABELS)
    fig = plt.figure(figsize=(4 * n_cols, 4 * N_SAMPLES))
    gs = gridspec.GridSpec(N_SAMPLES, n_cols, figure=fig, hspace=0.08, wspace=0.05)

    with torch.no_grad():
        for row, idx in enumerate(tqdm(indices, desc="Muestras")):
            triple = dataset.triples[idx]
            key = triple["key"]
            print(f"\n[idx={idx}] roi={key[0]} season={key[1]} num={key[2]} patch={key[3]}")
            print(f"  cloudy : {triple['cloudy'].name}")
            print(f"  clear  : {triple['s2'].name}")

            s1, cloudy, clear = dataset[idx]
            stats = get_rgb_stats(cloudy)

            preds = []

            for name, cfg in MODELS_CFG.items():
                pred = run_inference(
                    model=models[name],
                    model_type=cfg["type"],
                    sar_mode=cfg["sar_mode"],
                    cloudy=cloudy,
                    s1=s1,
                    device=device,
                )


                preds.append(pred)

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
                    ax.set_title(COL_LABELS[col], fontsize=10, pad=6)

    plt.savefig(f"imgs/comparacion_modelos_2.png", bbox_inches="tight", dpi=150)
    print("Figura guardada en imgs/comparacion_modelos_2.png")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Cargando modelos...")
    models = {
        "DBCR-S": load_simple("dbcr_no_sar_naf_v2.pth", sar_mode="None", device=device),
        "DBCR-SC (sin TL)": load_simple("dbcr_concat_v2.pth", sar_mode="Concat", device=device),
        "DBCR": load_complex("dbcr_complex_v7.pth", device=device),
    }

    ds = SEN12MSCRDataset(split="test", include_s1=True, include_mask=False)
    make_figure(models, ds, device)