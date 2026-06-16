from pathlib import Path
import torch
import sys
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import date

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
from dataset_utils import unpack_batch
from metrics import mae, psnr, ssim, sam

import torch
from tqdm import tqdm


# ─────────────────────────────────────────────
# Scheduler (igual que simple, reusable)
# ─────────────────────────────────────────────
def sigmoid_scheduler(T, sigmoid_k, t, device):
    tau = torch.clamp(t.float() / T, 0.0, 1.0)
    s = torch.sigmoid((tau - 0.5) * sigmoid_k)

    s_min = torch.sigmoid(torch.tensor(-0.5 * sigmoid_k, device=device))
    s_max = torch.sigmoid(torch.tensor(0.5 * sigmoid_k, device=device))

    alpha = torch.clamp((s - s_min) / (s_max - s_min), 0.0, 1.0)
    return alpha[:, None, None, None]


# ─────────────────────────────────────────────
# Bridge (training diffusion-like interpolation)
# ─────────────────────────────────────────────
def make_bridge_sample(s2_clean, s2_cloudy, t, T, sigmoid_k, device):
    alpha_t = sigmoid_scheduler(T, sigmoid_k, t, device)
    return (1.0 - alpha_t) * s2_clean + alpha_t * s2_cloudy


# ─────────────────────────────────────────────
# INFERENCE (iterative refinement)
# ─────────────────────────────────────────────
def inference(
    model,
    cloudy_b,
    condition,
    device,
    T=1000,
    steps=10,
    sar=None,
    sigmoid_k=10.0,
    show_progress: bool = False
):
    """
    DBCR complex inference loop (iterative refinement)
    """

    model.eval()

    x_t = cloudy_b.clone()
    timesteps = torch.linspace(T, 1, steps).long().to(device)

    iterator = tqdm(timesteps, desc="Infer steps", unit="step") if show_progress else timesteps

    with torch.no_grad():
        for t_val in iterator:

            B = x_t.shape[0]
            t = t_val.repeat(B).to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred_clean = model(
                    x_t=x_t,
                    t=t,
                    s2_cloudy=condition,
                    sar=sar
                )

            prev_t = (t_val - (T // steps)).clamp(min=1)
            prev_t = prev_t.repeat(B).to(device)

            x_t = make_bridge_sample(
                s2_clean=pred_clean.float(),
                s2_cloudy=cloudy_b,
                t=prev_t,
                T=T,
                sigmoid_k=sigmoid_k,
                device=device
            ).clamp(0, 1)

    return pred_clean.float()


# ─────────────────────────────────────────────
def _count_nonfinite(x):
    if x is None:
        return 0
    return int((~torch.isfinite(x)).sum().item())


# ─────────────────────────────────────────────
def evaluate(model, loader, sar_mode, device, T=1000, steps=10, sigmoid_k=10.0):
    model.eval()

    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_sam  = 0.0

    n_batches = 0
    skipped = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluando"), start=1):

            s2_cloudy, s2_clean, condition, sar = unpack_batch(batch, sar_mode, device)

            # sanity check
            if any(_count_nonfinite(x) > 0 for x in [s2_cloudy, s2_clean, condition]):
                skipped += 1
                print(f"[WARN] batch {i} inputs inválidos")
                continue

            # forward inference (IMPORTANTE: DBCR complex usa SAR + bridge)
            pred = inference(
                model,
                s2_cloudy,
                condition,
                device,
                T=T,
                steps=steps,
                sar=sar,
                sigmoid_k=sigmoid_k
            ).clamp(0, 1)

            if _count_nonfinite(pred) > 0:
                skipped += 1
                print(f"[WARN] batch {i} pred inválida")
                continue

            total_mae  += mae(pred, s2_clean)
            total_psnr += psnr(pred, s2_clean)
            total_ssim += ssim(pred, s2_clean)
            total_sam  += sam(pred, s2_clean)

            n_batches += 1

    if n_batches == 0:
        raise RuntimeError("No hay batches válidos")

    metrics = {
        "mae": float(total_mae / n_batches),
        "psnr": float(total_psnr / n_batches),
        "ssim": float(total_ssim / n_batches),
        "sam": float(total_sam / n_batches),
    }

    print("\n===== RESULTADOS DBCR COMPLEX =====")
    for k, v in metrics.items():
        print(f"{k.upper():5}: {v:.6f}")
    print(f"Batches válidos: {n_batches}")
    print(f"Omitidos       : {skipped}")
    print("===================================\n")

    return metrics


# ─────────────────────────────────────────────
def register_eval(filename, metrics, split, sar_mode, steps, yaml_path="eval/results.yaml"):
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {}
    data.setdefault("models", {})

    key = f"dbcr_complex_{sar_mode.lower()}"
    data["models"].setdefault(key, {})

    entry_key = filename.replace(".pth", "")

    data["models"][key][entry_key] = {
        "filename": filename,
        "eval": {
            "split": split,
            "date": str(date.today()),
            "steps": steps,
            "mae": round(metrics["mae"], 6),
            "psnr": round(metrics["psnr"], 4),
            "ssim": round(metrics["ssim"], 6),
            "sam": round(metrics["sam"], 4),
        }
    }

    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Guardado en {yaml_path}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    # python eval_dbcr_complex.py --config configs/dbcr_complex.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sar_mode = cfg["sar_mode"]

    repo_id = cfg["huggingface"]["repo_id"]
    filename = cfg["huggingface"]["save_filename"]

    T = cfg["train"]["T"]
    sigmoid_k = cfg["train"].get("sigmoid_k", 10.0)
    batch_size = cfg["train"]["batch_size"]
    image_channels = int(cfg["model_args"]["image_channels"])
    condition_channels = int(cfg["model_args"]["condition_channels"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device} SAR={sar_mode} split={args.split}")

    # ── load checkpoint
    ckpt = download_model(repo_id=repo_id, filename=filename, map_location=device)

    # ── model
    model = DBCR(
        image_channels=image_channels,
        condition_channels=condition_channels,
        sar_channels=2,
        base_channels=64,
        time_dim=128,
        num_heads=1,
        window_size_sf0=8,
        window_size_not_sf0=None,
        use_checkpoint=True,
        include_encoder_4=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # ── dataset
    ds = SEN12MSCRDataset(
        split=args.split,
        include_s1=(sar_mode != "None"),
        include_mask=False,
        total_bands=(image_channels == 13)
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # ── eval
    metrics = evaluate(
        model=model,
        loader=loader,
        sar_mode=sar_mode,
        device=device,
        T=T,
        steps=args.steps,
        sigmoid_k=sigmoid_k
    )

    register_eval(
        filename=filename,
        metrics=metrics,
        split=args.split,
        sar_mode=sar_mode,
        steps=args.steps,
    )