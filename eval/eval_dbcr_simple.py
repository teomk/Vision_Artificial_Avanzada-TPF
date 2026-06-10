from pathlib import Path
import torch
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
import argparse
from datetime import date

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


# ── Métricas ───────────────────────────────────────────────────────────

def mae(pred, target):
    return torch.abs(pred - target).mean().item()

def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(max_val ** 2 / mse)).item()

def ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    scores = []
    for b in range(pred.shape[1]):
        p = pred[:, b:b+1]
        t = target[:, b:b+1]
        mu_p = torch.nn.functional.avg_pool2d(p, window_size, stride=1, padding=window_size//2)
        mu_t = torch.nn.functional.avg_pool2d(t, window_size, stride=1, padding=window_size//2)
        mu_p2 = mu_p**2; mu_t2 = mu_t**2; mu_pt = mu_p*mu_t
        sigma_p2 = torch.nn.functional.avg_pool2d(p*p, window_size, stride=1, padding=window_size//2) - mu_p2
        sigma_t2 = torch.nn.functional.avg_pool2d(t*t, window_size, stride=1, padding=window_size//2) - mu_t2
        sigma_pt = torch.nn.functional.avg_pool2d(p*t, window_size, stride=1, padding=window_size//2) - mu_pt
        ssim_map = ((2*mu_pt + C1)*(2*sigma_pt + C2)) / ((mu_p2 + mu_t2 + C1)*(sigma_p2 + sigma_t2 + C2))
        scores.append(ssim_map.mean().item())
    return np.mean(scores)

def sam(pred, target, eps=1e-8):
    dot = (pred * target).sum(dim=1)
    norm_p = pred.norm(dim=1).clamp(min=eps)
    norm_t = target.norm(dim=1).clamp(min=eps)
    cos = (dot / (norm_p * norm_t)).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos)).mean().item()


# ── Bridge inference ────────────────────────────────────────────────────

def sigmoid_scheduler(T, sigmoid_k, t, device):
    tau = torch.clamp(t.float() / T, 0.0, 1.0)
    s = torch.sigmoid((tau - 0.5) * sigmoid_k)
    s_min = torch.sigmoid(torch.tensor(-0.5 * sigmoid_k, device=device))
    s_max = torch.sigmoid(torch.tensor(0.5 * sigmoid_k, device=device))
    return torch.clamp((s - s_min) / (s_max - s_min), 0.0, 1.0)[:, None, None, None]

def inference(model, cloudy_b, condition, device, T=1000, steps=10):
    x_t = cloudy_b.clone()
    timesteps = torch.linspace(T, 1, steps).long().to(device)
    with torch.no_grad():
        for t_val in timesteps:
            t = t_val.repeat(x_t.shape[0])
            pred_clean = model(x_t=x_t, t=t, s2_cloudy=condition)
            alpha_prev = sigmoid_scheduler(T, 10.0, (t_val - T//steps).clamp(min=1).repeat(x_t.shape[0]), device)
            x_t = (1.0 - alpha_prev) * pred_clean + alpha_prev * cloudy_b
    return pred_clean


# ── Evaluación ─────────────────────────────────────────────────────────

def evaluate(model, loader, use_sar, device, T=1000, steps=10):
    model.eval()
    total_mae = total_psnr = total_ssim = total_sam = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluando", unit="batch"):
            if use_sar:
                cloudy, s1, clean = batch
                cloudy = cloudy.to(device); s1 = s1.to(device); clean = clean.to(device)
                condition = torch.cat([cloudy, s1], dim=1)
            else:
                cloudy, clean = batch
                cloudy = cloudy.to(device); clean = clean.to(device)
                condition = cloudy

            pred = inference(model, cloudy, condition, device, T=T, steps=steps).clamp(0, 1)
            total_mae  += mae(pred, clean)
            total_psnr += psnr(pred, clean)
            total_ssim += ssim(pred, clean)
            total_sam  += sam(pred, clean)
            n_batches  += 1

    metrics = {"mae": float(total_mae/n_batches), "psnr": float(total_psnr/n_batches), "ssim": float(total_ssim/n_batches), "sam": float(total_sam/n_batches)}

    print(f"\n{'='*40}")
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  PSNR : {metrics['psnr']:.4f} dB")
    print(f"  SSIM : {metrics['ssim']:.6f}")
    print(f"  SAM  : {metrics['sam']:.4f} °")
    print(f"{'='*40}\n")

    return metrics


# ── Registro de resultados ──────────────────────────────────────────────

def register_eval(filename, *, metrics, split, use_sar, steps, yaml_path="eval/results.yaml"):
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {}
    if "models" not in data:
        data["models"] = {}

    mkey = "dbcr_sar" if use_sar else "dbcr_no_sar"
    if mkey not in data["models"]:
        data["models"][mkey] = {}

    target_vkey = None
    for vkey, entry in data["models"][mkey].items():
        if entry.get("filename") == filename:
            target_vkey = vkey
            break
    if target_vkey is None:
        target_vkey = filename.replace(".pth", "")
        data["models"][mkey][target_vkey] = {"filename": filename}

    data["models"][mkey][target_vkey]["eval"] = {"split": split, "date": str(date.today()), "steps": steps, "mae": round(metrics["mae"], 6), "psnr": round(metrics["psnr"], 4), "ssim": round(metrics["ssim"], 6), "sam": round(metrics["sam"], 4)}

    yaml_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))
    print(f"Métricas guardadas en {yaml_path} (models.{mkey}.{target_vkey}.eval)")


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # python eval/eval_dbcr_simple.py --config configs/dbcr_no_sar.yaml
    # python eval/eval_dbcr_simple.py --config configs/dbcr_sar.yaml --split test --steps 10
    parser = argparse.ArgumentParser(description="Evaluar DBCR (SAR o No-SAR)")
    parser.add_argument("--config", type=str, required=True, help="Ruta al config YAML")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Split a evaluar (default: test)")
    parser.add_argument("--steps", type=int, default=10, help="Pasos de inferencia iterativa (default: 10)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    use_sar       = cfg["sar"]
    repo_id       = cfg["huggingface"]["repo_id"]
    save_filename = cfg["huggingface"]["save_filename"]
    T             = cfg["train"]["T"]
    batch_size    = cfg["train"]["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SAR: {use_sar} | split: {args.split} | steps: {args.steps}")

    checkpoint = download_model(repo_id=repo_id, filename=save_filename, map_location=device)

    image_channels     = 6
    condition_channels = 8 if use_sar else 6

    model = DBCRSimple(image_channels=image_channels, condition_channels=condition_channels, base_channels=64, time_dim=128)
    model.load_state_dict(checkpoint)
    model = model.float().to(device)

    ds = SEN12MSCRDataset(split=args.split, include_s1=use_sar, include_mask=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=cfg["train"]["num_workers"])

    metrics = evaluate(model=model, loader=loader, use_sar=use_sar, device=device, T=T, steps=args.steps)

    register_eval(filename=save_filename, metrics=metrics, split=args.split, use_sar=use_sar, steps=args.steps, yaml_path="eval/results.yaml")