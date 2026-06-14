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
from dataset_utils import unpack_batch
from dbcr_simple_utils import inference
from metrics import mae, psnr, ssim, sam

# ── Evaluación ─────────────────────────────────────────────────────────

def _count_nonfinite(tensor):
    if tensor is None:
        return 0
    return int((~torch.isfinite(tensor)).sum().item())

def evaluate(model, loader, sar_mode, device, T=1000, steps=10, sigmoid_k=10.0):
    model.eval()
    total_mae = total_psnr = total_ssim = total_sam = 0.0
    n_batches = 0
    skipped_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluando", unit="batch"), start=1):
            s2_cloudy, s2_clean, condition, sar = unpack_batch(batch, sar_mode, device)

            bad_inputs = {
                "s2_cloudy": _count_nonfinite(s2_cloudy),
                "s2_clean": _count_nonfinite(s2_clean),
                "condition": _count_nonfinite(condition),
                "sar": _count_nonfinite(sar),
            }
            bad_inputs = {name: count for name, count in bad_inputs.items() if count > 0}
            if bad_inputs:
                skipped_batches += 1
                print(f"[WARN] Batch {batch_idx}: valores no finitos en inputs -> {bad_inputs}. Se omite.")
                continue

            pred = inference(model, s2_cloudy, condition, device, T=T, steps=steps, sar=sar, sigmoid_k=sigmoid_k).clamp(0, 1)

            bad_pred = _count_nonfinite(pred)
            if bad_pred > 0:
                skipped_batches += 1
                print(f"[WARN] Batch {batch_idx}: predicción con {bad_pred} valores no finitos. Se omite.")
                continue

            total_mae  += mae(pred, s2_clean)
            total_psnr += psnr(pred, s2_clean)
            total_ssim += ssim(pred, s2_clean)
            total_sam  += sam(pred, s2_clean)
            n_batches  += 1

    if n_batches == 0:
        raise RuntimeError("No hubo batches válidos para calcular métricas.")

    metrics = {"mae": float(total_mae/n_batches), "psnr": float(total_psnr/n_batches), "ssim": float(total_ssim/n_batches), "sam": float(total_sam/n_batches)}

    print(f"\n{'='*40}")
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  PSNR : {metrics['psnr']:.4f} dB")
    print(f"  SSIM : {metrics['ssim']:.6f}")
    print(f"  SAM  : {metrics['sam']:.4f} °")
    print(f"  Batches válidos: {n_batches}")
    print(f"  Batches omitidos: {skipped_batches}")
    print(f"{'='*40}\n")

    return metrics


# ── Registro de resultados ──────────────────────────────────────────────

def register_eval(filename, *, metrics, split, sar_mode, steps, yaml_path="eval/results.yaml"):
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {}
    if "models" not in data:
        data["models"] = {}

    mkey = f"dbcr_{sar_mode.lower()}"   # "dbcr_none" | "dbcr_concat" | "dbcr_controlnet"
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
 
    data["models"][mkey][target_vkey]["eval"] = {
        "split": split,
        "date":  str(date.today()),
        "steps": steps,
        "mae":   round(metrics["mae"],  6),
        "psnr":  round(metrics["psnr"], 4),
        "ssim":  round(metrics["ssim"], 6),
        "sam":   round(metrics["sam"],  4),
    }
 
    yaml_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))
    print(f"Métricas guardadas en {yaml_path} (models.{mkey}.{target_vkey}.eval)")


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # python eval/eval_dbcr_simple.py --config configs/dbcr_no_sar.yaml
    # python eval/eval_dbcr_simple.py --config configs/dbcr_sar.yaml --split test --steps 10
    # python eval/eval_dbcr_simple.py --config configs/dbcr_controlnet.yaml
    parser = argparse.ArgumentParser(description="Evaluar DBCR (SAR o No-SAR)")
    parser.add_argument("--config", type=str, required=True, help="Ruta al config YAML")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Split a evaluar (default: test)")
    parser.add_argument("--steps", type=int, default=10, help="Pasos de inferencia iterativa (default: 10)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sar_mode = cfg["sar_mode"]  # "None" | "Concat" | "ControlNet"
    repo_id       = cfg["huggingface"]["repo_id"]
    save_filename = cfg["huggingface"]["save_filename"]
    T             = cfg["train"]["T"]
    sigmoid_k = cfg["train"].get("sigmoid_k", 10.0)
    batch_size    = cfg["train"]["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SAR: {sar_mode} | split: {args.split} | steps: {args.steps}")

    checkpoint = download_model(repo_id=repo_id, filename=save_filename, map_location=device)

    image_channels     = 6
    condition_channels = 8 if sar_mode == "Concat" else 6

    model = DBCRSimple(image_channels=image_channels, condition_channels=condition_channels, base_channels=64, time_dim=128, control_net=(sar_mode == "ControlNet"))
    if sar_mode != "ControlNet":
        checkpoint = {
            k: v for k, v in checkpoint.items()
            if not k.startswith("control_net.")
        }
    model.load_state_dict(checkpoint, strict=True)
    model = model.float().to(device)

    

    ds = SEN12MSCRDataset(split=args.split, include_s1=(sar_mode != "None"), include_mask=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=cfg["train"]["num_workers"])

    metrics = evaluate(model=model, loader=loader, sar_mode=sar_mode, device=device, T=T, steps=args.steps, sigmoid_k=sigmoid_k)

    register_eval(filename=save_filename, metrics=metrics, split=args.split, sar_mode=sar_mode, steps=args.steps, yaml_path="eval/results.yaml")