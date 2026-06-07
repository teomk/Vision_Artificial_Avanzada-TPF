from pathlib import Path
import torch
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
import argparse

ROOT = Path(__file__).resolve().parent.parent
LAMA_DIR = ROOT / "external" / "lama"
sys.path.append(str(LAMA_DIR))

DATA_DIR = ROOT / "dataset"
sys.path.append(str(DATA_DIR))

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from dataset_lama import SEN12MSCRDataset

# ── Métricas ───────────────────────────────────────────────────────────

def mae(pred, target):
    return torch.abs(pred - target).mean().item()

def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(max_val ** 2 / mse)).item()

def ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    """SSIM por banda, promediado."""
    scores = []
    for b in range(pred.shape[1]):
        p = pred[:, b:b+1]
        t = target[:, b:b+1]

        mu_p = torch.nn.functional.avg_pool2d(p, window_size, stride=1, padding=window_size//2)
        mu_t = torch.nn.functional.avg_pool2d(t, window_size, stride=1, padding=window_size//2)

        mu_p2  = mu_p ** 2
        mu_t2  = mu_t ** 2
        mu_pt  = mu_p * mu_t

        sigma_p2  = torch.nn.functional.avg_pool2d(p*p, window_size, stride=1, padding=window_size//2) - mu_p2
        sigma_t2  = torch.nn.functional.avg_pool2d(t*t, window_size, stride=1, padding=window_size//2) - mu_t2
        sigma_pt  = torch.nn.functional.avg_pool2d(p*t, window_size, stride=1, padding=window_size//2) - mu_pt

        ssim_map = ((2*mu_pt + C1) * (2*sigma_pt + C2)) / \
                   ((mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))
        scores.append(ssim_map.mean().item())

    return np.mean(scores)

def sam(pred, target, eps=1e-8):
    """Spectral Angle Mapper — en grados."""
    dot    = (pred * target).sum(dim=1)                          # [B, H, W]
    norm_p = pred.norm(dim=1).clamp(min=eps)
    norm_t = target.norm(dim=1).clamp(min=eps)
    cos    = (dot / (norm_p * norm_t)).clamp(-1, 1)
    angle  = torch.acos(cos)                                     # radianes
    return torch.rad2deg(angle).mean().item()

def build_model(use_sar: bool) -> FFCResNetGenerator:
    return FFCResNetGenerator(
        input_nc  = 9 if use_sar else 7,
        output_nc = 6,
        ngf       = 64,
        n_downsampling = 3,
        n_blocks   = 18,
        init_conv_kwargs      ={"ratio_gin": 0,    "ratio_gout": 0},
        downsample_conv_kwargs={"ratio_gin": 0,    "ratio_gout": 0},
        resnet_conv_kwargs    ={"ratio_gin": 0.75, "ratio_gout": 0.75, "enable_lfu": False},
    )

def prepare_batch(batch, use_sar, device):
    if use_sar:
        s1_b, cloudy_b, mask_b, clear_b = batch
        s1_b     = s1_b.to(device)
        cloudy_b = cloudy_b.to(device)
        mask_b   = mask_b.to(device)
        clear_b  = clear_b.to(device)
        x = torch.cat([cloudy_b * (1 - mask_b), mask_b, s1_b], dim=1)  # [B, 9, H, W]
    else:
        cloudy_b, mask_b, clear_b = batch
        cloudy_b = cloudy_b.to(device)
        mask_b   = mask_b.to(device)
        clear_b  = clear_b.to(device)
        x = torch.cat([cloudy_b * (1 - mask_b), mask_b], dim=1)        # [B, 7, H, W]

    return x, clear_b

# ── Evaluación ─────────────────────────────────────────────────────────

def evaluate(model, loader, use_sar, device):
    model.eval()

    total_mae  = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_sam  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluando", unit="batch"):
            x, clear_b = prepare_batch(batch, use_sar, device=device)

            output = model(x).clamp(0, 1)

            total_mae  += mae(output, clear_b)
            total_psnr += psnr(output, clear_b)
            total_ssim += ssim(output, clear_b)
            total_sam  += sam(output, clear_b)
            n_batches  += 1
            #un solo batch para ser mas rapido
            break

    print(f"\n{'='*40}")
    print(f"  MAE  : {total_mae  / n_batches:.6f}")
    print(f"  PSNR : {total_psnr / n_batches:.4f} dB")
    print(f"  SSIM : {total_ssim / n_batches:.6f}")
    print(f"  SAM  : {total_sam  / n_batches:.4f} °")
    print(f"{'='*40}\n")

    return {
        "mae":  total_mae  / n_batches,
        "psnr": total_psnr / n_batches,
        "ssim": total_ssim / n_batches,
        "sam":  total_sam  / n_batches,
    }


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # python eval/eval_lama.py --config configs/eval_lama_no_sar.yaml
    # python eval/eval_lama.py --config configs/eval_lama_sar.yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_eval = cfg["eval"]

    use_sar    = cfg.get("sar", False)
    model_path = ROOT / cfg_eval["model_path"]
    batch_size = cfg_eval.get("batch_size", 4)
    num_workers= cfg_eval.get("num_workers", 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {device} | SAR: {use_sar}")

    model = build_model(use_sar=use_sar)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model = model.float().to(device)

    ds_test = SEN12MSCRDataset(split="test", include_s1=use_sar)
    loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    metrics = evaluate(model, loader, use_sar=use_sar, device=device)