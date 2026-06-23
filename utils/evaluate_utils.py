import torch
import numpy as np
from tqdm import tqdm

from dataset_utils import unpack_batch

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

        sigma_p2 = torch.nn.functional.avg_pool2d(p*p, window_size, stride=1, padding=window_size//2) - mu_p2
        sigma_t2 = torch.nn.functional.avg_pool2d(t*t, window_size, stride=1, padding=window_size//2) - mu_t2
        sigma_pt = torch.nn.functional.avg_pool2d(p*t, window_size, stride=1, padding=window_size//2) - mu_pt

        ssim_map = ((2*mu_pt + C1) * (2*sigma_pt + C2)) / \
                   ((mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))
        scores.append(ssim_map.mean().item())

    return np.mean(scores)

def sam(pred, target, eps=1e-8):
    """Spectral Angle Mapper — en grados."""
    dot    = (pred * target).sum(dim=1)
    norm_p = pred.norm(dim=1).clamp(min=eps)
    norm_t = target.norm(dim=1).clamp(min=eps)
    cos    = (dot / (norm_p * norm_t)).clamp(-1, 1)
    angle  = torch.acos(cos)
    return torch.rad2deg(angle).mean().item()


def _count_nonfinite(x):
    if x is None:
        return 0
    return int((~torch.isfinite(x)).sum().item())


def evaluate(inference, model, loader, sar_mode, device, T=1000, steps=10, sigmoid_k=10.0):
    model.eval()
    total_mae = total_psnr = total_ssim = total_sam = 0.0
    n_batches = 0
    skipped = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluando", unit="batch"), start=1):
            s2_cloudy, s2_clean, condition, sar = unpack_batch(batch, sar_mode, device)

            tensors_to_check = {"s2_cloudy": s2_cloudy, "s2_clean": s2_clean, "condition": condition}
            if sar is not None:
                tensors_to_check["sar"] = sar

            bad_inputs = {name: _count_nonfinite(t) for name, t in tensors_to_check.items() if _count_nonfinite(t) > 0}
            if bad_inputs:
                skipped += 1
                print(f"[WARN] Batch {batch_idx}: valores no finitos en inputs -> {bad_inputs}. Se omite.")
                continue

            pred = inference(model, s2_cloudy, condition, device, T=T, steps=steps, sar=sar, sigmoid_k=sigmoid_k).clamp(0, 1)

            bad_pred = _count_nonfinite(pred)
            if bad_pred > 0:
                skipped += 1
                print(f"[WARN] Batch {batch_idx}: predicción con {bad_pred} valores no finitos. Se omite.")
                continue

            total_mae  += mae(pred, s2_clean)
            total_psnr += psnr(pred, s2_clean)
            total_ssim += ssim(pred, s2_clean)
            total_sam  += sam(pred, s2_clean)
            n_batches  += 1

    if n_batches == 0:
        raise RuntimeError("No hubo batches válidos para calcular métricas.")

    metrics = {
        "mae":  float(total_mae  / n_batches),
        "psnr": float(total_psnr / n_batches),
        "ssim": float(total_ssim / n_batches),
        "sam":  float(total_sam  / n_batches),
    }

    print(f"\n{'='*40}")
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  PSNR : {metrics['psnr']:.4f} dB")
    print(f"  SSIM : {metrics['ssim']:.6f}")
    print(f"  SAM  : {metrics['sam']:.4f} °")
    print(f"  Batches válidos : {n_batches}")
    print(f"  Batches omitidos: {skipped}")
    print(f"{'='*40}\n")

    return metrics