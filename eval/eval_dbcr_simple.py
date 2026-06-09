from pathlib import Path
import torch
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import yaml

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dataset import SEN12MSCRDataset

from dbcr_simple import DBCRSimple

from hf_utils import (
    download_model,
    resolve_load_version,
)

def mae(pred, target):
    return torch.abs(pred - target).mean().item()


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return float("inf")

    return (
        10 * torch.log10(max_val ** 2 / mse)
    ).item()


def ssim(pred, target,
         window_size=11,
         C1=0.01**2,
         C2=0.03**2):

    scores = []

    for b in range(pred.shape[1]):

        p = pred[:, b:b+1]
        t = target[:, b:b+1]

        mu_p = torch.nn.functional.avg_pool2d(
            p,
            window_size,
            stride=1,
            padding=window_size//2
        )

        mu_t = torch.nn.functional.avg_pool2d(
            t,
            window_size,
            stride=1,
            padding=window_size//2
        )

        mu_p2 = mu_p**2
        mu_t2 = mu_t**2
        mu_pt = mu_p*mu_t

        sigma_p2 = (
            torch.nn.functional.avg_pool2d(
                p*p,
                window_size,
                stride=1,
                padding=window_size//2
            )
            - mu_p2
        )

        sigma_t2 = (
            torch.nn.functional.avg_pool2d(
                t*t,
                window_size,
                stride=1,
                padding=window_size//2
            )
            - mu_t2
        )

        sigma_pt = (
            torch.nn.functional.avg_pool2d(
                p*t,
                window_size,
                stride=1,
                padding=window_size//2
            )
            - mu_pt
        )

        ssim_map = (
            ((2*mu_pt + C1)*(2*sigma_pt + C2))
            /
            ((mu_p2 + mu_t2 + C1)*(sigma_p2 + sigma_t2 + C2))
        )

        scores.append(ssim_map.mean().item())

    return np.mean(scores)


def sam(pred, target, eps=1e-8):

    dot = (pred * target).sum(dim=1)

    norm_p = pred.norm(dim=1).clamp(min=eps)
    norm_t = target.norm(dim=1).clamp(min=eps)

    cos = (dot / (norm_p * norm_t)).clamp(-1, 1)

    angle = torch.acos(cos)

    return torch.rad2deg(angle).mean().item()

def sigmoid_scheduler(T, sigmoid_k, t, device):
    tau = torch.clamp(t.float() / T, 0.0, 1.0)
    s = torch.sigmoid((tau - 0.5) * sigmoid_k)
    s_min = torch.sigmoid(torch.tensor(-0.5 * sigmoid_k, device=device))
    s_max = torch.sigmoid(torch.tensor(0.5 * sigmoid_k, device=device))
    alpha = torch.clamp((s - s_min) / (s_max - s_min), 0.0, 1.0)
    return alpha[:, None, None, None]

def inference(model, s2_cloudy, device, T=1000, steps=10):
    """
    Inferencia iterativa del diffusion bridge.
    steps: cuántos pasos de refinamiento (10-50 suele ser suficiente)
    """
    x_t = s2_cloudy.clone()  # arrancamos en x_T = cloudy
    
    # Timesteps de T a 1, uniformemente espaciados
    timesteps = torch.linspace(T, 1, steps).long().to(device)
    
    with torch.no_grad():
        for t_val in timesteps:
            t = t_val.repeat(x_t.shape[0])
            pred_clean = model(x_t=x_t, t=t, s2_cloudy=s2_cloudy)
            
            # Interpolamos x_t hacia la predicción según el schedule
            alpha_t = sigmoid_scheduler(T, sigmoid_k=10.0, t=t, device=device)
            alpha_prev = sigmoid_scheduler(T, sigmoid_k=10.0, 
                                           t=(t_val - T//steps).clamp(min=1).repeat(x_t.shape[0]), 
                                           device=device)
            
            # x_{t-1} = mezcla entre pred_clean y s2_cloudy según alpha_prev
            x_t = (1.0 - alpha_prev) * pred_clean + alpha_prev * s2_cloudy
    
    return pred_clean

def evaluate(model, loader, device, T=1000, steps = 10):
    model.eval()

    total_mae = 0
    total_psnr = 0
    total_ssim = 0
    total_sam = 0

    n_batches = 0

    with torch.no_grad():

        for batch in tqdm(loader, desc="Evaluando"):

            cloudy, clean = batch

            cloudy = cloudy.to(device)
            clean = clean.to(device)

            pred = inference(model, cloudy, device, T=T, steps=steps)

            pred = pred.clamp(0, 1)

            total_mae += mae(pred, clean)
            total_psnr += psnr(pred, clean)
            total_ssim += ssim(pred, clean)
            total_sam += sam(pred, clean)

            n_batches += 1
            # break

    metrics = {
        "mae": total_mae / n_batches,
        "psnr": total_psnr / n_batches,
        "ssim": total_ssim / n_batches,
        "sam": total_sam / n_batches,
    }

    return metrics

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--version",
    #     type=int,
    #     default=None
    # )

    # parser.add_argument(
    #     "--split",
    #     type=str,
    #     default="test"
    # )

    # args = parser.parse_args()

    repo_id = "LucioLuque/lama"

    # _, filename = resolve_load_version(
    #     repo_id=repo_id,
    #     filename_prefix="dbcr_no_sar_naf",
    #     requested_version=args.version,
    # )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    filename = "dbcr_no_sar_naf_v1.pth"

    checkpoint = download_model(
        repo_id=repo_id,
        filename=filename,
        map_location=device,
    )

    model = DBCRSimple(
        image_channels=6,
        condition_channels=6,
        base_channels=64,
        time_dim=128,
    )

    model.load_state_dict(checkpoint)

    model = model.float().to(device)

    ds = SEN12MSCRDataset(
        split="test",
        include_s1=False,
        include_mask=False,
    )

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
    )

    steps = 10

    metrics = evaluate(model=model, loader=loader, device=device, T=1000, steps=steps)

    print("\nRESULTADOS")
    print("=" * 40)

    for k, v in metrics.items():
        print(f"{k:5s}: {v:.6f}")

    print("=" * 40)