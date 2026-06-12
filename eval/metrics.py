import torch
import numpy as np

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
