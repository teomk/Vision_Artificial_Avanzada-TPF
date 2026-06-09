# eval/eval_lama.py
from pathlib import Path
import torch
import sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
import argparse
from datetime import date

ROOT     = Path(__file__).resolve().parent.parent
LAMA_DIR = ROOT / "external" / "lama"
DATA_DIR = ROOT / "dataset"
sys.path.append(str(LAMA_DIR))
sys.path.append(str(DATA_DIR))
sys.path.append(str(ROOT / "utils"))

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from dataset import SEN12MSCRDataset
from hf_utils import download_model, resolve_load_version


# ── Modelo ─────────────────────────────────────────────────────────────

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


# ── Preparación de batch ───────────────────────────────────────────────

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


def register_eval(
    filename: str,
    *,
    metrics: dict,
    split: str,
    use_sar: bool,
    yaml_path: str = "eval/results.yaml",
) -> None:
    """
    Registra (o sobreescribe) los resultados de evaluación en un YAML local.

    Estructura resultante en eval/results.yaml:
```yaml
    models:
      lama_no_sar:
        v1:
          filename: lama_no_sar_finetuned_v1.pth
          eval:
            split: test
            date: "2026-06-08"
            mae:  0.012345
            psnr: 32.1234
            ssim: 0.987654
            sam:  1.2345
```
    """
    from datetime import date

    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    if "models" not in data:
        data["models"] = {}

    mkey = "lama_sar" if use_sar else "lama_no_sar"
    if mkey not in data["models"]:
        data["models"][mkey] = {}

    model_versions = data["models"][mkey]

    # Buscar vkey cuyo filename coincida, si no crear entrada mínima
    target_vkey = None
    for vkey, entry in model_versions.items():
        if entry.get("filename") == filename:
            target_vkey = vkey
            break

    if target_vkey is None:
        target_vkey = filename.replace(".pth", "")
        model_versions[target_vkey] = {"filename": filename}

    model_versions[target_vkey]["eval"] = {
        "split": split,
        "date":  str(date.today()),
        "mae":   round(metrics["mae"],  6),
        "psnr":  round(metrics["psnr"], 4),
        "ssim":  round(metrics["ssim"], 6),
        "sam":   round(metrics["sam"],  4),
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"Métricas guardadas en {yaml_path} (models.{mkey}.{target_vkey}.eval)")

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
            break

    print(f"\n{'='*40}")
    print(f"  MAE  : {total_mae  / n_batches:.6f}")
    print(f"  PSNR : {total_psnr / n_batches:.4f} dB")
    print(f"  SSIM : {total_ssim / n_batches:.6f}")
    print(f"  SAM  : {total_sam  / n_batches:.4f} °")
    print(f"{'='*40}\n")

    return {
        "mae":  float(total_mae  / n_batches),
        "psnr": float(total_psnr / n_batches),
        "ssim": float(total_ssim / n_batches),
        "sam":  float(total_sam  / n_batches),
    }


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Ejemplos de uso:
    #   python eval/eval_lama.py --config configs/lama_no_sar.yaml
    #   python eval/eval_lama.py --config configs/lama_no_sar.yaml --version 2
    #   python eval/eval_lama.py --config configs/lama_no_sar.yaml --split train
    #   python eval/eval_lama.py --config configs/lama_no_sar.yaml --pretrained

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, required=True)
    parser.add_argument("--version", type=int, default=None,
                        help="Versión del modelo a evaluar (e.g. 2 → _v2.pth). "
                             "Si no se indica, usa la última disponible.")
    parser.add_argument("--split",   type=str, default="test",
                        choices=["train", "test"],
                        help="Split a evaluar (default: test)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Evaluar el modelo base (pretrained) en vez del finetuned")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_eval = cfg["eval"]
    cfg_hf   = cfg["huggingface"]

    use_sar     = cfg.get("sar", False)
    batch_size  = cfg_eval.get("batch_size",  4)
    num_workers = cfg_eval.get("num_workers", 2)

    repo_id     = cfg_hf["repo_id"]         # e.g. "LucioLuque/lama"
    save_prefix = cfg_hf["save_prefix"]     # e.g. "lama_no_sar_finetuned"
    base_filename = cfg_hf["base_filename"] # e.g. "lama_no_sar_pretrained_v1.pth"

    # ── Resolver qué modelo cargar ────────────────────────────────────────────
    if args.pretrained:
        filename = base_filename
        print(f"Modo pretrained: usando '{filename}'")
    else:
        _, filename = resolve_load_version(
            repo_id=repo_id,
            filename_prefix=save_prefix,
            requested_version=args.version,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {device} | SAR: {use_sar} | split: {args.split} | Modelo: {filename}")

    # ── Cargar modelo desde HuggingFace ──────────────────────────────────────
    checkpoint = download_model(repo_id=repo_id, filename=filename, map_location=device)

    model = build_model(use_sar=use_sar)
    model.load_state_dict(checkpoint)
    model = model.float().to(device)

    # ── Dataset y DataLoader ──────────────────────────────────────────────────
    ds_test = SEN12MSCRDataset(split=args.split, include_s1=use_sar)
    loader  = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ── Evaluar ───────────────────────────────────────────────────────────────
    metrics = evaluate(model, loader, use_sar=use_sar, device=device)

    # ── Registrar métricas en versions.yaml ───────────────────────────────────
    register_eval(
       filename=filename,
        metrics=metrics,
        split=args.split,
        use_sar=use_sar,
        yaml_path="eval/results.yaml",
    )