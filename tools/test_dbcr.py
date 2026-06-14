"""
Profiling de VRAM para DBCR.

Corre forward+backward con distintas configuraciones (window attention
en sf0, resolución, batch size) y reporta:
  - VRAM pico total
  - VRAM usada solo por los parámetros del modelo
  - Batch size máximo estimado para 24 GB

Uso:
    python profile_dbcr_vram.py
"""

import gc
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Importá tu modelo. Ajustá el path/import según tu estructura de proyecto.
# ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dbcr_complex import DBCR  # Ajustá el import según tu estructura de proyecto


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_model_param_memory(model):
    """VRAM ocupada solo por los parámetros (sin activaciones)."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / 1e9  # GB


def run_forward_backward(model, B, H, W, device):
    x_t       = torch.randn(B, 6, H, W, device=device)
    t         = torch.randint(0, 1000, (B,), device=device)
    s2_cloudy = torch.randn(B, 6, H, W, device=device)
    sar       = torch.randn(B, 2, H, W, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out  = model(x_t, t, s2_cloudy, sar)
        loss = F.l1_loss(out, s2_cloudy)  # placeholder loss

    loss.backward()
    model.zero_grad(set_to_none=True)


def profile_config(
    label,
    window_size_sf0,
    window_size_not_sf0,
    use_checkpoint,
    base_channels,
    time_dim,
    H, W,
    batch_sizes,
    device,
    max_vram_gb=24.0,
):
    print(f"\n{'='*70}")
    print(f"  CONFIG: {label}")
    print(f"  window_sf0={window_size_sf0} | window_not_sf0={window_size_not_sf0} | "
          f"checkpoint={use_checkpoint} | base_channels={base_channels} | "
          f"resolución={H}x{W}")
    print(f"{'='*70}")

    reset_memory()

    model = DBCR(
        image_channels=6,
        condition_channels=6,
        sar_channels=2,
        base_channels=base_channels,
        time_dim=time_dim,
        num_heads=1,
        window_size_not_sf0=window_size_not_sf0,
        window_size_sf0=window_size_sf0,
        use_checkpoint=use_checkpoint,
    ).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    param_vram = get_model_param_memory(model)

    print(f"  Parámetros totales : {n_params:.2f} M")
    print(f"  VRAM solo params   : {param_vram:.3f} GB")
    print()

    results = []
    max_ok_batch = 0

    for B in batch_sizes:
        reset_memory()
        try:
            run_forward_backward(model, B, H, W, device)
            peak = torch.cuda.max_memory_allocated() / 1e9
            fits = peak <= max_vram_gb
            status = "OK " if fits else "NO ENTRA (excede max_vram_gb)"
            print(f"  batch={B:>3} -> VRAM pico: {peak:6.2f} GB   [{status}]")
            results.append((B, peak, fits))
            if fits:
                max_ok_batch = B
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={B:>3} -> OOM (CUDA out of memory)")
            results.append((B, None, False))
            reset_memory()
            break  # batches mayores también van a OOM

    print()
    if max_ok_batch > 0:
        print(f"  >> Batch size máximo recomendado para {max_vram_gb} GB: {max_ok_batch}")
    else:
        print(f"  >> Ningún batch size probado entra en {max_vram_gb} GB")

    del model
    reset_memory()

    return {
        "label": label,
        "n_params_M": n_params,
        "param_vram_gb": param_vram,
        "results": results,
        "max_ok_batch": max_ok_batch,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA no disponible. Este profiling necesita GPU.")
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM total: {total_vram:.2f} GB")

    # ── Parámetros a ajustar ──────────────────────────────────────────
    H, W           = 256, 256     # resolución de entrada
    base_channels  = 64
    time_dim       = 128
    max_vram_gb    = 24.0
    batch_sizes    = [1, 2, 4, 6, 8, 12, 16]
    # ────────────────────────────────────────────────────────────────

    configs = [
        dict(
            label="Window en sf0 (ws=8) + checkpoint ON",
            window_size_sf0=8,
            window_size_not_sf0=None,
            use_checkpoint=True,
        ),
        dict(
            label="Window en sf0 (ws=8) + checkpoint OFF",
            window_size_sf0=8,
            window_size_not_sf0=None,
            use_checkpoint=False,M
        ),
        dict(
            label="Sin window en sf0 (global) + checkpoint ON",
            window_size_sf0=None,
            window_size_not_sf0=None,
            use_checkpoint=True,
        ),
        dict(
            label="Sin window en sf0 (global) + checkpoint OFF",
            window_size_sf0=None,
            window_size_not_sf0=None,
            use_checkpoint=False,
        ),
    ]

    all_results = []
    for cfg in configs:
        try:
            res = profile_config(
                **cfg,
                base_channels=base_channels,
                time_dim=time_dim,
                H=H, W=W,
                batch_sizes=batch_sizes,
                device=device,
                max_vram_gb=max_vram_gb,
            )
            all_results.append(res)
        except Exception as e:
            print(f"  ERROR en config '{cfg['label']}': {e}")
            reset_memory()

    # ── Resumen final ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESUMEN")
    print(f"{'='*70}")
    print(f"  Resolución: {H}x{W} | base_channels={base_channels} | time_dim={time_dim}")
    print(f"  Límite VRAM considerado: {max_vram_gb} GB\n")
    print(f"  {'Config':<45} {'Params (M)':>10} {'Batch máx':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10}")
    for r in all_results:
        print(f"  {r['label']:<45} {r['n_params_M']:>10.2f} {r['max_ok_batch']:>10}")


if __name__ == "__main__":
    main()