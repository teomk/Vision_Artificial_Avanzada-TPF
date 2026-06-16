import gc
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dbcr_complex import DBCR


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()


def memory_gb():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak_allocated = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9
    return allocated, reserved, peak_allocated, peak_reserved


def make_fake_batch(batch_size, h, w, device):
    s2_clean = torch.randn(batch_size, 6, h, w, device=device)
    s2_cloudy = torch.randn(batch_size, 6, h, w, device=device)
    sar = torch.randn(batch_size, 2, h, w, device=device)

    t = torch.randint(0, 1000, (batch_size,), device=device)

    alpha = t.float() / 999.0
    alpha = alpha[:, None, None, None]

    # Bridge DBCR: t=0 cerca de clean, t=999 cerca de cloudy.
    x_t = (1.0 - alpha) * s2_clean + alpha * s2_cloudy

    return x_t, t, s2_cloudy, sar, s2_clean


def train_step(model, optimizer, batch_size, h, w, device):
    x_t, t, s2_cloudy, sar, s2_clean = make_fake_batch(batch_size, h, w, device)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = model(x_t, t, s2_cloudy, sar)
        loss = F.l1_loss(pred, s2_clean)

    loss.backward()
    optimizer.step()

    return loss.item()


def test_config(
    label,
    batch_size,
    h,
    w,
    base_channels,
    time_dim,
    window_size_sf0,
    window_size_not_sf0,
    use_checkpoint,
    include_encoder_4,
    lr,
    weight_decay,
    device,
):
    print("\n" + "=" * 80)
    print(f"CONFIG: {label}")
    print(f"batch={batch_size} | res={h}x{w} | base_channels={base_channels}")
    print(f"window_sf0={window_size_sf0} | window_not_sf0={window_size_not_sf0}")
    print(f"checkpoint={use_checkpoint} | optimizer=AdamW")
    print(f"include_encoder_4={include_encoder_4}")
    print("=" * 80)

    reset_memory()

    model = DBCR(
        image_channels=6,
        condition_channels=6,
        sar_channels=2,
        base_channels=base_channels,
        time_dim=time_dim,
        num_heads=1,
        window_size_sf0=window_size_sf0,
        window_size_not_sf0=window_size_not_sf0,
        use_checkpoint=use_checkpoint,
        include_encoder_4=include_encoder_4,
    ).to(device)

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

    print(f"Parámetros: {n_params:.2f} M")
    print(f"VRAM params: {param_mem:.3f} GB")

    try:
        # Primer step: inicializa estados de AdamW.
        loss1 = train_step(model, optimizer, batch_size, h, w, device)

        # Reseteamos pico después de que AdamW ya creó sus estados internos.
        torch.cuda.reset_peak_memory_stats()

        # Segundo step: mide entrenamiento real con optimizer ya inicializado.
        loss2 = train_step(model, optimizer, batch_size, h, w, device)

        allocated, reserved, peak_allocated, peak_reserved = memory_gb()

        print(f"loss step 1: {loss1:.6f}")
        print(f"loss step 2: {loss2:.6f}")
        print(f"VRAM allocated actual : {allocated:.2f} GB")
        print(f"VRAM reserved actual  : {reserved:.2f} GB")
        print(f"VRAM peak allocated   : {peak_allocated:.2f} GB")
        print(f"VRAM peak reserved    : {peak_reserved:.2f} GB")
        print("RESULTADO: OK")

        ok = True

    except torch.cuda.OutOfMemoryError:
        print("RESULTADO: OOM")
        ok = False

    finally:
        del model
        del optimizer
        reset_memory()

    return ok


def main():
    # python 
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--w", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument(
        "--enc4",
        choices=["off", "on", "both"],
        default="both",
        help="Probar sin enc4, con enc4 o ambas variantes."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA no disponible.")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    configs = [
        {
            "label": "MEJOR ACTUAL: sf0 window + checkpoint ON",
            "window_size_sf0": 8,
            "window_size_not_sf0": None,
            "use_checkpoint": True,
        },
        {
            "label": "SAFE: todos los SFBlock con window + checkpoint ON",
            "window_size_sf0": 8,
            "window_size_not_sf0": 8,
            "use_checkpoint": True,
        },
        {
            "label": "Sin checkpoint: sf0 window + checkpoint OFF",
            "window_size_sf0": 8,
            "window_size_not_sf0": None,
            "use_checkpoint": False,
        },
    ]

    summary = []

    for cfg in configs:
        max_ok = 0

        for batch_size in args.batches:
            ok = test_config(
                label=cfg["label"],
                batch_size=batch_size,
                h=args.h,
                w=args.w,
                base_channels=args.base_channels,
                time_dim=args.time_dim,
                window_size_sf0=cfg["window_size_sf0"],
                window_size_not_sf0=cfg["window_size_not_sf0"],
                use_checkpoint=cfg["use_checkpoint"],
                include_encoder_4=cfg["include_encoder_4"],
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )

            if ok:
                max_ok = batch_size
            else:
                break

        summary.append((cfg["label"], max_ok))

    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    for label, max_ok in summary:
        print(f"{label:<55} batch máximo OK: {max_ok}")


if __name__ == "__main__":
    main()