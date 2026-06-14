import gc
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dataset import SEN12MSCRDataset
from dbcr_complex import DBCR

try:
    from dataset_utils import unpack_batch
except ImportError:
    unpack_batch = None


def parse_window(value):
    if value is None:
        return None
    value = str(value).lower()
    if value in ["none", "null", "-1"]:
        return None
    return int(value)


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def make_bridge_xt(s2_clean, s2_cloudy, t, num_timesteps=1000):
    """
    Bridge DBCR:
    t bajo  -> más cerca de clean
    t alto  -> más cerca de cloudy
    """
    alpha = t.float() / float(num_timesteps - 1)
    alpha = alpha[:, None, None, None]
    x_t = (1.0 - alpha) * s2_clean + alpha * s2_cloudy
    return x_t


def unpack_dbcr_batch(batch, sar_mode, device):
    """
    Usa tu unpack_batch si existe.
    Espera:
        s2_cloudy: [B, 6, H, W]
        s2_clean : [B, 6, H, W]
        sar      : [B, 2, H, W]
    """

    if unpack_batch is not None:
        s2_cloudy, s2_clean, condition, sar = unpack_batch(batch, sar_mode, device)

        if sar is None:
            raise ValueError("DBCR necesita SAR. Usá --sar-mode Concat o el modo que cargue S1.")

        return (
            s2_cloudy.float(),
            s2_clean.float(),
            sar.float(),
        )

    if isinstance(batch, dict):
        def get_any(keys):
            for k in keys:
                if k in batch:
                    return batch[k]
            return None

        s2_cloudy = get_any(["s2_cloudy", "cloudy", "input"])
        s2_clean = get_any(["s2_clean", "s2_clear", "s2", "clear", "target"])
        sar = get_any(["s1", "sar", "s1_sar"])

        if s2_cloudy is None or s2_clean is None or sar is None:
            raise ValueError(f"No pude interpretar las keys del batch: {batch.keys()}")

        return (
            s2_cloudy.to(device, non_blocking=True).float(),
            s2_clean.to(device, non_blocking=True).float(),
            sar.to(device, non_blocking=True).float(),
        )

    raise ValueError(
        "No pude desempaquetar el batch. Importá dataset_utils.unpack_batch "
        "o adaptá unpack_dbcr_batch al formato de tu Dataset."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sar-mode", type=str, default="ControlNet")

    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=128)

    parser.add_argument("--window-size-sf0", type=str, default="8")
    parser.add_argument("--window-size-not-sf0", type=str, default="None")
    parser.add_argument("--checkpoint", action="store_true", default=True)
    parser.add_argument("--no-checkpoint", action="store_false", dest="checkpoint")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-timesteps", type=int, default=1000)

    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible.")

    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    window_size_sf0 = parse_window(args.window_size_sf0)
    window_size_not_sf0 = parse_window(args.window_size_not_sf0)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    print("CONFIG")
    print(f"batch_size           : {args.batch_size}")
    print(f"num_workers          : {args.num_workers}")
    print(f"sar_mode             : {args.sar_mode}")
    print(f"base_channels        : {args.base_channels}")
    print(f"time_dim             : {args.time_dim}")
    print(f"window_size_sf0      : {window_size_sf0}")
    print(f"window_size_not_sf0  : {window_size_not_sf0}")
    print(f"checkpoint           : {args.checkpoint}")
    print(f"lr                   : {args.lr}")
    print(f"weight_decay         : {args.weight_decay}")
    print()

    ds_train = SEN12MSCRDataset(
        split="train",
        include_s1=args.sar_mode != "None",
        include_mask=False,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = DBCR(
        image_channels=6,
        condition_channels=6,
        sar_channels=2,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        num_heads=1,
        window_size_sf0=window_size_sf0,
        window_size_not_sf0=window_size_not_sf0,
        use_checkpoint=args.checkpoint,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Dataset train: {len(ds_train)} muestras")
    print(f"Modelo       : {n_params:.2f} M parámetros")
    print()

    model.train()
    reset_memory()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    running_loss = 0.0
    num_seen = 0

    total_batches = len(loader_train)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)

    pbar = tqdm(loader_train, total=total_batches, desc="Train 1 epoch")

    for batch_idx, batch in enumerate(pbar, start=1):
        if args.max_batches is not None and batch_idx > args.max_batches:
            break

        s2_cloudy, s2_clean, sar = unpack_dbcr_batch(batch, args.sar_mode, device)

        B = s2_clean.shape[0]
        t = torch.randint(
            low=0,
            high=args.num_timesteps,
            size=(B,),
            device=device,
        )

        x_t = make_bridge_xt(
            s2_clean=s2_clean,
            s2_cloudy=s2_cloudy,
            t=t,
            num_timesteps=args.num_timesteps,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x_t, t, s2_cloudy, sar)
            loss = F.l1_loss(pred, s2_clean)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * B
        num_seen += B

        if batch_idx % args.log_every == 0 or batch_idx == 1:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            peak = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            avg_loss = running_loss / max(num_seen, 1)
            samples_per_sec = num_seen / elapsed

            pbar.set_postfix({
                "loss": f"{avg_loss:.5f}",
                "VRAM": f"{peak:.2f}GB",
                "it/samp": f"{samples_per_sec:.2f}",
            })

            print(
                f"\nBatch {batch_idx}/{total_batches} | "
                f"loss={avg_loss:.6f} | "
                f"elapsed={elapsed/60:.2f} min | "
                f"samples/s={samples_per_sec:.2f} | "
                f"peak_alloc={peak:.2f} GB | "
                f"peak_reserved={reserved:.2f} GB"
            )

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    avg_loss = running_loss / max(num_seen, 1)
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9

    print()
    print("=" * 80)
    print("FIN 1 ÉPOCA")
    print("=" * 80)
    print(f"muestras vistas       : {num_seen}")
    print(f"loss promedio         : {avg_loss:.6f}")
    print(f"tiempo total          : {total_time / 60:.2f} min")
    print(f"samples/s             : {num_seen / total_time:.2f}")
    print(f"VRAM peak allocated   : {peak:.2f} GB")
    print(f"VRAM peak reserved    : {reserved:.2f} GB")

    if args.save:
        out_dir = ROOT / "runs" / "dbcr_one_epoch_test"
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = out_dir / "dbcr_one_epoch.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "loss": avg_loss,
            },
            ckpt_path,
        )
        print(f"Checkpoint guardado en: {ckpt_path}")


if __name__ == "__main__":
    main()