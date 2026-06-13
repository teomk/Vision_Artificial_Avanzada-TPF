import sys
import time
import argparse
from pathlib import Path

import yaml
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
from dataset_utils import unpack_batch
from dbcr_simple_utils import make_bridge_sample

# Importa EL MODELO NUEVO, el del vram_check.py
from vram_check import DBCR


def run_dbcr_batch(
    model,
    batch,
    optimizer,
    device,
    T=1000,
    sigmoid_k=10.0,
    use_bf16=True,
    accum_steps=1,
    do_step=True,
):
    model.train()

    # Importante: este DBCR SIEMPRE necesita SAR
    # unpack_batch devuelve: s2_cloudy, s2_clean, condition, sar
    # Para este modelo ignoramos "condition" y usamos s2_cloudy + sar separados.
    s2_cloudy, s2_clean, _, sar = unpack_batch(batch, "ControlNet", device)

    if sar is None:
        raise RuntimeError("Este DBCR necesita SAR, pero sar llegó como None. Revisá include_s1=True.")

    B = s2_clean.shape[0]

    t = torch.randint(
        low=1,
        high=T + 1,
        size=(B,),
        device=device,
    )

    x_t = make_bridge_sample(
        s2_clean=s2_clean,
        s2_cloudy=s2_cloudy,
        t=t,
        T=T,
        sigmoid_k=sigmoid_k,
        device=device,
    )

    amp_enabled = use_bf16 and device.type == "cuda"

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        pred_clean = model(
            x_t=x_t,
            t=t,
            s2_cloudy=s2_cloudy,
            sar=sar,
        )

        loss = F.l1_loss(pred_clean, s2_clean)
        loss_for_backward = loss / accum_steps

    loss_for_backward.backward()

    if do_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return loss.item()


def benchmark_epoch_time(
    model,
    train_loader,
    lr,
    device,
    T=1000,
    sigmoid_k=10.0,
    use_bf16=True,
    warmup_batches=5,
    timed_batches=30,
    accum_steps=1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    model.train()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 70)
    print("BENCHMARK DBCR — TIEMPO ESTIMADO POR ÉPOCA")
    print("=" * 70)
    print(f"Dataset samples      : {len(train_loader.dataset)}")
    print(f"Batches por época    : {len(train_loader)}")
    print(f"Batch size real      : {train_loader.batch_size}")
    print(f"Accum steps          : {accum_steps}")
    print(f"Batch efectivo       : {train_loader.batch_size * accum_steps}")
    print(f"Warmup batches       : {warmup_batches}")
    print(f"Timed batches        : {timed_batches}")
    print(f"bf16 autocast        : {use_bf16}")
    print("=" * 70)

    it = iter(train_loader)

    # Warmup
    print("\nWarmup...")
    for i in tqdm(range(warmup_batches), desc="Warmup", unit="batch"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        do_step = ((i + 1) % accum_steps == 0)

        _ = run_dbcr_batch(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            T=T,
            sigmoid_k=sigmoid_k,
            use_bf16=use_bf16,
            accum_steps=accum_steps,
            do_step=do_step,
        )

    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Medición real
    print("\nMidiendo batches reales...")
    batch_times = []
    losses = []

    for i in tqdm(range(timed_batches), desc="Benchmark", unit="batch"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        do_step = ((i + 1) % accum_steps == 0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        loss = run_dbcr_batch(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            T=T,
            sigmoid_k=sigmoid_k,
            use_bf16=use_bf16,
            accum_steps=accum_steps,
            do_step=do_step,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        batch_times.append(t1 - t0)
        losses.append(loss)

    avg_batch_time = sum(batch_times) / len(batch_times)
    epoch_seconds = avg_batch_time * len(train_loader)

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_vram = 0.0

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Tiempo promedio/batch : {avg_batch_time:.3f} s")
    print(f"Batches por época     : {len(train_loader)}")
    print(f"Tiempo estimado época : {epoch_seconds / 60:.2f} min")
    print(f"Tiempo estimado época : {epoch_seconds / 3600:.2f} h")
    print(f"VRAM pico             : {peak_vram:.3f} GB")
    print(f"Última loss medida    : {losses[-1]:.6f}")
    print("=" * 70 + "\n")


def fit_dbcr(
    model,
    train_loader,
    lr,
    device,
    num_epochs=50,
    T=1000,
    sigmoid_k=10.0,
    use_bf16=True,
    accum_steps=1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    history = {"train_loss": []}

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Época {epoch + 1}/{num_epochs}",
            unit="batch",
        )

        for step, batch in enumerate(progress_bar):
            do_step = ((step + 1) % accum_steps == 0)

            loss = run_dbcr_batch(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                T=T,
                sigmoid_k=sigmoid_k,
                use_bf16=use_bf16,
                accum_steps=accum_steps,
                do_step=do_step,
            )

            epoch_loss += loss
            num_batches += 1
            avg_loss = epoch_loss / num_batches

            progress_bar.set_postfix({
                "loss": f"{loss:.6f}",
                "avg": f"{avg_loss:.6f}",
            })

        # Si la cantidad de batches no es múltiplo de accum_steps,
        # hacemos un último step pendiente.
        if num_batches % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)
        scheduler.step(avg_loss)

        print(f"Época {epoch + 1} terminada | avg_loss={avg_loss:.6f}")

    return history


if __name__ == "__main__":

    #python train/benchmark_dbcr.py   --config configs/dbcr_controlnet.yaml   --benchmark-only   --batch-size 2   --window-size-sf0 8   --accum-steps 4   --timed-batches 30
    parser = argparse.ArgumentParser(description="Benchmark / train DBCR real")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--benchmark-only", action="store_true")

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)

    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=1)

    # Esta versión replica lo que probaste:
    # window solamente en sf0; el resto global.
    parser.add_argument("--window-size-sf0", type=int, default=8)

    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--no-bf16", action="store_true")

    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--timed-batches", type=int, default=30)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_train = cfg["train"]

    batch_size = args.batch_size if args.batch_size is not None else cfg_train["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg_train["num_workers"]
    num_epochs = args.epochs if args.epochs is not None else cfg_train["num_epochs"]

    lr = cfg_train["lr"]
    T = cfg_train["T"]
    sigmoid_k = cfg_train["sigmoid_k"]

    use_bf16 = not args.no_bf16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Modelo: DBCR real con SAR branch")
    print(f"Batch size: {batch_size}")
    print(f"Accum steps: {args.accum_steps}")
    print(f"Batch efectivo: {batch_size * args.accum_steps}")
    print(f"Window sf0: {args.window_size_sf0}")
    print(f"bf16: {use_bf16}")

    # Dataset: este DBCR necesita S1/SAR sí o sí
    ds_train = SEN12MSCRDataset(
        split="train",
        include_s1=True,
        include_mask=False,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = DBCR(
        image_ch=6,
        cond_ch=6,
        sar_ch=2,
        base_ch=args.base_ch,
        time_dim=args.time_dim,
        num_heads=args.num_heads,
        use_checkpoint=True,
        window_size=None,                     # sf1/sf2/sf3/mid global
        window_size_sf0=args.window_size_sf0, # sf0 window
    ).to(device)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {parameters:,}")

    if args.benchmark_only:
        benchmark_epoch_time(
            model=model,
            train_loader=loader_train,
            lr=lr,
            device=device,
            T=T,
            sigmoid_k=sigmoid_k,
            use_bf16=use_bf16,
            warmup_batches=args.warmup_batches,
            timed_batches=args.timed_batches,
            accum_steps=args.accum_steps,
        )
        raise SystemExit

    history = fit_dbcr(
        model=model,
        train_loader=loader_train,
        lr=lr,
        device=device,
        num_epochs=num_epochs,
        T=T,
        sigmoid_k=sigmoid_k,
        use_bf16=use_bf16,
        accum_steps=args.accum_steps,
    )

    print("Entrenamiento terminado.")
    print(history)
