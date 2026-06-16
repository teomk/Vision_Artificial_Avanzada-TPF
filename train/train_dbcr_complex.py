import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import yaml

import argparse
from torch.utils.data import DataLoader

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dbcr_complex import DBCR
from hf_utils import download_model, upload_model, register_version
from dbcr_simple_utils import make_bridge_sample
from dataset_utils import unpack_batch

from dataset import SEN12MSCRDataset


def parse_window(value):
    """'8' -> 8, 'None'/'none'/'null'/'-1' -> None"""
    if value is None:
        return None
    value = str(value).lower()
    if value in ["none", "null", "-1"]:
        return None
    return int(value)


def run(model, batch, optimizer, device, sar_mode, T=1000, sigmoid_k=10.0):
    model.train()

    s2_cloudy, s2_clean, condition, sar = unpack_batch(batch, sar_mode, device)

    if sar is None:
        raise ValueError(
            f"DBCR (complex, con SFBlocks) necesita SAR como tensor separado. "
            f"sar_mode='{sar_mode}' devolvió sar=None. Usá sar_mode='ControlNet' "
            f"(o el modo de tu unpack_batch que entregue s1 por separado)."
        )

    B = s2_clean.shape[0]

    t = torch.randint(low=1, high=T + 1, size=(B,), device=device)

    x_t = make_bridge_sample(s2_clean=s2_clean, s2_cloudy=s2_cloudy, t=t, T=T, sigmoid_k=sigmoid_k, device=device)

    if device.type == "cuda":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred_clean = model(x_t=x_t, t=t, s2_cloudy=s2_cloudy, sar=sar)
            loss = F.l1_loss(pred_clean, s2_clean)
    else:
        pred_clean = model(x_t=x_t, t=t, s2_cloudy=s2_cloudy, sar=sar)
        loss = F.l1_loss(pred_clean, s2_clean)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def fit(model, train_loader, lr, device, sar_mode, num_epochs=50, T=1000, sigmoid_k=10.0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    history = {"train_loss": []}

    for epoch in range(num_epochs):

        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs}", unit="batch")

        for batch in progress_bar:
            loss = run(model=model, batch=batch, optimizer=optimizer, device=device, sar_mode=sar_mode, T=T, sigmoid_k=sigmoid_k)

            epoch_loss += loss
            num_batches += 1
            avg_loss = epoch_loss / num_batches

            postfix = {"loss": f"{loss:.6f}", "avg_loss": f"{avg_loss:.6f}"}
            if device.type == "cuda":
                postfix["VRAM"] = f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            progress_bar.set_postfix(postfix)

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)
        scheduler.step(avg_loss)

        if device.type == "cuda":
            print(f"  Época {epoch+1}: avg_loss={avg_loss:.6f} | "
                  f"VRAM pico={torch.cuda.max_memory_allocated()/1e9:.2f}GB | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    return history


if __name__ == "__main__":
    # python train/train_dbcr.py --config configs/dbcr_complex.yaml

    parser = argparse.ArgumentParser(description="Entrenar DBCR con SFBlocks (cross-attention SAR)")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Ruta al config YAML (e.g. configs/dbcr_complex.yaml)"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sar_mode    = cfg["sar_mode"]            # debe ser "ControlNet" (o el modo que entregue s1 separado)
    cfg_train   = cfg["train"]
    cfg_model   = cfg.get("model_args", {})
    cfg_hf      = cfg["huggingface"]

    batch_size  = cfg_train["batch_size"]
    num_workers = cfg_train["num_workers"]
    num_epochs  = cfg_train["num_epochs"]
    lr          = cfg_train["lr"]
    T           = cfg_train["T"]
    sigmoid_k   = cfg_train["sigmoid_k"]
    load_filename = cfg_train.get("load_filename")
    load_filename = None if load_filename in (None, "None") else load_filename

    repo_id       = cfg_hf["repo_id"]
    save_filename = cfg_hf["save_filename"]
    version       = cfg_hf["version"]
    notes         = cfg_hf["notes"]

    # Arquitectura
    image_channels      = cfg_model.get("image_channels", 6)
    condition_channels  = cfg_model.get("condition_channels", 6)
    sar_channels        = cfg_model.get("sar_channels", 2)
    base_channels       = cfg_model.get("base_channels", 64)
    time_dim            = cfg_model.get("time_dim", 128)
    num_heads           = cfg_model.get("num_heads", 1)
    window_size_sf0     = parse_window(cfg_model.get("window_size_sf0", 8))
    window_size_not_sf0 = parse_window(cfg_model.get("window_size_not_sf0", None))
    use_checkpoint      = cfg_model.get("use_checkpoint", True)
    include_encoder_4   = cfg_model.get("include_encoder_4", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SAR Mode: {sar_mode} | Epochs: {num_epochs}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print()
    print("CONFIG MODELO")
    print(f"  image_channels      : {image_channels}")
    print(f"  condition_channels  : {condition_channels}")
    print(f"  sar_channels        : {sar_channels}")
    print(f"  base_channels       : {base_channels}")
    print(f"  time_dim            : {time_dim}")
    print(f"  num_heads           : {num_heads}")
    print(f"  window_size_sf0     : {window_size_sf0}")
    print(f"  window_size_not_sf0 : {window_size_not_sf0}")
    print(f"  use_checkpoint      : {use_checkpoint}")
    print(f"  include_encoder_4   : {include_encoder_4}")
    print()

    # Dataset
    ds_train = SEN12MSCRDataset(split="train", include_s1=(sar_mode != "None"), include_mask=False)

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = DBCR(
        image_channels=image_channels,
        condition_channels=condition_channels,
        sar_channels=sar_channels,
        base_channels=base_channels,
        time_dim=time_dim,
        num_heads=num_heads,
        window_size_sf0=window_size_sf0,
        window_size_not_sf0=window_size_not_sf0,
        use_checkpoint=use_checkpoint,
        include_encoder_4=include_encoder_4,
    ).to(device)

    if load_filename is not None:
        checkpoint = download_model(repo_id=repo_id, filename=load_filename, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Modelo cargado desde HuggingFace: {repo_id}/{load_filename}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales    : {n_params:.2f} M")
    print(f"Parámetros entrenables: {parameters:,}")
    print(f"Samples train: {len(ds_train)}")
    print(f"Batches por época: {len(loader_train)}")
    print()

    # Entrenar
    history = fit(
        model=model, train_loader=loader_train,
        lr=lr, device=device, sar_mode=sar_mode,
        num_epochs=num_epochs, T=T, sigmoid_k=sigmoid_k
    )

    # Subir a HuggingFace
    upload_model(model_state_dict=model.state_dict(), repo_id=repo_id, filename=save_filename)

    # Registrar versión
    register_version(
        repo_id=repo_id,
        version=version,
        filename=save_filename,
        base_model=save_filename,
        sar_mode=sar_mode,
        phase1_info={
            "num_epochs": num_epochs, "lr": lr,
            "T": T, "sigmoid_k": sigmoid_k,
            "batch_size": batch_size, "num_weights": parameters,
            "sar_mode": sar_mode,
            "base_channels": base_channels,
            "time_dim": time_dim,
            "window_size_sf0": window_size_sf0,
            "window_size_not_sf0": window_size_not_sf0,
            "use_checkpoint": use_checkpoint,
            "include_encoder_4": include_encoder_4,
        },
        phase2_info={
            "num_epochs": 0, "lr": 0,
            "T": 0, "sigmoid_k": 0,
            "batch_size": 0, "num_weights": 0
        },
        notes=notes,
    )

    history_data = {
        "train_loss": history["train_loss"],
        "config": {
            "num_epochs": num_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "T": T,
            "sigmoid_k": sigmoid_k,
            "sar_mode": sar_mode,
            "num_parameters": parameters,
            "base_channels": base_channels,
            "time_dim": time_dim,
            "window_size_sf0": window_size_sf0,
            "window_size_not_sf0": window_size_not_sf0,
            "include_encoder_4": include_encoder_4,
        }
    }

    history_dir = ROOT / "training_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    history_filename = save_filename.replace(".pth", "_history.yaml")
    history_path = history_dir / history_filename

    with open(history_path, "w") as f:
        yaml.safe_dump(history_data, f, sort_keys=False)

    print(f"History guardado en: {history_path}")