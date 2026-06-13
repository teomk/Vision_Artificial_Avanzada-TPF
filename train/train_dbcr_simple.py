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

from dbcr_simple import DBCRSimple
from hf_utils import download_model, upload_model, resolve_save_version, register_version
from dbcr_simple_utils import make_bridge_sample
from dataset_utils import unpack_batch

from dataset import SEN12MSCRDataset

def run(model, batch, optimizer, device, sar_mode, T=1000, sigmoid_k=10.0):

    model.train()

    s2_cloudy, s2_clean, condition, sar = unpack_batch(batch, sar_mode, device)

    B = s2_clean.shape[0]

    t = torch.randint(low=1, high=T + 1, size=(B,), device=device)

    x_t = make_bridge_sample(s2_clean=s2_clean, s2_cloudy=s2_cloudy, t=t, T=T, sigmoid_k=sigmoid_k, device=device)

    pred_clean = model(x_t=x_t, t=t, s2_cloudy=condition, sar=sar)

    loss = F.l1_loss(pred_clean, s2_clean)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()

def fit(model, train_loader, lr , device, sar_mode, num_epochs=50, T=1000, sigmoid_k=10.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)
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
            progress_bar.set_postfix({"loss": f"{loss:.6f}", "avg_loss": f"{avg_loss:.6f}"})
            break

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)
        scheduler.step(avg_loss)

    return history
 
if __name__ == "__main__":
    # python train/train_dbcr_simple.py --config configs/dbcr_none.yaml
    # python train/train_dbcr_simple.py --config configs/dbcr_concat.yaml
    # python train/train_dbcr_simple.py --config configs/dbcr_controlnet.yaml
 
    parser = argparse.ArgumentParser(description="Entrenar DBCR (SAR o No-SAR)")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Ruta al config YAML (e.g. configs/dbcr_no_sar.yaml)"
    )
    args = parser.parse_args()
 
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
 
    sar_mode    = cfg["sar_mode"]
    cfg_train   = cfg["train"]
    cfg_hf      = cfg["huggingface"]
 
    batch_size  = cfg_train["batch_size"]
    num_workers = cfg_train["num_workers"]
    num_epochs  = cfg_train["num_epochs"]
    lr          = cfg_train["lr"]
    T           = cfg_train["T"]
    sigmoid_k   = cfg_train["sigmoid_k"]
    load_filename = cfg_train["load_filename"]
    load_filename = None if load_filename == "None" else load_filename

 
    repo_id       = cfg_hf["repo_id"]
    save_filename = cfg_hf["save_filename"]
    version       = cfg_hf["version"]
    notes         = cfg_hf["notes"]
 
    # Arquitectura fija
    image_channels     = 6
    base_channels      = 64
    time_dim           = 128

    condition_channels = 8 if sar_mode == "Concat" else 6
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SAR Mode: {sar_mode} | Condition Channels: {condition_channels} | Epochs: {num_epochs}")
 
    # Dataset
    ds_train = SEN12MSCRDataset(split="train", include_s1= sar_mode != "None", include_mask=False)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = DBCRSimple(
        image_channels=image_channels,
        condition_channels=condition_channels,
        base_channels=base_channels,
        time_dim=time_dim,
        control_net=(sar_mode == "ControlNet")
    ).to(device)

    if load_filename is not None:
        checkpoint = download_model(repo_id=repo_id, filename=load_filename, map_location=device)
        if sar_mode == "Concat":
            old_weight = checkpoint["condition_encoder.in_conv.weight"]  # [64, 6, 3, 3]
            new_weight = torch.zeros(
                old_weight.shape[0],       # 64
                8,                          # 8 canales
                old_weight.shape[2],       # 3
                old_weight.shape[3],       # 3
                device=old_weight.device,
                dtype=old_weight.dtype,
            )
            new_weight[:, :6, :, :] = old_weight  # canales ópticos
            new_weight[:, 6:, :, :] = 0           # SAR arranca en cero
            checkpoint["condition_encoder.in_conv.weight"] = new_weight

        model.load_state_dict(checkpoint, strict=False)
        print(f"Modelo cargado desde HuggingFace: {repo_id}/{load_filename}")

    if sar_mode == "ControlNet":
        model.freeze_unet()
        print("UNet congelada. Solo se entrenará el ControlNet.")
        #chequear que efectivamente solo el control net tiene parámetros entrenables
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        controlnet_params = sum(p.numel() for p in model.control_net.parameters() if p.requires_grad)
        assert trainable_params == controlnet_params, f"Error: Se esperaban {controlnet_params} parámetros entrenables, pero se encontraron {trainable_params}."
        print(f"Parámetros entrenables (ControlNet): {trainable_params:,}")
 
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {parameters:,}")
 
    # Entrenar
    history = fit(
        model=model, train_loader=loader_train,
        lr=lr, device=device, sar_mode=sar_mode,
        num_epochs=num_epochs, T=T, sigmoid_k=sigmoid_k
    )
 
    # Subir a HuggingFace
    upload_model(
        model_state_dict=model.state_dict(),
        repo_id=repo_id,
        filename=save_filename,
    )
 
    # Registrar versión
    register_version(
        repo_id=repo_id,
        version=version,
        filename=save_filename,
        base_model=save_filename,
        use_sar=sar_mode,
        phase1_info={
            "num_epochs": num_epochs, "lr": lr,
            "T": T, "sigmoid_k": sigmoid_k,
            "batch_size": batch_size, "num_weights": parameters, "sar_mode": sar_mode
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
        "use_sar": sar_mode,
        "num_parameters": parameters,
        }
    }

    history_dir = ROOT / "training_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    history_filename = save_filename.replace(".pth", "_history.yaml")
    history_path = history_dir / history_filename

    with open(history_path, "w") as f:
        yaml.safe_dump(history_data, f, sort_keys=False)

    print(f"History guardado en: {history_path}")
 
