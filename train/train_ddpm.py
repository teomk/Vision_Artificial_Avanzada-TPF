import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from torch.utils.data import DataLoader

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR  = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from ddpm import ConditionalDDPMUNet
from hf_utils import download_model, upload_model, register_version
from ddpm_utils import build_sigmoid_ddpm_scheduler
from dataset_utils import unpack_batch
from dataset import SEN12MSCRDataset

# ──────────────────────────────────────────────────────────────────────────────
# Train step
# ──────────────────────────────────────────────────────────────────────────────

def run(model, batch, optimizer, device, sar_mode, noise_scheduler):
    model.train()

    s2_cloudy, s2_clean, condition, sar = unpack_batch(batch, sar_mode, device)

    B = s2_clean.shape[0]

    # Ruido gaussiano y timesteps aleatorios
    noise = torch.randn_like(s2_clean)
    t     = torch.randint(low=1, high=noise_scheduler.config.num_train_timesteps, size=(B,), device=device)

    # Forward: agregar ruido según el scheduler
    x_t = noise_scheduler.add_noise(s2_clean, noise, t)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=="cuda"):
        noise_pred = model(x_t=x_t, t=t, s2_cloudy=condition, sar=sar)
        loss = F.mse_loss(noise_pred, noise)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


# ──────────────────────────────────────────────────────────────────────────────
# Fit loop
# ──────────────────────────────────────────────────────────────────────────────

def fit(model, train_loader, lr, device, sar_mode, noise_scheduler, num_epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    history = {"train_loss": []}

    for epoch in range(num_epochs):
        epoch_loss  = 0.0
        num_batches = 0
        progress_bar = tqdm(
            train_loader,
            desc=f"Época {epoch+1}/{num_epochs}",
            unit="batch"
        )

        for batch in progress_bar:
            loss = run(
                model=model, batch=batch, optimizer=optimizer,
                device=device, sar_mode=sar_mode,
                noise_scheduler=noise_scheduler,
            )
            epoch_loss  += loss
            num_batches += 1
            avg_loss = epoch_loss / num_batches
            progress_bar.set_postfix({
                "loss":     f"{loss:.6f}",
                "avg_loss": f"{avg_loss:.6f}"
            })
            # break

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)
        scheduler.step(avg_loss)

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # python train/train_ddpm.py --config configs/ddpm_none.yaml
    # python train/train_ddpm.py --config configs/ddpm_concat.yaml
    # python train/train_ddpm.py --config configs/ddpm_controlnet.yaml

    parser = argparse.ArgumentParser(description="Entrenar ConditionalDDPMUNet (None | Concat | ControlNet)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sar_mode  = cfg["sar_mode"]
    cfg_train = cfg["train"]
    cfg_hf    = cfg["huggingface"]

    batch_size    = cfg_train["batch_size"]
    num_workers   = cfg_train["num_workers"]
    num_epochs    = cfg_train["num_epochs"]
    lr            = cfg_train["lr"]
    T             = cfg_train["T"]
    load_filename = cfg_train.get("load_filename")
    load_filename = None if load_filename in (None, "None") else load_filename

    repo_id       = cfg_hf["repo_id"]
    save_filename = cfg_hf["save_filename"]
    version       = cfg_hf["version"]
    notes         = cfg_hf["notes"]

    # Arquitectura
    image_channels     = 6
    base_channels      = 64
    time_dim           = 128
    condition_channels = 8 if sar_mode == "Concat" else 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | sar_mode: {sar_mode} | condition_channels: {condition_channels} | Epochs: {num_epochs}")

    # Noise scheduler — sigmoid beta schedule
    sigmoid_k = cfg_train.get("sigmoid_k", 25.0)
    alpha_min = cfg_train.get("alpha_min", 0.0001)

    noise_scheduler = build_sigmoid_ddpm_scheduler(
        T=T,
        sigmoid_k=sigmoid_k,
        alpha_min=alpha_min
    )

    # # Dataset
    # ds_train = SEN12MSCRDataset(
    #     split="train",
    #     include_s1=(sar_mode != "None"),
    #     include_mask=False
    # )
    # loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # # Modelo
    # model = ConditionalDDPMUNet(
    #     image_channels=image_channels,
    #     condition_channels=condition_channels,
    #     base_channels=base_channels,
    #     time_dim=time_dim,
    # ).to(device)

    # if load_filename is not None:
    #     checkpoint = download_model(repo_id=repo_id, filename=load_filename, map_location=device)
    #     model.load_state_dict(checkpoint, strict=False)
    #     print(f"Modelo cargado desde HuggingFace: {repo_id}/{load_filename}")

    # parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Parámetros entrenables: {parameters:,}")

    # # Entrenar
    # history = fit(
    #     model=model,
    #     train_loader=loader_train,
    #     lr=lr,
    #     device=device,
    #     sar_mode=sar_mode,
    #     noise_scheduler=noise_scheduler,
    #     num_epochs=num_epochs,
    # )

    # # Subir a HuggingFace
    # upload_model(
    #     model_state_dict=model.state_dict(),
    #     repo_id=repo_id,
    #     filename=save_filename,
    # )

    # # Registrar versión
    # register_version(
    #     repo_id=repo_id,
    #     version=version,
    #     filename=save_filename,
    #     base_model=cfg_hf.get("base_model", save_filename),
    #     sar_mode=sar_mode,
    #     phase1_info={
    #         "num_epochs": num_epochs, "lr": lr,
    #         "T": T, "sigmoid": sigmoid_k, "alpha_min": alpha_min,
    #         "batch_size": batch_size, "num_weights": parameters,
    #         "sar_mode": sar_mode,
    #     },
    #     phase2_info={
    #         "Nada": 0
    #     },
    #     notes=notes,
    # )

    # # Guardar history local
    # history_data = {
    #     "train_loss": history["train_loss"],
    #     "config": {
    #         "num_epochs": num_epochs,
    #         "lr": lr,
    #         "batch_size": batch_size,
    #         "T": T,
    #         "beta_schedule": "sigmoid",
    #         "sar_mode": sar_mode,
    #         "num_parameters": parameters,
    #     }
    # }

    # history_dir = ROOT / "training_history"
    # history_dir.mkdir(parents=True, exist_ok=True)

    # history_filename = save_filename.replace(".pth", "_history.yaml")
    # history_path     = history_dir / history_filename

    # with open(history_path, "w") as f:
    #     yaml.safe_dump(history_data, f, sort_keys=False)

    # print(f"History guardado en: {history_path}")