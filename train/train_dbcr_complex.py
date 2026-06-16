import sys
import io
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


def fit(model, train_loader, device, sar_mode, optimizer, scheduler,
        start_epoch=0, num_epochs=50, T=1000, sigmoid_k=10.0, history=None,
        repo_id=None, save_filename=None, checkpoint_every=None):
    """
    start_epoch: índice (0-based) de la primera época a correr en esta llamada.
    num_epochs:  cuántas épocas correr en esta llamada (no el total acumulado).
    history:     dict previo a continuar, o None para arrancar de cero.
    checkpoint_every: si se especifica (y hay repo_id/save_filename), sube a HF
                       un checkpoint intermedio cada N épocas (resiliencia ante
                       cortes, ya que no se guarda nada en disco local).
    """
    if history is None:
        history = {"train_loss": []}

    last_epoch = start_epoch - 1  # por si num_epochs == 0

    for epoch in range(start_epoch, start_epoch + num_epochs):

        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}", unit="batch")

        for batch in progress_bar:
            loss = run(model=model, batch=batch, optimizer=optimizer, device=device,
                       sar_mode=sar_mode, T=T, sigmoid_k=sigmoid_k)

            epoch_loss += loss
            num_batches += 1
            avg_loss = epoch_loss / num_batches

            postfix = {"loss": f"{loss:.6f}", "avg_loss": f"{avg_loss:.6f}"}
            if device.type == "cuda":
                postfix["VRAM"] = f"{torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            progress_bar.set_postfix(postfix)
            # break

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)
        scheduler.step(avg_loss)
        last_epoch = epoch

        if device.type == "cuda":
            print(f"  Época {epoch+1}: avg_loss={avg_loss:.6f} | "
                  f"VRAM pico={torch.cuda.max_memory_allocated()/1e9:.2f}GB | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # checkpoint intermedio a HF (sin nada en disco local persistente)
        if checkpoint_every and repo_id and save_filename and (epoch + 1) % checkpoint_every == 0:
            ckpt = build_checkpoint(model, optimizer, scheduler, last_epoch, history)
            upload_checkpoint_to_hf(ckpt, repo_id=repo_id, filename=save_filename)
            print(f"  Checkpoint intermedio subido a HF (época {epoch+1})")

    return history, last_epoch


def build_checkpoint(model, optimizer, scheduler, epoch, history):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,          # última época completada (0-based)
        "history": history,
    }


def upload_checkpoint_to_hf(checkpoint, repo_id, filename):
    """
    Sube el checkpoint completo a HF SIN escribir nada a disco local.
    Usa torch.save sobre un buffer en memoria (io.BytesIO) y entrega
    los bytes a upload_model. Esto asume que upload_model (en hf_utils.py)
    sabe aceptar bytes/buffer ademas de un state_dict; si tu implementacion
    actual sólo acepta un state_dict y hace torch.save a un path interno,
    hay que ajustar hf_utils.py (ver nota más abajo).
    """
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    upload_model(model_state_dict=buffer, repo_id=repo_id, filename=filename)


def download_checkpoint_from_hf(repo_id, filename, map_location):
    """
    Descarga un checkpoint completo desde HF. download_model ya devuelve
    el objeto deserializado (asumiendo que internamente hace torch.load),
    así que esto funciona igual tanto para un state_dict "plano" (modelos
    viejos) como para un checkpoint completo (dict con varias claves).
    """
    return download_model(repo_id=repo_id, filename=filename, map_location=map_location)


if __name__ == "__main__":
    # python train/train_dbcr_complex.py --config configs/dbcr_complex.yaml

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

    # load_filename: carga SOLO pesos (transfer/finetune, arranca optimizer/epoch de cero)
    load_filename = cfg_train.get("load_filename")
    load_filename = None if load_filename in (None, "None") else load_filename

    # resume_filename: carga checkpoint COMPLETO (model+optimizer+scheduler+epoch+history)
    resume_filename = cfg_train.get("resume_filename")
    resume_filename = None if resume_filename in (None, "None") else resume_filename

    checkpoint_every = cfg_train.get("checkpoint_every")  # opcional, ej: 5

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
    include_encoder_4   = cfg_model.get("include_encoder_4", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | SAR Mode: {sar_mode} | Epochs a correr: {num_epochs}")
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
    ds_train = SEN12MSCRDataset(split="train", include_s1=(sar_mode != "None"), include_mask=False, total_bands=(image_channels == 13))

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    history = {"train_loss": []}
    start_epoch = 0

    if resume_filename is not None:
        # --- Continuar un entrenamiento anterior: model + optimizer + scheduler + epoch + history
        # El history de las épocas anteriores viaja DENTRO del checkpoint de HF (ckpt["history"]),
        # así que no depende de que exista el YAML local. El YAML local que se escribe al final
        # es simplemente una copia legible/respaldo, no la fuente de verdad para resumir.
        ckpt = download_checkpoint_from_hf(repo_id=repo_id, filename=resume_filename, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Checkpoint completo cargado desde HF: {repo_id}/{resume_filename}")
        print(f"Reanudando desde época {start_epoch + 1} (épocas previas: {start_epoch})")

    elif load_filename is not None:
        # --- Cargar solo pesos (finetune / transfer): optimizer y epoch arrancan de cero
        loaded = download_model(repo_id=repo_id, filename=load_filename, map_location=device)
        state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        model.load_state_dict(state_dict, strict=True)
        print(f"Pesos cargados desde HuggingFace: {repo_id}/{load_filename} (optimizer/epoch reiniciados)")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales    : {n_params:.2f} M")
    print(f"Parámetros entrenables: {parameters:,}")
    print(f"Samples train: {len(ds_train)}")
    print(f"Batches por época: {len(loader_train)}")
    print()

    # Entrenar
    history, last_epoch = fit(
        model=model, train_loader=loader_train,
        device=device, sar_mode=sar_mode,
        optimizer=optimizer, scheduler=scheduler,
        start_epoch=start_epoch, num_epochs=num_epochs,
        T=T, sigmoid_k=sigmoid_k, history=history,
        repo_id=repo_id, save_filename=save_filename,
        checkpoint_every=checkpoint_every,
    )

    total_epochs_completadas = last_epoch + 1

    # --- Subir checkpoint COMPLETO a HuggingFace (nada se guarda en disco local) ---
    final_checkpoint = build_checkpoint(model, optimizer, scheduler, last_epoch, history)
    torch.save(final_checkpoint, "saved_models/temp_checkpoint.pth")
    upload_checkpoint_to_hf(final_checkpoint, repo_id=repo_id, filename=save_filename)
    print(f"Checkpoint completo subido a HF: {repo_id}/{save_filename}")
    print(f"Épocas totales acumuladas: {total_epochs_completadas}")

    # Registrar versión
    register_version(
        repo_id=repo_id,
        version=version,
        filename=save_filename,
        model_name="dbcr_complex",
        base_model=save_filename,
        sar_mode=sar_mode,
        phase1_info={
            "num_epochs": total_epochs_completadas, "lr": lr,
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

    # --- History se guarda LOCAL (no a HF) ---
    history_data = {
        "train_loss": history["train_loss"],
        "config": {
            "total_epochs_completadas": total_epochs_completadas,
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