from pathlib import Path
import torch
import sys
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT     = Path(__file__).resolve().parent.parent
LAMA_DIR = ROOT / "external" / "lama"
DATA_DIR = ROOT / "dataset"
UTILS_DIR = ROOT / "utils"
sys.path.append(str(LAMA_DIR))
sys.path.append(str(DATA_DIR))
sys.path.append(str(UTILS_DIR))

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from dataset import SEN12MSCRDataset
from hf_utils import (
    download_model,
    upload_model,
    resolve_save_version,
    register_version,
)


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


def train(model, loader_train, cfg, use_sar, device):
    """
    Entrena el modelo en dos fases y devuelve el historial de pérdidas
    junto con la información de cada fase (para registrar en versions.yaml).

    Returns:
        history:     {"train_first_epochs": [...], "train_full_epochs": [...]}
        phase1_info: dict con metadatos de fase 1
        phase2_info: dict con metadatos de fase 2
    """
    loss_fn = torch.nn.L1Loss()
    history = {"train_first_epochs": [], "train_full_epochs": []}

    epochs_frozen = int(cfg["epochs_frozen"])
    epochs_full   = int(cfg["epochs_full"])
    lr_frozen     = float(cfg["lr_frozen"])
    lr_full       = float(cfg["lr_full"])

    print("epocs_frozen:", epochs_frozen)
    print("epocs_full:", epochs_full)

    # ── Fase 1: solo capas modificadas ────────────────────────────────────────
    modified_layers = ["model.1.ffc.convl2l.weight", "model.34.weight", "model.34.bias"]

    for name, param in model.named_parameters():
        param.requires_grad = name in modified_layers

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params_phase1 = sum(p.numel() for p in trainable)
    print(f"Parámetros entrenables primeras épocas: {trainable_params_phase1:,}")

    optimizer = torch.optim.Adam(trainable, lr=lr_frozen)

    history["train_first_epochs"] = _run_epochs(
        model, loader_train, optimizer, loss_fn,
        epochs_frozen, phase=1, use_sar=use_sar, device=device)

    phase1_info = {
        "epochs":           epochs_frozen,
        "lr":               lr_frozen,
        "trainable_params": trainable_params_phase1,
        "trainable_layers": modified_layers,
    }

    # ── Fase 2: capas de upsampling + capas modificadas ───────────────────────
    # trainable_prefixes_phase2 = [
    #     "model.1.ffc.convl2l.weight",
    #     "model.34.weight",
    #     "model.34.bias",
    #     "model.24", "model.25",
    #     "model.27", "model.28",
    #     "model.30", "model.31",
    # ]

    # for name, param in model.named_parameters():
    #     param.requires_grad = any(name.startswith(k) for k in trainable_prefixes_phase2)

    for name, param in model.named_parameters():
        param.requires_grad = True  # Descongelar todo

    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params        = sum(p.numel() for p in model.parameters())
    trainable_params_phase2 = sum(p.numel() for p in trainable)
    print(f"Parámetros totales: {total_params:,} | entrenables: {trainable_params_phase2:,}")

    optimizer = torch.optim.Adam(trainable, lr=lr_full)

    history["train_full_epochs"] = _run_epochs(
        model, loader_train, optimizer, loss_fn,
        epochs_full, phase=2, use_sar=use_sar, device=device)

    phase2_info = {
        "epochs":           epochs_full,
        "lr":               lr_full,
        "trainable_params": trainable_params_phase2,
        # "trainable_layers": trainable_prefixes_phase2,
        "trainable_layers": "todas",
    }

    return history, phase1_info, phase2_info


def _run_epochs(model, loader_train, optimizer, loss_fn, n_epochs, phase, use_sar, device):
    epoch_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(loader_train, desc=f"[Fase {phase}] Época {epoch}/{n_epochs}",
                    unit="batch")

        for batch in pbar:
            x, clear_b = prepare_batch(batch, use_sar, device)

            optimizer.zero_grad()
            output = model(x)
            loss   = loss_fn(output, clear_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")
            #un solo batch para probar:
            # break

        train_loss /= len(loader_train)
        epoch_losses.append(train_loss)

    return epoch_losses


if __name__ == "__main__":

    # Ejemplos de uso:
    #   python train/train_lama.py --config configs/lama_no_sar.yaml
    #   python train/train_lama.py --config configs/lama_sar.yaml
    #   python train/train_lama.py --config configs/lama_no_sar.yaml --version 3
    #   python train/train_lama.py --config configs/lama_no_sar.yaml --notes "prueba lr más bajo"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, required=True,
                        help="Ruta al archivo de configuración YAML")
    parser.add_argument("--version", type=int, default=None,
                        help="Versión a guardar (e.g. 3 → _v3.pth). "
                             "Si no se indica, se auto-incrementa.")
    parser.add_argument("--notes",   type=str, default="",
                        help="Comentario libre que se guardará en versions.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_train = cfg["train"]
    cfg_hf    = cfg["huggingface"]

    use_sar   = cfg["sar"]
    batch_size  = cfg_train["batch_size"]
    num_workers = cfg_train["num_workers"]

    # ── Parámetros HuggingFace ────────────────────────────────────────────────
    repo_id          = cfg_hf["repo_id"]           # e.g. "LucioLuque/lama"
    base_filename    = cfg_hf["base_filename"]      # e.g. "lama_no_sar_pretrained_v1.pth"
    save_prefix      = cfg_hf["save_prefix"]        # e.g. "lama_no_sar_finetuned"

    # ── Resolver versión y nombre de archivo de salida ────────────────────────
    version, save_filename = resolve_save_version(
        repo_id=repo_id,
        filename_prefix=save_prefix,
        requested_version=args.version,
    )

    # ── Setup ──────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {device} | SAR: {use_sar}")
    print(f"Guardará como: {repo_id}/{save_filename}")

    # ── Cargar modelo base desde HuggingFace ──────────────────────────────────
    checkpoint = download_model(
        repo_id=repo_id,
        filename=base_filename,
        map_location=device,
    )

    model = build_model(use_sar=use_sar)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # ── Dataset y DataLoader ──────────────────────────────────────────────────
    ds_train = SEN12MSCRDataset(split="train", include_s1=use_sar)
    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # ── Entrenar ──────────────────────────────────────────────────────────────
    history, phase1_info, phase2_info = train(
        model, loader_train, cfg_train,
        use_sar=use_sar, device=device,
    )

    # ── Subir modelo a HuggingFace ────────────────────────────────────────────
    upload_model(
        model_state_dict=model.state_dict(),
        repo_id=repo_id,
        filename=save_filename,
    )

    # ── Registrar versión en versions.yaml ────────────────────────────────────
    register_version(
        repo_id=repo_id,
        version=version,
        filename=save_filename,
        base_model=base_filename,
        use_sar=use_sar,
        phase1_info=phase1_info,
        phase2_info=phase2_info,
        notes=args.notes,
    )