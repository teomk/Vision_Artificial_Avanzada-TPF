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
sys.path.append(str(LAMA_DIR))
sys.path.append(str(DATA_DIR))

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from dataset_lama import SEN12MSCRDataset


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

def train(model, loader_train, cfg, use_sar, save_path, device):

    loss_fn = torch.nn.L1Loss()
    history = {"train_first_epochs": [], "train_full_epochs": []}

    epochs_frozen = int(cfg.get("epochs_frozen", 3))
    epochs_full   = int(cfg.get("epochs_full",   10))
    lr_frozen     = float(cfg.get("lr_frozen",   1e-3))
    lr_full       = float(cfg.get("lr_full",     1e-4))

    modified = {"model.1.ffc.convl2l.weight", "model.34.weight", "model.34.bias"}

    for name, param in model.named_parameters():
        param.requires_grad = name in modified

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Parámetros entrenables primeras épocas: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.Adam(trainable, lr=lr_frozen)

    history["train_first_epochs"] = _run_epochs(
        model, loader_train, optimizer, loss_fn,
        epochs_frozen, phase=1, use_sar=use_sar, device=device)

    # # ── Fase 2: todas las capas ────────────────────────────────────────
    # print(f"\n{'='*50}")
    # print(f"FASE 2: Entrenando todas las capas ({epochs_full} épocas)")
    # print(f"{'='*50}")

    trainable_phase2 = {
        "model.1.ffc.convl2l.weight",   # primera conv (modificada)
        "model.34.weight",               # última conv  (modificada)
        "model.34.bias",
        "model.24", "model.25",          # primer upsample
        "model.27", "model.28",          # segundo upsample
        "model.30", "model.31",          # tercer upsample
    }

    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(k) for k in trainable_phase2)

    # for param in model.parameters():
    #     param.requires_grad = True

    # Contar solo parámetros marcados como entrenables (requires_grad=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    print(f"Parámetros totales: {total_params:,} | entrenables: {trainable_params:,}")
    optimizer = torch.optim.Adam(trainable, lr=lr_full)

    history["train_full_epochs"] = _run_epochs(
        model, loader_train, optimizer, loss_fn,
        epochs_full, phase=2, use_sar=use_sar, device=device)

    # ── Guardar modelo ─────────────────────────────────────────────────
    torch.save(model.state_dict(), save_path)
    print(f"\nModelo guardado en: {save_path}")

    return history


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
            # #por ahora un solo batch por época para probar que corre
            break

        train_loss /= len(loader_train)
        epoch_losses.append(train_loss)

    return epoch_losses


if __name__ == "__main__":

    # python train/train_lama.py --config configs/lama_no_sar.yaml
    # python train/train_lama.py --config configs/lama_sar.yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_train = cfg["train"]

    use_sar    = cfg.get("sar", False)
    model_path = ROOT / cfg_train["model_path"]
    save_path  = ROOT / cfg_train["save_path"]
    batch_size = cfg_train.get("batch_size", 4)
    num_workers= cfg_train.get("num_workers", 2)

    if save_path.exists():
        print(f"El archivo de destino ya existe: {save_path}")
        overwrite = input("¿Desea sobrescribirlo? (s/n): ").strip().lower()
        if overwrite != 's':
            print("Operación cancelada por el usuario.")
            sys.exit(0)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {device} | SAR: {use_sar}")

    model = build_model(use_sar=use_sar)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model = model.to(device)

    ds_train = SEN12MSCRDataset(split="train", include_s1=use_sar)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers)

    history = train(model, loader_train, cfg,
          use_sar=use_sar, save_path=str(save_path), device=device)