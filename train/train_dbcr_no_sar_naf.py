import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

import argparse
from torch.utils.data import DataLoader

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))


from dbcr_no_sar_naf import DBCRNoSARNAF

from hf_utils import (
    download_model,
    upload_model,
    resolve_save_version,
    register_version,
)

from dataset import SEN12MSCRDataset
# from hf_utils import (
#     download_model,
#     upload_model,
#     resolve_save_version,
#     register_version,
# )

def sigmoid_scheduler(T, sigmoid_k, t, device):
    tau = torch.clamp(t.float() / T, 0.0, 1.0)
    s = torch.sigmoid((tau - 0.5) * sigmoid_k)
    s_min = torch.sigmoid(torch.tensor(-0.5 * sigmoid_k, device=device))
    s_max = torch.sigmoid(torch.tensor(0.5 * sigmoid_k, device=device))
    alpha = torch.clamp((s - s_min) / (s_max - s_min), 0.0, 1.0)
    return alpha[:, None, None, None]

def make_bridge_sample(s2_clean, s2_cloudy, t, T, sigmoid_k, device):
    alpha_t = sigmoid_scheduler(T, sigmoid_k, t, device)
    return (1.0 - alpha_t) * s2_clean + alpha_t * s2_cloudy

def run(model, batch, optimizer, device, T=1000, sigmoid_k=10.0):

    model.train()

    s2_cloudy, s2_clean = batch

    s2_cloudy = s2_cloudy.to(device)
    s2_clean  = s2_clean.to(device)

    B = s2_clean.shape[0]

    t = torch.randint(low=1, high=T + 1, size=(B,), device=device)

    x_t = make_bridge_sample(s2_clean=s2_clean, s2_cloudy=s2_cloudy, t=t, T=T, sigmoid_k=sigmoid_k, device=device)

    pred_clean = model(x_t=x_t, t=t, s2_cloudy=s2_cloudy)

    loss = F.l1_loss(pred_clean, s2_clean)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def fit(model, train_loader, lr , device, num_epochs=50, T=1000, sigmoid_k=10.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": []}
    for epoch in range(num_epochs):

        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs}", unit="batch")

        for batch in progress_bar:
            loss = run(model=model, batch=batch, optimizer=optimizer, device=device, T=T, sigmoid_k=sigmoid_k)

            epoch_loss += loss
            num_batches += 1
            avg_loss = epoch_loss / num_batches
            progress_bar.set_postfix({"loss": f"{loss:.6f}", "avg_loss": f"{avg_loss:.6f}"})
            break

        avg_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_loss)

    return history


if __name__ == "__main__":
    

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)
    # args = parser.parse_args()

    # # Cargar configuración
    # import yaml
    # with open(args.config, "r") as f:
    #     config = yaml.safe_load(f)

    # # Configuración de entrenamiento
    # batch_size = config["train"]["batch_size"]
    # num_workers = config["train"]["num_workers"]
    # lr = config["train"]["lr"]
    # num_epochs = config["train"]["num_epochs"]

    batch_size = 4
    num_workers = 2
    lr = 1e-3
    num_epochs = 1

    # Configuración del modelo
    image_channels = 6
    condition_channels = 6
    base_channels = 64
    time_dim = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset y DataLoader
    ds_train = SEN12MSCRDataset(split="train", include_s1=False, include_mask = False)
    loader_train  = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Modelo
    model = DBCRNoSARNAF(image_channels=image_channels, condition_channels=condition_channels,
        base_channels=base_channels, time_dim=time_dim).to(device)
    
    #parameters to train:

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {parameters}")
    
    history = fit(model=model, train_loader=loader_train, lr=lr, device=device, num_epochs=num_epochs)

    # #save .pth in ../saved_models
    # save_path = ROOT / "saved_models" / "dbcr_no_sar_naf_v1.pth"
    # torch.save(model.state_dict(), save_path)
    # print(f"Modelo guardado en: {save_path}")

    repo_id = "LucioLuque/lama"
    save_filename = "dbcr_no_sar_naf_v1.pth"


    upload_model(
        model_state_dict=model.state_dict(),
        repo_id=repo_id,
        filename=save_filename,
    )
    
    version = 1
    base_filename = save_filename
    use_sar = False
    phase1_info = {"num_epochs": num_epochs, "lr": lr, "T": 1000, "sigmoid_k": 10.0, "batch_size": batch_size, "num_weights": parameters}
    phase2_info = {"num_epochs": 0, "lr": 0, "T": 0, "sigmoid_k": 0, "batch_size": 0, "num_weights": 0}

    # ── Registrar versión en versions.yaml ────────────────────────────────────
    register_version(
        repo_id=repo_id,
        version=version,
        filename=save_filename,
        base_model=base_filename,
        use_sar=use_sar,
        phase1_info=phase1_info,
        phase2_info=phase2_info,
        notes="Nada",
    )