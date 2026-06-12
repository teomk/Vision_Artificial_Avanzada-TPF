import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import sys
import argparse
import yaml
from pathlib import Path
from diffusers import DDPMScheduler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dataset import SEN12MSCRDataset
from hf_utils import download_model
from ddpm_utils import build_sigmoid_ddpm_scheduler
from ddpm import ConditionalDDPMUNet
from dbcr_simple import DBCRSimple
from dbcr_simple_utils import make_bridge_sample, sigmoid_scheduler
from visualize_utils import to_rgb


# ── Schedulers ─────────────────────────────────────────────────────────

def forward_ddpm(clean, scheduler, T, frame_every=100):
    """
    Forward DDPM exacto al train.

    En train:
        noise = torch.randn_like(s2_clean)
        x_t = scheduler.add_noise(s2_clean, noise, t)

    Acá usamos timesteps fijos solo para visualizar.
    """

    device = clean.device

    clean_b = clean.unsqueeze(0)  # [1, 6, H, W]

    # Mismo ruido base para ver una progresión suave
    noise = torch.randn_like(clean_b)

    timesteps = torch.arange(0, T, frame_every, device=device).long()

    frames = []

    for t_val in timesteps:
        if t_val.item() == 0:
            x_t = clean_b.clone()
        else:
            t = t_val.repeat(clean_b.shape[0])
            x_t = scheduler.add_noise(clean_b, noise, t)

        frames.append((t_val.item(), x_t.squeeze(0).clamp(0, 1).detach().cpu()))

    return frames

def forward_bridge(clean, cloudy, T, frame_every=100, sigmoid_k=10.0):
    """
    Forward DBCR exacto al entrenamiento.

    En train:
        alpha_t = sigmoid_scheduler(T, sigmoid_k, t, device)
        x_t = (1 - alpha_t) * s2_clean + alpha_t * s2_cloudy

    No hay ruido gaussiano.
    El 'ruido' es la contribución progresiva de la imagen nubosa.
    """

    device = clean.device

    # clean/cloudy vienen como [6, H, W]
    # Para copiar exactamente el train, agregamos batch:
    clean_b = clean.unsqueeze(0)    # [1, 6, H, W]
    cloudy_b = cloudy.unsqueeze(0)  # [1, 6, H, W]

    timesteps = torch.arange(
        0,
        T + 1,
        frame_every,
        device=device
    ).long()

    frames = []

    for t_val in timesteps:
        t = t_val.view(1)  # [1]

        if t_val.item() == 0:
            # En train no usás t=0 porque sampleás desde 1,
            # pero para visualizar conviene mostrar la imagen limpia exacta.
            x_t_b = clean_b.clone()

        else:
            x_t_b = make_bridge_sample(s2_clean=clean_b, s2_cloudy=cloudy_b, t=t, T=T, sigmoid_k=sigmoid_k, device=device)

        # Guardamos frame sin batch: [6, H, W]
        x_t = x_t_b.squeeze(0).clamp(0, 1).detach().cpu()

        frames.append((t_val.item(), x_t))

    return frames


# ── Reverse processes ───────────────────────────────────────────────────

def reverse_ddpm(model, condition, device, scheduler, T, n_steps, sar=None):
    """
    Proceso reverso DDPM: ruido puro → imagen limpia.
    Retorna lista de imágenes en distintos timesteps intermedios.
    """
    B, _, H, W     = condition.shape
    image_channels = model.out[-1].out_channels
    x_t            = torch.randn(B, image_channels, H, W, device=device)

    scheduler.set_timesteps(n_steps)
    step_indices = np.linspace(0, n_steps - 1, min(n_steps, 8)).astype(int)
    frames = []

    with torch.no_grad():
        for i, t_val in enumerate(scheduler.timesteps):
            t          = t_val.repeat(B).to(device)
            noise_pred = model(x_t=x_t, t=t, s2_cloudy=condition, sar=sar)
            x_t        = scheduler.step(noise_pred, t_val, x_t).prev_sample
            if i in step_indices:
                frames.append((t_val.item(), x_t.squeeze(0).clamp(0, 1).cpu()))

    return frames


def reverse_bridge(model, cloudy_b, clean_gt, condition, device, T, steps=10, sigmoid_k=10.0, sar=None, add_ground_truth=True):
    """
    Reverse DBCR visual copiando exactamente la lógica de eval_dbcr_simple.py.

    En eval:
        x_t = cloudy_b.clone()
        timesteps = torch.linspace(T, 1, steps).long()

        for t_val in timesteps:
            pred_clean = model(x_t, t, condition)
            alpha_prev = sigmoid_scheduler(T, sigmoid_k, t_val - T//steps)
            x_t = (1 - alpha_prev) * pred_clean + alpha_prev * cloudy_b

        return pred_clean

    Shapes:
        cloudy_b   : [B, 6, H, W]
        clean_gt   : [B, 6, H, W]
        condition  : [B, 6 o 8, H, W]
        sar        : [B, 2, H, W] o None
        t          : [B]
        alpha_prev : [B, 1, 1, 1]
        pred_clean : [B, 6, H, W]
        x_t        : [B, 6, H, W]
    """

    model.eval()

    x_t = cloudy_b.clone()  # [B, 6, H, W]

    timesteps = torch.linspace(
        T,
        1,
        steps
    ).long().to(device)

    frames = []

    # Frame inicial: imagen nubosa completa
    frames.append(
        (
            T,
            x_t[0].clamp(0, 1).detach().cpu()
        )
    )

    pred_clean = None

    with torch.no_grad():
        for t_val in timesteps:
            # Igual que en eval_dbcr_simple.py
            t = t_val.repeat(x_t.shape[0]).to(device)  # [B]

            pred_clean = model(x_t=x_t, t=t, s2_cloudy=condition, sar=sar)  # [B, 6, H, W]

            prev_t = (t_val - T // steps).clamp(min=1)

            alpha_prev = sigmoid_scheduler(T, sigmoid_k, prev_t.repeat(x_t.shape[0]).to(device), device)  # [B, 1, 1, 1]

            x_t = ((1.0 - alpha_prev) * pred_clean + alpha_prev * cloudy_b)  # [B, 6, H, W]

            x_t = x_t.clamp(0, 1)

            # Este frame representa el nuevo x_t luego del update
            frames.append((int(prev_t.item()), x_t[0].clamp(0, 1).detach().cpu()))

    # IMPORTANTE:
    # La evaluación retorna pred_clean, no x_t.
    # Por eso conviene mostrarlo explícitamente.
    if pred_clean is not None:
        frames.append(("PRED", pred_clean[0].clamp(0, 1).detach().cpu()))

    if add_ground_truth:
        frames.append(("GT", clean_gt[0].clamp(0, 1).detach().cpu()))

    return frames

# ── Plotting ────────────────────────────────────────────────────────────

def plot_frames(frames, title, fig, gs_row, row_label):
    n_cols = len(frames)
    for col, (t_val, img) in enumerate(frames):
        ax = fig.add_subplot(gs_row[col])
        ax.imshow(to_rgb(img))
        ax.axis("off")
        ax.set_title(f"t={t_val}", fontsize=8, pad=3)
        if col == 0:
            ax.set_ylabel(row_label, fontsize=9, rotation=90, labelpad=6)

    fig.text(
        0.5,
        gs_row[0].get_position(fig).y1 + 0.01,
        title,
        ha="center", fontsize=10, fontweight="bold"
    )


def build_figure(rows):
    """
    rows: lista de (title, row_label, frames)
    """
    n_cols = max(len(f) for _, _, f in rows)
    n_rows = len(rows)

    fig = plt.figure(figsize=(2.5 * n_cols, 3.2 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.05)

    for row_idx, (title, row_label, frames) in enumerate(rows):
        # Pad con frames vacíos si hay menos columnas
        padded = frames + [(None, None)] * (n_cols - len(frames))
        for col, (t_val, img) in enumerate(padded):
            ax = fig.add_subplot(gs[row_idx, col])
            if img is not None:
                ax.imshow(to_rgb(img))
                ax.set_title(f"t={t_val}", fontsize=8, pad=3)
            else:
                ax.set_visible(False)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9)

        # Título de fila
        fig.text(
            0.01,
            gs[row_idx, 0].get_position(fig).y0 + gs[row_idx, 0].get_position(fig).height / 2,
            title,
            va="center", ha="left", fontsize=9, fontweight="bold", rotation=0
        )

    return fig


# ── Sample helpers ──────────────────────────────────────────────────────

def load_sample(dataset, idx, sar_mode, device):
    if sar_mode == "None":
        cloudy, clear = dataset[idx]
        cloudy_b  = cloudy.unsqueeze(0).float().to(device)
        condition = cloudy_b
        sar       = None

    elif sar_mode == "Concat":
        s1, cloudy, clear = dataset[idx]
        cloudy_b  = cloudy.unsqueeze(0).float().to(device)
        s1_b      = s1.unsqueeze(0).float().to(device)
        condition = torch.cat([cloudy_b, s1_b], dim=1)
        sar       = None

    elif sar_mode == "ControlNet":
        s1, cloudy, clear = dataset[idx]
        cloudy_b  = cloudy.unsqueeze(0).float().to(device)
        s1_b      = s1.unsqueeze(0).float().to(device)
        condition = cloudy_b
        sar       = s1_b

    else:
        raise ValueError(f"sar_mode desconocido: '{sar_mode}'")

    clear_b = clear.unsqueeze(0).float().to(device) if not isinstance(clear, torch.Tensor) else clear.unsqueeze(0).to(device)
    return cloudy_b, clear_b, condition, sar


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # python visualize/visualize_noise.py --config configs/ddpm_none.yaml --direction both
    # python visualize/visualize_noise.py --config configs/dbcr_no_sar.yaml --direction forward
    # python visualize/visualize_noise.py --config configs/ddpm_concat.yaml --direction reverse --steps 50

    parser = argparse.ArgumentParser(description="Visualizar proceso de ruido forward/reverse (DDPM o DBCR)")
    parser.add_argument("--config",    type=str, required=True,                              help="Ruta al config YAML")
    parser.add_argument("--direction", type=str, default="both", choices=["forward", "reverse", "both"], help="Proceso a visualizar (default: both)")
    parser.add_argument("--steps",     type=int, default=10,                                 help="Pasos de inferencia (default: 10)")
    parser.add_argument("--frame_every",  type=int, default=100,                                  help="Frames cada n frames (default: 100)")
    parser.add_argument("--idx",       type=int, default=0,                                  help="Índice de muestra del dataset (default: 0)")
    parser.add_argument("--save_path", type=str, default=None,                               help="Ruta para guardar la figura")
    parser.add_argument("--seed",      type=int, default=17,                                 help="Semilla (default: 17)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_type    = cfg["model"]        # "ddpm" o "dbcr"
    sar_mode      = cfg["sar_mode"]
    repo_id       = cfg["huggingface"]["repo_id"]
    save_filename = cfg["huggingface"]["save_filename"]
    T             = cfg["train"]["T"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {model_type} | SAR: {sar_mode} | Direction: {args.direction}")

    # Dataset
    ds = SEN12MSCRDataset(split="test", include_s1=(sar_mode != "None"), include_mask=False)
    cloudy_b, clear_b, condition, sar = load_sample(ds, args.idx, sar_mode, device)

    # # Scheduler DDPM (se usa para forward en ambos modelos, y para reverse en DDPM)
    sigmoid_k = cfg["train"].get("sigmoid_k", 25.0)
    alpha_min = cfg["train"].get("alpha_min", 1e-4)

    scheduler = build_sigmoid_ddpm_scheduler(
        T=T,
        sigmoid_k=sigmoid_k,
        alpha_min=alpha_min
    )

    rows = []

    # ── Forward ──
    if args.direction in ("forward", "both"):
        if model_type == "ddpm":
            fwd_frames = forward_ddpm(clear_b.squeeze(0), scheduler, T, args.frame_every)
            rows.append(("DDPM — Forward (ruido gaussiano progresivo)", "x_t", fwd_frames))

        elif model_type == "dbcr":
            fwd_frames = forward_bridge(clear_b.squeeze(0), cloudy_b.squeeze(0), T, args.frame_every)
            rows.append(("DBCR — Forward (bridge: limpia → nubosa)", "x_t", fwd_frames))

    # ── Reverse (necesita modelo) ──
    if args.direction in ("reverse", "both"):
        checkpoint = download_model(repo_id=repo_id, filename=save_filename, map_location=device)

        image_channels     = 6
        condition_channels = 8 if sar_mode == "Concat" else 6

        if model_type == "ddpm":
            
            model = ConditionalDDPMUNet(
                image_channels=image_channels,
                condition_channels=condition_channels,
                base_channels=64,
                time_dim=256,
            )
            model.load_state_dict(checkpoint)
            model = model.float().to(device)

            rev_frames = reverse_ddpm(model, condition, device, scheduler, T, args.steps, sar=sar)
            rows.append(("DDPM — Reverse (ruido → imagen limpia)", "x_t", rev_frames))

        elif model_type == "dbcr":
            
            model = DBCRSimple(
                image_channels=image_channels,
                condition_channels=condition_channels,
                base_channels=64,
                time_dim=128,
                control_net=(sar_mode == "ControlNet"),
            )
            model.load_state_dict(checkpoint, strict=False)
            model = model.float().to(device)

            rev_frames = reverse_bridge(
                model=model,
                cloudy_b=cloudy_b,
                clean_gt=clear_b,
                condition=condition,
                device=device,
                T=T,
                steps=args.steps,
                sigmoid_k=10,
                sar=sar,
                add_ground_truth=True
            )
            rows.append(("DBCR — Reverse (nubosa → imagen limpia)", "x_t", rev_frames))

    # ── Plot ──
    fig = build_figure(rows)
    fig.suptitle(
        f"{model_type.upper()} | SAR: {sar_mode} | sample idx: {args.idx}",
        fontsize=12, y=1.01
    )

    if args.save_path is not None:
        plt.savefig(args.save_path, bbox_inches="tight", dpi=150)
        print(f"Figura guardada en: {args.save_path}")
    else:
        plt.show()
    plt.close()