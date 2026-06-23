import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
sys.path.append(str(DATA_DIR))

from dataset import SEN12MSCRDataset


# ─────────────────────────────────────────
# Estadísticas usadas en tu Dataset.py
# Sirven para desnormalizar S1 desde z-score a dB aprox.
# ─────────────────────────────────────────

S1_MEAN = np.array(
    [-8.999908447265625, -14.78221321105957],
    dtype=np.float32
)

S1_STD = np.array(
    [2.413282871246338, 2.3029115200042725],
    dtype=np.float32
)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _as_numpy(array_like):
    """
    Convierte tensores de PyTorch o arrays a numpy.
    """
    if isinstance(array_like, np.ndarray):
        return array_like
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def unnormalize_s1(s1):
    """
    Convierte S1 desde z-score a valores aproximados originales en dB.

    Tu Dataset hace:
        s1 = (s1_raw - mean) / std

    Entonces acá invertimos:
        s1_raw = s1 * std + mean
    """
    s1 = _as_numpy(s1).astype(np.float32)

    mean = S1_MEAN[:, None, None]
    std = S1_STD[:, None, None]

    return s1 * std + mean


def norm_fixed(x, vmin, vmax):
    """
    Normalización fija a [0, 1].
    Esto evita que cada patch tenga un contraste distinto.
    """
    x = x.astype(np.float32)
    return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)


def norm_percentile(x, p_low=2, p_high=98):
    """
    Normalización por percentiles.
    Útil para exploración, pero puede acentuar el speckle.
    """
    x = x.astype(np.float32)
    p1, p2 = np.percentile(x, p_low), np.percentile(x, p_high)
    return np.clip((x - p1) / (p2 - p1 + 1e-8), 0, 1)


# ─────────────────────────────────────────
# Visualizaciones SAR recomendadas
# ─────────────────────────────────────────

def to_sar_gray_db(s1, band=0, use_percentiles=False):
    """
    Visualiza VV o VH en escala de grises usando valores en dB.

    band=0 → VV
    band=1 → VH
    """
    s1_db = unnormalize_s1(s1)

    img = s1_db[band]

    if use_percentiles:
        return norm_percentile(img)

    if band == 0:
        # Rango típico para VV en dB
        return norm_fixed(img, vmin=-25, vmax=0)
    else:
        # Rango típico para VH en dB
        return norm_fixed(img, vmin=-32, vmax=-5)


def to_sar_rgb_db(s1, use_percentiles=False):
    """
    Falso color SAR estable.

    R = VV
    G = VH
    B = VV - VH

    Importante:
    Si los valores están en dB, el ratio VV/VH se representa como diferencia:
        ratio_db = VV_dB - VH_dB

    No conviene hacer vv / vh sobre z-score.
    """
    s1_db = unnormalize_s1(s1)

    vv = s1_db[0]
    vh = s1_db[1]
    ratio_db = vv - vh

    if use_percentiles:
        R = norm_percentile(vv)
        G = norm_percentile(vh)
        B = norm_percentile(ratio_db)
    else:
        R = norm_fixed(vv, vmin=-25, vmax=0)
        G = norm_fixed(vh, vmin=-32, vmax=-5)
        B = norm_fixed(ratio_db, vmin=-5, vmax=15)

    return np.stack([R, G, B], axis=-1)


def to_sar_green_db(s1, use_percentiles=False):
    """
    Visualización tipo 'verde' para SAR.

    R = 0
    G = VV
    B = VH

    Sirve para ver estructura, humedad y textura sin inventar color óptico.
    """
    s1_db = unnormalize_s1(s1)

    vv = s1_db[0]
    vh = s1_db[1]

    if use_percentiles:
        G = norm_percentile(vv)
        B = norm_percentile(vh)
    else:
        G = norm_fixed(vv, vmin=-25, vmax=0)
        B = norm_fixed(vh, vmin=-32, vmax=-5)

    R = np.zeros_like(G)

    return np.stack([R, G, B], axis=-1)


def to_sar_ratio_db_gray(s1, use_percentiles=False):
    """
    Visualización en gris del ratio VV/VH expresado en dB.

    En dB:
        ratio = VV - VH
    """
    s1_db = unnormalize_s1(s1)

    vv = s1_db[0]
    vh = s1_db[1]
    ratio_db = vv - vh

    if use_percentiles:
        return norm_percentile(ratio_db)

    return norm_fixed(ratio_db, vmin=-5, vmax=15)


# ─────────────────────────────────────────
# Visualización RGB de Sentinel-2 para comparar
# ─────────────────────────────────────────

def to_s2_rgb(s2):
    """
    Convierte S2 [6, H, W] a RGB usando las bandas que estás cargando:
    BANDS = [2, 3, 4, 8, 12, 13]

    Eso significa:
        índice 0 → B2 azul
        índice 1 → B3 verde
        índice 2 → B4 rojo

    RGB visual:
        R = B4
        G = B3
        B = B2
    """
    s2 = _as_numpy(s2).astype(np.float32)

    blue = s2[0]
    green = s2[1]
    red = s2[2]

    rgb = np.stack([red, green, blue], axis=-1)

    # Para óptico suele verse mejor con percentiles.
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

    return rgb


# ─────────────────────────────────────────
# Exploración de una muestra
# ─────────────────────────────────────────

def explore_sar(dataset, idx=0, use_percentiles=False):
    """
    Explora una muestra del dataset.

    Si use_percentiles=False:
        usa rangos fijos en dB.
        Es lo más recomendable para comparar muestras.

    Si use_percentiles=True:
        estira cada imagen por percentiles.
        Puede verse más contrastado, pero también más ruidoso.
    """
    sample = dataset[idx]

    # Como instanciamos include_s1=True e include_mask=False:
    # sample = (s1, s2_cloudy, s2_clear)
    s1, cloudy, clear = sample

    s1_np = _as_numpy(s1)
    s1_db = unnormalize_s1(s1)

    print(f"\nMuestra idx={idx}")
    print(f"SAR normalizado shape: {s1_np.shape}")
    print(
        f"VV z-score — min={s1_np[0].min():.4f} "
        f"max={s1_np[0].max():.4f} "
        f"mean={s1_np[0].mean():.4f}"
    )
    print(
        f"VH z-score — min={s1_np[1].min():.4f} "
        f"max={s1_np[1].max():.4f} "
        f"mean={s1_np[1].mean():.4f}"
    )

    print(
        f"VV dB aprox — min={s1_db[0].min():.4f} "
        f"max={s1_db[0].max():.4f} "
        f"mean={s1_db[0].mean():.4f}"
    )
    print(
        f"VH dB aprox — min={s1_db[1].min():.4f} "
        f"max={s1_db[1].max():.4f} "
        f"mean={s1_db[1].mean():.4f}"
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(to_sar_gray_db(s1, band=0, use_percentiles=use_percentiles), cmap="gray")
    axes[0, 0].set_title("SAR VV en dB")

    axes[0, 1].imshow(to_sar_gray_db(s1, band=1, use_percentiles=use_percentiles), cmap="gray")
    axes[0, 1].set_title("SAR VH en dB")

    axes[0, 2].imshow(to_sar_ratio_db_gray(s1, use_percentiles=use_percentiles), cmap="gray")
    axes[0, 2].set_title("Ratio dB: VV - VH")

    axes[0, 3].imshow(to_sar_rgb_db(s1, use_percentiles=use_percentiles))
    axes[0, 3].set_title("Falso RGB: VV / VH / VV-VH")

    axes[1, 0].imshow(to_sar_green_db(s1, use_percentiles=use_percentiles))
    axes[1, 0].set_title("SAR verde: R=0, G=VV, B=VH")

    axes[1, 1].imshow(to_s2_rgb(cloudy))
    axes[1, 1].set_title("S2 cloudy RGB")

    axes[1, 2].imshow(to_s2_rgb(clear))
    axes[1, 2].set_title("S2 clear RGB")

    # Diferencia visual simple entre cloudy y clear
    diff = np.abs(to_s2_rgb(clear) - to_s2_rgb(cloudy))
    axes[1, 3].imshow(diff)
    axes[1, 3].set_title("|S2 clear - S2 cloudy|")

    for ax in axes.ravel():
        ax.axis("off")

    mode = "percentiles por imagen" if use_percentiles else "rangos fijos en dB"
    plt.suptitle(f"Muestra idx={idx} — visualización SAR con {mode}", fontsize=14)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Índice de la muestra")
    parser.add_argument("--n", type=int, default=1, help="Cuántas muestras explorar seguidas")
    parser.add_argument(
        "--percentiles",
        action="store_true",
        help="Usar normalización por percentiles en lugar de rangos fijos en dB"
    )

    args = parser.parse_args()

    ds = SEN12MSCRDataset(
        split="test",
        include_s1=True,
        include_mask=False
    )

    print(f"Dataset size: {len(ds)}")

    for i in range(args.n):
        explore_sar(
            dataset=ds,
            idx=args.idx + i,
            use_percentiles=args.percentiles
        )