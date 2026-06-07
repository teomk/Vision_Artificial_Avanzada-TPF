from pathlib import Path
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def stretch_rgb(img, p_low=2, p_high=98):
    out = np.zeros_like(img, dtype=np.float32)

    for c in range(3):
        band = img[:, :, c]
        p1, p2 = np.percentile(band, p_low), np.percentile(band, p_high)
        out[:, :, c] = np.clip((band - p1) / (p2 - p1 + 1e-8), 0, 1)

    return out


def read_dataset_style(path, bands):
    """
    Simula lo que hace dataset_lama.py:
    src.read(indexes=BANDS) / 10000
    """
    with rasterio.open(path) as src:
        arr = src.read(indexes=bands).astype(np.float32) / 10000.0

    return arr


def to_rgb_like_your_code(tensor, rgb_order=(2, 1, 0)):
    """
    Simula tu función to_rgb:
    tensor [C, H, W] -> RGB [H, W, 3]
    """
    img = tensor[[rgb_order[0], rgb_order[1], rgb_order[2]]].transpose(1, 2, 0)
    return stretch_rgb(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out", type=str, default="eval/outputs/test_band_selection.png")
    args = parser.parse_args()

    path = Path(args.path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    # Caso A: lo que estás usando ahora en dataset_lama.py
    # Tensor interno: [B1, B2, B3, B7, B11, B12]
    # Visualizado con (2,1,0): [B3, B2, B1]
    current_bands = [1, 2, 3, 7, 12, 13]
    current_tensor = read_dataset_style(path, current_bands)
    current_rgb = to_rgb_like_your_code(current_tensor, rgb_order=(2, 1, 0))

    # Caso B: propuesta corregida para 6 bandas
    # Tensor interno: [B2, B3, B4, B8, B11, B12]
    # Visualizado con (2,1,0): [B4, B3, B2]
    proposed_bands = [2, 3, 4, 8, 12, 13]
    proposed_tensor = read_dataset_style(path, proposed_bands)
    proposed_rgb = to_rgb_like_your_code(proposed_tensor, rgb_order=(2, 1, 0))

    # Caso C: prueba mínima que pediste: solo B2, B3, B4
    # Tensor interno: [B2, B3, B4]
    # Visualizado con (2,1,0): [B4, B3, B2]
    only_rgb_bands = [2, 3, 4]
    only_rgb_tensor = read_dataset_style(path, only_rgb_bands)
    only_rgb = to_rgb_like_your_code(only_rgb_tensor, rgb_order=(2, 1, 0))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(current_rgb)
    axs[0].set_title(
        "Actual dataset_lama.py\n"
        "BANDS=[1,2,3,7,12,13]\n"
        "Se ve como B3,B2,B1"
    )
    axs[0].axis("off")

    axs[1].imshow(proposed_rgb)
    axs[1].set_title(
        "Propuesta 6 bandas\n"
        "BANDS=[2,3,4,8,12,13]\n"
        "Se ve como B4,B3,B2"
    )
    axs[1].axis("off")

    axs[2].imshow(only_rgb)
    axs[2].set_title(
        "Prueba mínima RGB\n"
        "BANDS=[2,3,4]\n"
        "Se ve como B4,B3,B2"
    )
    axs[2].axis("off")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()