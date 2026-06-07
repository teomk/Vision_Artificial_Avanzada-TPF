from pathlib import Path
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def stretch_rgb(img, p_low=2, p_high=98):
    """
    img: [H, W, 3], valores preferentemente en reflectancia [0, 1].
    Hace stretch por canal solo para visualizar.
    """
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(img, dtype=np.float32)

    for c in range(3):
        band = img[:, :, c]
        p1, p2 = np.percentile(band, p_low), np.percentile(band, p_high)
        out[:, :, c] = np.clip((band - p1) / (p2 - p1 + 1e-8), 0, 1)

    return out


def stretch_gray(band, p_low=2, p_high=98):
    """
    band: [H, W]
    """
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
    p1, p2 = np.percentile(band, p_low), np.percentile(band, p_high)
    return np.clip((band - p1) / (p2 - p1 + 1e-8), 0, 1)


def read_rgb(raw, indexes_1based):
    """
    raw: [C, H, W]
    indexes_1based: tupla/lista con índices rasterio, por ejemplo (4, 3, 2)
    """
    idx0 = [i - 1 for i in indexes_1based]
    rgb = raw[idx0].transpose(1, 2, 0)
    return stretch_rgb(rgb)


def print_metadata(path):
    with rasterio.open(path) as src:
        print("\n" + "=" * 80)
        print("ARCHIVO")
        print("=" * 80)
        print(path)
        print("\nCantidad de bandas:", src.count)
        print("Tamaño:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Dtypes:", src.dtypes)
        print("Nodata:", src.nodatavals)

        print("\nDescriptions por banda:")
        for i, desc in enumerate(src.descriptions, start=1):
            print(f"  Banda rasterio {i:02d}: {desc}")

        print("\nColor interpretation:")
        for i, ci in enumerate(src.colorinterp, start=1):
            print(f"  Banda rasterio {i:02d}: {ci}")

        print("\nTags generales:")
        tags = src.tags()
        if tags:
            for k, v in tags.items():
                print(f"  {k}: {v}")
        else:
            print("  Sin tags generales.")

        print("\nTags por banda:")
        for i in range(1, src.count + 1):
            tags_i = src.tags(i)
            print(f"  Banda rasterio {i:02d}: {tags_i if tags_i else 'sin tags'}")


def print_band_stats(path, scale=10000.0):
    with rasterio.open(path) as src:
        raw = src.read().astype(np.float32)

    scaled = raw / scale

    print("\n" + "=" * 80)
    print("ESTADÍSTICAS POR BANDA")
    print("=" * 80)
    print(f"Valores mostrados dividiendo por scale={scale}")
    print()

    header = (
        f"{'Rasterio':>8} | {'min':>10} | {'p02':>10} | {'mean':>10} | "
        f"{'p50':>10} | {'p98':>10} | {'max':>10} | {'std':>10}"
    )
    print(header)
    print("-" * len(header))

    for i in range(scaled.shape[0]):
        band = scaled[i]
        finite = band[np.isfinite(band)]

        if finite.size == 0:
            print(f"{i + 1:8d} | sin valores finitos")
            continue

        mn = finite.min()
        p02 = np.percentile(finite, 2)
        mean = finite.mean()
        p50 = np.percentile(finite, 50)
        p98 = np.percentile(finite, 98)
        mx = finite.max()
        std = finite.std()

        print(
            f"{i + 1:8d} | {mn:10.4f} | {p02:10.4f} | {mean:10.4f} | "
            f"{p50:10.4f} | {p98:10.4f} | {mx:10.4f} | {std:10.4f}"
        )


def save_visual_report(path, out_path, scale=10000.0):
    with rasterio.open(path) as src:
        raw = src.read().astype(np.float32)

    raw = raw / scale
    n_bands, h, w = raw.shape

    # Combinaciones candidatas.
    # Se usan índices rasterio, es decir 1-based.
    rgb_candidates = []

    if n_bands >= 4:
        rgb_candidates.append(
            ("RGB Sentinel-2 estándar: B4 B3 B2", (4, 3, 2))
        )

    if n_bands >= 3:
        rgb_candidates.append(
            ("Lo que verías si usás bandas 1,2,3 como B2,B3,B4: B3 B2 B1", (3, 2, 1))
        )

    if n_bands >= 8:
        rgb_candidates.append(
            ("Falso color vegetación: B8 B4 B3", (8, 4, 3))
        )

    if n_bands >= 13:
        rgb_candidates.append(
            ("SWIR falso color: B12 B8 B4", (13, 8, 4))
        )

    # Figura 1: candidatos RGB
    n_rgb = len(rgb_candidates)
    fig, axes = plt.subplots(1, n_rgb, figsize=(5 * n_rgb, 5))

    if n_rgb == 1:
        axes = [axes]

    for ax, (title, idxs) in zip(axes, rgb_candidates):
        img = read_rgb(raw, idxs)
        ax.imshow(img)
        ax.set_title(f"{title}\nÍndices rasterio {idxs}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_rgb = out_path.with_name(out_path.stem + "_rgb_candidates.png")
    fig.savefig(out_rgb, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figura 2: todas las bandas en escala de grises
    cols = 4
    rows = int(np.ceil(n_bands / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n_bands):
        ax = axes[i]
        ax.imshow(stretch_gray(raw[i]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Banda rasterio {i + 1}", fontsize=10)
        ax.axis("off")

    for j in range(n_bands, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_bands = out_path.with_name(out_path.stem + "_all_bands_gray.png")
    fig.savefig(out_bands, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 80)
    print("REPORTES VISUALES GUARDADOS")
    print("=" * 80)
    print("RGB candidates:", out_rgb)
    print("Todas las bandas:", out_bands)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Ruta a un .tif Sentinel-2 clear o cloudy."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Ruta base de salida. Si no se indica, guarda junto al script con nombre inspect_s2."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10000.0,
        help="Factor de escala para reflectancia Sentinel-2. Default: 10000."
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    if args.out is None:
        out_path = Path("inspect_s2.png")
    else:
        out_path = Path(args.out)

    print_metadata(path)
    print_band_stats(path, scale=args.scale)
    save_visual_report(path, out_path, scale=args.scale)


if __name__ == "__main__":
    main()
