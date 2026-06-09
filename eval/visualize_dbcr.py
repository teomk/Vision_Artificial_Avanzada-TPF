import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dataset import SEN12MSCRDataset

from dbcr_no_sar_naf import DBCRNoSARNAF

from hf_utils import (
    download_model,
    resolve_load_version,
)

def to_rgb(tensor, bands=(2, 1, 0)):
    """
    Dataset:
    [B2, B3, B4, B8, B11, B12]

    RGB = B4,B3,B2 = índices 2,1,0
    """

    img = tensor[
        [bands[0], bands[1], bands[2]]
    ].permute(1, 2, 0).cpu().numpy()

    img = np.clip(img, 0, 1)

    out = np.zeros_like(img)

    for c in range(3):
        p2 = np.percentile(img[:, :, c], 2)
        p98 = np.percentile(img[:, :, c], 98)

        out[:, :, c] = np.clip(
            (img[:, :, c] - p2) / (p98 - p2 + 1e-8),
            0,
            1,
        )

    return out


def visualize_samples(
    model,
    dataset,
    device,
    n_samples=4,
    T=1000,
    save_path=None,
    seed=17,
):

    model.eval()
    np.random.seed(seed)

    indices = np.random.choice(
        len(dataset),
        n_samples,
        replace=False,
    )

    n_cols = 3

    col_labels = [
        "Cloudy",
        "Prediction",
        "Ground Truth"
    ]

    fig = plt.figure(
        figsize=(4 * n_cols, 4 * n_samples)
    )

    gs = gridspec.GridSpec(
        n_samples,
        n_cols,
        figure=fig,
        hspace=0.05,
        wspace=0.05,
    )

    with torch.no_grad():

        for row, idx in enumerate(indices):

            cloudy, clear = dataset[idx]

            cloudy_b = cloudy.unsqueeze(0).float().to(device)

            t = torch.full(
                (1,),
                T,
                device=device,
                dtype=torch.long
            )

            pred = model(
                x_t=cloudy_b,
                t=t,
                s2_cloudy=cloudy_b
            )

            pred = pred.squeeze(0).clamp(0, 1).cpu()

            images = [
                to_rgb(cloudy),
                to_rgb(pred),
                to_rgb(clear)
            ]

            for col, img in enumerate(images):

                ax = fig.add_subplot(gs[row, col])

                ax.imshow(img)

                ax.axis("off")

                if row == 0:
                    ax.set_title(
                        col_labels[col],
                        fontsize=11,
                        pad=6,
                    )

    if save_path is not None:

        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=150,
        )

        print(
            f"Figura guardada en: {save_path}"
        )

    else:
        plt.show()

    plt.close()


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--version",
    #     type=int,
    #     default=None
    # )

    # parser.add_argument(
    #     "--split",
    #     type=str,
    #     default="test"
    # )

    # args = parser.parse_args()

    repo_id = "LucioLuque/lama"

    # _, filename = resolve_load_version(
    #     repo_id=repo_id,
    #     filename_prefix="dbcr_no_sar_naf",
    #     requested_version=args.version,
    # )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    filename = "dbcr_no_sar_naf_v1.pth"

    checkpoint = download_model(
        repo_id=repo_id,
        filename=filename,
        map_location=device,
    )

    model = DBCRNoSARNAF(
        image_channels=6,
        condition_channels=6,
        base_channels=64,
        time_dim=128,
    )

    model.load_state_dict(checkpoint)

    model = model.float().to(device)

    ds = SEN12MSCRDataset(
        split="test",
        include_s1=False,
        include_mask=False,
    )

    visualize_samples(
        model=model,
        dataset=ds,
        device=device,
        n_samples=4,
        T=1000,
    )
