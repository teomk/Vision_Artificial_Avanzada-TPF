from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

sys.path.append(str(DATA_DIR))
sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dbcr_complex import DBCR
from dbcr_utils import inference
from evaluate_utils import psnr

from path_dataset import PathAwareSEN12MSCRDataset


def load_config(config_path: Path) -> dict:
    if not config_path.exists() and not config_path.is_absolute():
        candidate = ROOT / "configs" / config_path
        if candidate.exists():
            config_path = candidate

    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de config: {config_path}. "
            f"Probá con '{ROOT / 'configs' / 'dbcr_complex.yaml'}' o con el path completo."
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoint(repo_id: str, filename: str, map_location: torch.device):
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _as_int(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"none", "null", ""}:
            return None
        return int(value)
    return int(value)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "1", "yes", "y"}:
            return True
        if stripped in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def build_model(cfg: dict, checkpoint: dict, device: torch.device) -> DBCR:
    model_args = cfg["model_args"]
    model = DBCR(
        image_channels=_as_int(model_args["image_channels"]),
        condition_channels=_as_int(model_args["condition_channels"]),
        sar_channels=_as_int(model_args["sar_channels"]),
        base_channels=_as_int(model_args["base_channels"]),
        time_dim=_as_int(model_args["time_dim"]),
        num_heads=_as_int(model_args["num_heads"]),
        window_size_sf0=_as_int(model_args["window_size_sf0"]),
        window_size_not_sf0=_as_int(model_args.get("window_size_not_sf0")),
        use_checkpoint=_as_bool(model_args["use_checkpoint"]),
        include_encoder_4=_as_bool(model_args["include_encoder_4"]),
    ).to(device)

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def extract_batch_item(batch_value, index: int):
    if torch.is_tensor(batch_value):
        return batch_value[index : index + 1]
    if isinstance(batch_value, list):
        return batch_value[index]
    if isinstance(batch_value, tuple):
        return batch_value[index]
    if isinstance(batch_value, dict):
        return {key: extract_batch_item(value, index) for key, value in batch_value.items()}
    return batch_value


def evaluate_samples(model, loader, sar_mode: str, device: torch.device, T: int, steps: int, sigmoid_k: float):
    rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, total=len(loader), desc="Evaluando", unit="batch"), start=1):
            s1 = batch["s1"].to(device)
            s2_cloudy = batch["s2_cloudy"].to(device)
            s2_clean = batch["s2_clear"].to(device)
            paths = batch["paths"]
            keys = batch["key"]

            batch_size = s2_cloudy.shape[0]

            for idx in range(batch_size):
                s1_i = s1[idx : idx + 1]
                cloudy_i = s2_cloudy[idx : idx + 1]
                clean_i = s2_clean[idx : idx + 1]
                sample_paths = extract_batch_item(paths, idx)
                sample_key = extract_batch_item(keys, idx)

                if sar_mode == "None":
                    condition = cloudy_i
                    sar = None
                elif sar_mode == "Concat":
                    condition = torch.cat([cloudy_i, s1_i], dim=1)
                    sar = None
                elif sar_mode == "ControlNet":
                    condition = cloudy_i
                    sar = s1_i
                else:
                    raise ValueError(f"sar_mode desconocido: '{sar_mode}'")

                pred = inference(
                    model,
                    cloudy_i,
                    condition,
                    device,
                    T=T,
                    steps=steps,
                    sar=sar,
                    sigmoid_k=sigmoid_k,
                    show_progress=False,
                ).clamp(0, 1)

                sample_psnr = psnr(pred, clean_i)
                rows.append(
                    {
                        "key": sample_key,
                        "psnr": float(sample_psnr),
                        "paths": sample_paths,
                    }
                )

            # tqdm.write(f"Batch {batch_idx}/{len(loader)} procesado")

    rows.sort(key=lambda item: item["psnr"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def save_outputs(rows, output_dir: Path, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "dbcr_complex_v7_test_psnr_ranking.json"
    txt_path = output_dir / "dbcr_complex_v7_test_psnr_ranking.txt"
    csv_path = output_dir / "dbcr_complex_v7_test_psnr_ranking.csv"

    payload = {"metadata": metadata, "samples": rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    with open(txt_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"{row['rank']:04d}\tPSNR={row['psnr']:.4f}\tS2_CLEAN={row['paths']['s2']}\tS2_CLOUDY={row['paths']['s2_cloudy']}\tS1={row['paths']['s1']}\n"
            )

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "key", "psnr", "s1", "s2", "s2_cloudy", "mask"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "key": row["key"],
                    "psnr": f"{row['psnr']:.6f}",
                    "s1": row["paths"]["s1"],
                    "s2": row["paths"]["s2"],
                    "s2_cloudy": row["paths"]["s2_cloudy"],
                    "mask": row["paths"]["mask"],
                }
            )

    return json_path, txt_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Rank test samples by PSNR for DBCR Complex v7.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "dbcr_complex.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--filename", default=None, help="Checkpoint filename in HuggingFace. Defaults to resume_filename from config.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    sar_mode = cfg["sar_mode"]
    repo_id = cfg["huggingface"]["repo_id"]
    filename = args.filename or cfg["train"].get("resume_filename") or "dbcr_complex_v7.pth"

    train_cfg = cfg["train"]
    model_args = cfg["model_args"]
    T = int(train_cfg["T"])
    sigmoid_k = float(train_cfg.get("sigmoid_k", 10.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(repo_id=repo_id, filename=filename, map_location=device)
    model = build_model(cfg, checkpoint, device)

    dataset = PathAwareSEN12MSCRDataset(
        split=args.split,
        base_dir=ROOT,
        total_bands=(int(model_args["image_channels"]) == 13),
    )
    loader = DataLoader(dataset, batch_size=max(1, args.batch_size), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))

    rows = evaluate_samples(model, loader, sar_mode=sar_mode, device=device, T=T, steps=args.steps, sigmoid_k=sigmoid_k)

    metadata = {
        "repo_id": repo_id,
        "filename": filename,
        "split": args.split,
        "sar_mode": sar_mode,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "samples": len(rows),
        "device": str(device),
    }

    output_dir = Path(args.output_dir)
    json_path, txt_path, csv_path = save_outputs(rows, output_dir, metadata)

    print(f"Ranking guardado en: {json_path}")
    print(f"Resumen TXT: {txt_path}")
    print(f"Resumen CSV: {csv_path}")


if __name__ == "__main__":
    main()