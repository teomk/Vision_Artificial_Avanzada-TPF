from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
UTILS_DIR = ROOT / "utils"

import sys

sys.path.append(str(MODELS_DIR))
sys.path.append(str(UTILS_DIR))

from dbcr_simple import DBCRSimple
from hf_utils import download_model


def _load_checkpoint(args):
    if args.checkpoint is not None:
        return torch.load(args.checkpoint, map_location="cpu")

    if args.config is None:
        raise SystemExit("Debes pasar --checkpoint o --config.")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_id = cfg["huggingface"]["repo_id"]
    filename = cfg["huggingface"]["save_filename"]
    return download_model(repo_id=repo_id, filename=filename, map_location="cpu")


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = ckpt.get(key)
            if isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
                return value

    raise SystemExit("No pude encontrar un state_dict plano en el checkpoint.")


def _build_reference_model(sar_mode: str):
    image_channels = 6
    condition_channels = 8 if sar_mode == "Concat" else 6

    # Los checkpoints viejos de Concat fueron guardados con ControlNet creado,
    # aunque no se usara en el forward.
    use_control_net = sar_mode in {"Concat", "ControlNet"}

    model = DBCRSimple(
        image_channels=image_channels,
        condition_channels=condition_channels,
        base_channels=64,
        time_dim=128,
        control_net=use_control_net,
    )
    return model.state_dict()


def _tensor_diff(a: torch.Tensor, b: torch.Tensor):
    diff = (a - b).abs()
    return {
        "mean_abs": diff.mean().item(),
        "max_abs": diff.max().item(),
        "allclose": torch.allclose(a, b),
    }


def main():
    parser = argparse.ArgumentParser(description="Verifica si un checkpoint DBCR parece entrenado o solo inicializado.")
    parser.add_argument("--config", type=str, help="YAML de config con huggingface.repo_id y huggingface.save_filename")
    parser.add_argument("--checkpoint", type=str, help="Ruta local a un checkpoint .pth")
    parser.add_argument("--seed", type=int, default=1234, help="Seed para reproducir la referencia inicial (default: 1234)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ckpt_raw = _load_checkpoint(args)
    ckpt = _extract_state_dict(ckpt_raw)

    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sar_mode = cfg["sar_mode"]
    else:
        sar_mode = "ControlNet" if any(k.startswith("control_net.") for k in ckpt.keys()) else "None"

    ref_state = _build_reference_model(sar_mode)

    shared_keys = [k for k in ckpt.keys() if k in ref_state and ckpt[k].shape == ref_state[k].shape]
    control_keys = [k for k in shared_keys if k.startswith("control_net.")]
    backbone_keys = [k for k in shared_keys if not k.startswith("control_net.")]
    unexpected_keys = sorted(k for k in ckpt.keys() if k not in ref_state or ckpt[k].shape != ref_state.get(k, ckpt[k]).shape)

    print(f"SAR mode detectado: {sar_mode}")
    print(f"Claves en checkpoint: {len(ckpt)}")
    print(f"Claves compartidas con la referencia: {len(shared_keys)}")
    print(f"Claves inesperadas o con shape distinta: {len(unexpected_keys)}")

    if unexpected_keys:
        print("Primeras claves inesperadas:")
        for key in unexpected_keys[:10]:
            print(f"  - {key}")

    def summarize(group_name, keys):
        if not keys:
            print(f"\n{group_name}: no hay claves comparables.")
            return

        exact = 0
        mean_diffs = []
        max_diffs = []
        for key in keys:
            stats = _tensor_diff(ckpt[key], ref_state[key])
            exact += int(stats["allclose"])
            mean_diffs.append(stats["mean_abs"])
            max_diffs.append(stats["max_abs"])

        print(f"\n{group_name}:")
        print(f"  claves: {len(keys)}")
        print(f"  exactas vs referencia: {exact}")
        print(f"  mean |diff| promedio: {sum(mean_diffs) / len(mean_diffs):.6e}")
        print(f"  max |diff| promedio: {sum(max_diffs) / len(max_diffs):.6e}")

    summarize("Backbone / U-Net", backbone_keys)
    summarize("ControlNet", control_keys)

    if control_keys:
        zero_like = [k for k in control_keys if torch.count_nonzero(ckpt[k]) == 0]
        print(f"\nControlNet con tensor completamente en cero: {len(zero_like)} / {len(control_keys)}")

    print("\nLectura rápida:")
    print("- Si ControlNet sale muy parecido a la referencia, probablemente no aprendió.")
    print("- Si ControlNet muestra diferencias grandes en la mayoría de sus capas, sí cambió durante entrenamiento.")
    print("- Para una prueba exacta necesitas el checkpoint de inicialización usado antes de entrenar.")


if __name__ == "__main__":
    main()