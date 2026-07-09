from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RANKING_FILE = ROOT / "tools" / "dbcr_complex_v7_ranking" / "outputs" / "dbcr_complex_v7_test_psnr_ranking.json"

from tools.visualize_dbcr_complex_v7 import load_models, load_ranking_entry, plot_sample


def resolve_save_path(rank_number: int, output_dir: Path) -> Path:
    return output_dir / f"dbcr_complex_v7_rank{rank_number}.png"


def main():
    parser = argparse.ArgumentParser(
        description="Procesa todas las muestras del ranking v7, guarda las figuras y escribe un manifiesto para notebook."
    )
    parser.add_argument("--ranking-file", default=str(DEFAULT_RANKING_FILE))
    parser.add_argument("--output-dir", default=str(ROOT / "visualize" / "review_dbcr_complex_v7"))
    parser.add_argument(
        "--manifest",
        default=None,
        help="Ruta del manifiesto JSON. Por defecto se guarda dentro del output-dir.",
    )
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--end-rank", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe figuras existentes.")
    args = parser.parse_args()

    ranking_path = Path(args.ranking_file)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "dbcr_complex_v7_manifest.json"

    data = json.loads(ranking_path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if not samples:
        raise RuntimeError(f"No hay muestras en {ranking_path}")

    total_ranks = len(samples)
    start_rank = max(1, args.start_rank)
    end_rank = args.end_rank if args.end_rank is not None else total_ranks
    end_rank = min(end_rank, total_ranks)

    if start_rank > end_rank:
        raise ValueError(f"start-rank ({start_rank}) no puede ser mayor que end-rank ({end_rank})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    records = []
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for rank_number in range(start_rank, end_rank + 1):
            entry = load_ranking_entry(ranking_path, rank_number)
            save_path = resolve_save_path(rank_number, output_dir)

            print(f"Procesando rank {rank_number}/{total_ranks} | PSNR={entry['psnr']:.4f}")

            if save_path.exists() and not args.overwrite:
                print(f"  Ya existe: {save_path} (se omite)")
                records.append(
                    {
                        "rank": rank_number,
                        "psnr": float(entry["psnr"]),
                        "figure_path": str(save_path),
                        "saved": False,
                        "skipped": True,
                        "key": entry.get("key"),
                        "paths": entry.get("paths", {}),
                    }
                )
                continue

            fig = plot_sample(entry, models=models, show=False)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            records.append(
                {
                    "rank": rank_number,
                    "psnr": float(entry["psnr"]),
                    "figure_path": str(save_path),
                    "saved": True,
                    "skipped": False,
                    "key": entry.get("key"),
                    "paths": entry.get("paths", {}),
                }
            )

        manifest = {
            "ranking_file": str(ranking_path),
            "output_dir": str(output_dir),
            "start_rank": start_rank,
            "end_rank": end_rank,
            "total_ranks": total_ranks,
            "records": records,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"Manifiesto guardado en {manifest_path}")

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")


if __name__ == "__main__":
    main()