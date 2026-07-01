from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RANKING_FILE = ROOT / "tools" / "dbcr_complex_v7_ranking" / "outputs" / "dbcr_complex_v7_test_psnr_ranking.json"

from visualize_dbcr_complex_v7 import load_models, load_ranking_entry, plot_sample


def resolve_save_path(rank_number: int, output_dir: Path) -> Path:
    return output_dir / f"dbcr_complex_v7_rank{rank_number}.png"


def main():
    parser = argparse.ArgumentParser(
        description="Recorre todos los ranks del ranking v7, muestra cada figura y pregunta si querés guardarla."
    )
    parser.add_argument("--ranking-file", default=str(DEFAULT_RANKING_FILE))
    parser.add_argument("--start-rank", type=int, default=1, help="Primer rank a revisar (1-based).")
    parser.add_argument("--end-rank", type=int, default=None, help="Último rank a revisar (1-based).")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "visualize" / "outputs"),
        help="Directorio donde se guardan las figuras si elegís guardarlas.",
    )
    args = parser.parse_args()

    ranking_path = Path(args.ranking_file)
    output_dir = Path(args.output_dir)

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

    try:
        for rank_number in range(start_rank, end_rank + 1):
            entry = load_ranking_entry(ranking_path, rank_number)
            print(f"\n=== Rank {rank_number}/{total_ranks} | PSNR={entry['psnr']:.4f} ===")
            print(f"S1        : {entry['paths']['s1']}")
            print(f"S2_CLOUDY : {entry['paths']['s2_cloudy']}")
            print(f"S2_CLEAN  : {entry['paths']['s2']}")
            print(f"MASK      : {entry['paths']['mask']}")

            fig = plot_sample(entry, models=models)

            save_answer = input("Guardar figura? [s/n/q]: ").strip().lower()
            if save_answer in {"q", "quit", "salir"}:
                print("Corte pedido por teclado.")
                plt.close(fig)
                break

            if save_answer in {"s", "si", "sí", "y", "yes"}:
                save_path = resolve_save_path(rank_number, output_dir)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, bbox_inches="tight", dpi=150)
                print(f"Figura guardada en {save_path}")
            else:
                print("Figura no guardada.")

            plt.close(fig)

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")


if __name__ == "__main__":
    main()