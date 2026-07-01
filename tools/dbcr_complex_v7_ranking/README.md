# DBCR Complex V7 PSNR Ranking

Este directorio contiene un evaluador nuevo y aislado para ordenar las muestras del split `test` según PSNR usando `dbcr_complex_v7.pth`.

## Archivos

- `path_dataset.py`: dataset auxiliar que devuelve los tensores y los paths originales de cada muestra.
- `rank_dbcr_complex_v7.py`: script standalone que descarga el checkpoint desde Hugging Face, evalúa cada muestra y guarda el ranking.

## Uso

```bash
python tools/dbcr_complex_v7_ranking/rank_dbcr_complex_v7.py \
  --config configs/dbcr_complex.yaml \
  --split test \
  --batch-size 1
```

## Salidas

El script guarda, dentro de `tools/dbcr_complex_v7_ranking/outputs/`:

- `dbcr_complex_v7_test_psnr_ranking.json`
- `dbcr_complex_v7_test_psnr_ranking.txt`
- `dbcr_complex_v7_test_psnr_ranking.csv`

Cada registro incluye:

- `rank`
- `psnr`
- `s1`
- `s2`
- `s2_cloudy`
- `mask`

El orden del archivo es de mayor a menor PSNR.