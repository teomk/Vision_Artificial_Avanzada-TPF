"""
hf_utils.py
-----------
Utilidades compartidas para interactuar con HuggingFace Hub.
Usado por train_lama.py, eval_lama.py, adapt_lama.py, etc.
"""

from __future__ import annotations

import re
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import torch
import yaml
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    upload_file,
)

VERSIONS_FILENAME = "versions.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Descarga
# ──────────────────────────────────────────────────────────────────────────────

def download_model(
    repo_id: str,
    filename: str,
    *,
    map_location: str = "cpu",
) -> dict:
    """
    Descarga un checkpoint desde HuggingFace y lo carga con torch.

    Args:
        repo_id:      e.g. "LucioLuque/lama"
        filename:     e.g. "lama_no_sar_pretrained_v1.pth"
        map_location: dispositivo destino para torch.load

    Returns:
        State dict listo para model.load_state_dict()
    """
    print(f"Descargando '{filename}' desde '{repo_id}'...")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(path, map_location=map_location)
    print("Descarga OK.")
    return checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Versionado
# ──────────────────────────────────────────────────────────────────────────────

def _list_versions(repo_id: str, prefix: str) -> list[int]:
    """
    Devuelve lista de enteros de versión para archivos que matcheen
    '<prefix>_v<N>.pth' en el repo.

    Args:
        repo_id: e.g. "LucioLuque/lama"
        prefix:  e.g. "lama_no_sar_finetuned"
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.pth$")
    versions = []
    try:
        for f in list_repo_files(repo_id):
            m = pattern.match(f)
            if m:
                versions.append(int(m.group(1)))
    except Exception as e:
        print(f"Advertencia: no se pudieron listar archivos del repo ({e})")
    return sorted(versions)


def resolve_load_version(
    repo_id: str,
    filename_prefix: str,
    requested_version: Optional[int],
) -> tuple[int, str]:
    """
    Determina la versión a cargar y el nombre de archivo resultante.

    - Si requested_version es None → usa la última versión disponible en el repo.
    - Si requested_version está especificada → la valida y la usa.

    Args:
        repo_id:            e.g. "LucioLuque/lama"
        filename_prefix:    e.g. "lama_no_sar_finetuned"
        requested_version:  número entero o None

    Returns:
        (version_int, filename)  e.g. (2, "lama_no_sar_finetuned_v2.pth")

    Raises:
        SystemExit: si no hay versiones disponibles o la solicitada no existe.
    """
    existing = _list_versions(repo_id, filename_prefix)

    if not existing:
        print(f"Error: no se encontraron versiones de '{filename_prefix}' en '{repo_id}'.")
        sys.exit(1)

    if requested_version is None:
        version = max(existing)
        print(f"Versión no especificada. Usando la última disponible: v{version}.")
    else:
        if requested_version not in existing:
            print(
                f"Error: v{requested_version} no existe en '{repo_id}'. "
                f"Versiones disponibles: {existing}"
            )
            sys.exit(1)
        version = requested_version
        print(f"Usando versión solicitada: v{version}.")

    filename = f"{filename_prefix}_v{version}.pth"
    return version, filename


def resolve_save_version(
    repo_id: str,
    filename_prefix: str,
    requested_version: Optional[int],
) -> tuple[int, str]:
    """
    Determina la versión final a guardar y el nombre de archivo resultante.

    - Si requested_version es None → auto-incrementa sobre la última existente.
    - Si requested_version ya existe → pregunta al usuario si sobreescribir.
    - Si requested_version no existe → la usa directamente.

    Args:
        repo_id:            e.g. "LucioLuque/lama"
        filename_prefix:    e.g. "lama_no_sar_finetuned"
        requested_version:  número entero o None

    Returns:
        (version_int, filename)  e.g. (2, "lama_no_sar_finetuned_v2.pth")
    """
    existing = _list_versions(repo_id, filename_prefix)

    if requested_version is None:
        # Auto-incrementar
        version = (max(existing) + 1) if existing else 1
        print(f"Versión no especificada. Se usará v{version} (nueva).")
    else:
        version = requested_version
        if version in existing:
            filename = f"{filename_prefix}_v{version}.pth"
            answer = input(
                f"'{filename}' ya existe en '{repo_id}'. "
                "¿Desea sobreescribirlo? (s/n): "
            ).strip().lower()
            if answer != "s":
                print("Operación cancelada por el usuario.")
                sys.exit(0)
            print(f"Se sobreescribirá v{version}.")
        else:
            print(f"Se usará la versión solicitada: v{version}.")

    filename = f"{filename_prefix}_v{version}.pth"
    return version, filename


# ──────────────────────────────────────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────────────────────────────────────

def upload_model(
    model_state_dict: dict,
    repo_id: str,
    filename: str,
) -> None:
    """
    Guarda el state dict en un archivo temporal y lo sube a HuggingFace.

    Args:
        model_state_dict: resultado de model.state_dict()
        repo_id:          e.g. "LucioLuque/lama"
        filename:         e.g. "lama_no_sar_finetuned_v2.pth"
    """
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save(model_state_dict, tmp_path)
        print(f"Subiendo '{filename}' a '{repo_id}'...")
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Modelo guardado en HuggingFace: {repo_id}/{filename}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Registro de versiones (versions.yaml)
# ──────────────────────────────────────────────────────────────────────────────

def _download_versions_yaml(repo_id: str) -> dict:
    """Descarga el versions.yaml del repo. Si no existe, devuelve dict vacío."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename=VERSIONS_FILENAME)
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        return {}


def _upload_versions_yaml(repo_id: str, data: dict) -> None:
    """Serializa y sube el versions.yaml al repo."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(data, tmp, allow_unicode=True, sort_keys=False)
        tmp_path = tmp.name

    try:
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=VERSIONS_FILENAME,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"versions.yaml actualizado en '{repo_id}'.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _model_key(model_name, sar_mode: str) -> str:
    """Devuelve la clave de primer nivel para el modelo en versions.yaml."""
    # if use_sar:
    #     model_name += "_sar"
    # else:
    #     model_name += "_no_sar"
    return model_name + "_" + sar_mode.lower()


def register_version(
    repo_id: str,
    version: int,
    filename: str,
    *,
    base_model: str,
    sar_mode: str,
    phase1_info: dict,
    phase2_info: dict,
    notes: str = "",
) -> None:
    """
    Agrega (o sobreescribe) una entrada en versions.yaml del repo.

    Estructura resultante en versions.yaml:
    ```yaml
    models:
      lama_no_sar:
        v1:
          filename: lama_no_sar_finetuned_v1.pth
          base_model: lama_no_sar_pretrained_v1.pth
          date: "2026-06-08"
          notes: ""
          phase1:
            epochs: 10
            lr: 0.001
            trainable_params: 12345
            trainable_layers: [...]
          phase2:
            epochs: 100
            lr: 0.0001
            trainable_params: 67890
            trainable_layers: [...]
      lama_sar:
        v1:
          ...
    ```

    Args:
        repo_id:      HuggingFace repo
        version:      número entero de versión
        filename:     nombre del archivo .pth subido
        base_model:   nombre del archivo .pth base usado como punto de partida
        sar_mode:     modo SAR utilizado ("None", "Concat", o "ControlNet")
        phase1_info:  dict con epochs, lr, trainable_params, trainable_layers
        phase2_info:  dict con epochs, lr, trainable_params, trainable_layers
        notes:        comentario libre del usuario
    """
    data = _download_versions_yaml(repo_id)
    if "models" not in data:
        data["models"] = {}

    model_name = base_model.rsplit("_")[0]  # e.g. "lama_no_sar_pretrained_v1.pth" → "lama"
    mkey = _model_key(model_name, sar_mode)
    if mkey not in data["models"]:
        data["models"][mkey] = {}

    vkey = f"v{version}"
    data["models"][mkey][vkey] = {
        "filename":   filename,
        "base_model": base_model,
        "date":       str(date.today()),
        "notes":      notes,
        "phase1":     phase1_info,
        "phase2":     phase2_info,
    }

    _upload_versions_yaml(repo_id, data)
    print(f"Versión registrada en versions.yaml: models.{mkey}.{vkey}")


def print_versions(repo_id: str) -> None:
    """Imprime en consola el versions.yaml del repo (útil para diagnóstico)."""
    data = _download_versions_yaml(repo_id)
    if not data.get("models"):
        print(f"No hay versiones registradas en '{repo_id}'.")
        return
    print(f"\n{'='*50}")
    print(f"Versiones en '{repo_id}':")
    print(f"{'='*50}")
    print(yaml.dump(data, allow_unicode=True, sort_keys=False))