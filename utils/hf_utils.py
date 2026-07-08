from __future__ import annotations

import io
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

def download_model(repo_id: str, filename: str, *, map_location: str = "cpu",) -> dict:
    print(f"Descargando '{filename}' desde '{repo_id}'")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    if isinstance(checkpoint, io.BytesIO):
        checkpoint.seek(0)
        try:
            checkpoint = torch.load(checkpoint, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint, map_location=map_location)

    return checkpoint

def _list_versions(repo_id: str, prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.pth$")
    versions = []
    try:
        for f in list_repo_files(repo_id):
            m = pattern.match(f)
            if m:
                versions.append(int(m.group(1)))
    except Exception as e:
        print(f"Error no se pudieron listar archivos del repo {e}")
    return sorted(versions)


def resolve_load_version(repo_id: str, filename_prefix: str, requested_version: Optional[int]) -> tuple[int, str]:
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
        print(f"Usando version: v{version}.")

    filename = f"{filename_prefix}_v{version}.pth"
    return version, filename

def resolve_save_version(repo_id: str, filename_prefix: str, requested_version: Optional[int],) -> tuple[int, str]:
    existing = _list_versions(repo_id, filename_prefix)

    if requested_version is None:
        version = (max(existing) + 1) if existing else 1
        print(f"Versión no especificada. Se usará v{version} (nueva).")
    else:
        version = requested_version
        if version in existing:
            filename = f"{filename_prefix}_v{version}.pth"
            answer = input(
                f"'{filename}' ya existe en '{repo_id}'. "
                "sobreescribirlo? (s/n): "
            ).strip().lower()
            if answer != "s":
                print("cancelada.")
                sys.exit(0)
            print(f"Se sobreescribe v{version}.")
        else:
            print(f"Se usa la versión: v{version}.")

    filename = f"{filename_prefix}_v{version}.pth"
    return version, filename

def upload_model(model_state_dict, repo_id: str, filename: str) -> None:

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:

        if isinstance(model_state_dict, io.BytesIO):
            model_state_dict.seek(0)
            tmp.write(model_state_dict.read())

        elif isinstance(model_state_dict, (bytes, bytearray)):
            tmp.write(model_state_dict)

        else:
            torch.save(model_state_dict, tmp)

        tmp.flush()
        tmp_path = tmp.name

    try:
        print(f"Subiendo '{filename}' a '{repo_id}'")

        upload_file(path_or_fileobj=tmp_path, path_in_repo=filename, repo_id=repo_id, repo_type="model")

        print(f"Modelo guardado en HuggingFace: {repo_id}/{filename}")

    finally:
        Path(tmp_path).unlink(missing_ok=True)

def _download_versions_yaml(repo_id: str) -> dict:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=VERSIONS_FILENAME)
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        return {}

def _upload_versions_yaml(repo_id: str, data: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(data, tmp, allow_unicode=True, sort_keys=False)
        tmp_path = tmp.name

    try:
        upload_file(path_or_fileobj=tmp_path, path_in_repo=VERSIONS_FILENAME, repo_id=repo_id, repo_type="model",)
        print(f"versions.yaml actualizado en '{repo_id}'.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def register_version(repo_id: str, version: int, filename: str, *, model_name: str, base_model: str, sar_mode: str, phase1_info: dict, phase2_info: dict, notes: str = "",) -> None:
    data = _download_versions_yaml(repo_id)
    if "models" not in data:
        data["models"] = {}

    model_name = model_name.lower()
    mkey = model_name
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
    print(f"Versión guardada en versions.yaml: models.{mkey}.{vkey}")

def print_versions(repo_id: str) -> None:
    data = _download_versions_yaml(repo_id)
    if not data.get("models"):
        print(f"No hay versiones en '{repo_id}'.")
        return
    print(f"\n{'='*50}")
    print(f"Versiones en '{repo_id}':")
    print(f"{'='*50}")
    print(yaml.dump(data, allow_unicode=True, sort_keys=False))