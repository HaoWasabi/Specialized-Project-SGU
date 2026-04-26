import json
from pathlib import Path
from typing import Any

import joblib


def ensure_parent_dir(file_path: str | Path) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(file_path: str | Path, payload: dict[str, Any]) -> None:
    path = ensure_parent_dir(file_path)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=4, ensure_ascii=False)


def save_joblib_object(file_path: str | Path, payload: Any) -> None:
    path = ensure_parent_dir(file_path)
    joblib.dump(payload, path)


def save_keras_model(file_path: str | Path, model: Any) -> None:
    path = ensure_parent_dir(file_path)
    model.save(path)