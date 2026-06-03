from __future__ import annotations

import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_EXPERIMENT_CONFIG: dict[str, Any] = {
    "data": {
        "name": None,
        "seq_len": 24,
        "pred_len": 12,
        "scaling": "baseline",
        "limit_rows": None,
    },
    "harness": {
        "epochs": 200,
        "batch_size": 128,
        "lr": 0.001,
        "patience": 5,
        "seed": 2025,
        "device": "cuda",
        "cuda_id": 0,
        "num_workers": 0,
    },
    "model": {
        "hidden_dim": 256,
        "layer": 2,
        "dropout": 0.1,
    },
    "evolution": {
        "rounds": 1,
        "target_metric": "mae_inverse",
        "llm_mode": "auto",
    },
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to load FORGE configuration files")
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
            f.write("\n")
            tmp_name = f.name
        Path(tmp_name).replace(path)
    except Exception:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_vars(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    return value


def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    if path:
        loaded = load_yaml(path)
        cfg = _deep_merge(cfg, loaded)
    return _expand_vars(cfg)


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "***"
    return f"{value[:6]}...{value[-4:]}"
