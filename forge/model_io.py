from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .harness_spec import get_model_class_name


class ModelInterfaceError(RuntimeError):
    pass


def read_model_source(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_model_class_from_file(path: str | Path, class_name: str | None = None) -> type:
    class_name = class_name or get_model_class_name()
    path = Path(path)
    module_hash = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    module_name = f"forge_dynamic_model_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModelInterfaceError(f"Cannot import model file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise ModelInterfaceError(f"Model file must define class {class_name}")
    cls = getattr(module, class_name)
    return cls


def instantiate_model(path: str | Path, configs: Any, class_name: str | None = None) -> torch.nn.Module:
    class_name = class_name or get_model_class_name()
    cls = load_model_class_from_file(path, class_name)
    model = cls(configs)
    if not isinstance(model, torch.nn.Module):
        raise ModelInterfaceError(f"{class_name} must inherit torch.nn.Module")
    return model


def validate_model_source(source: str, configs: SimpleNamespace, feature_dim: int) -> None:
    class_name = get_model_class_name()
    namespace: dict[str, Any] = {}
    try:
        compiled = compile(source, "<forge_candidate_model>", "exec")
        exec(compiled, namespace)
    except Exception as exc:
        raise ModelInterfaceError(f"Candidate source does not compile: {exc}") from exc

    if class_name not in namespace:
        raise ModelInterfaceError(f"Candidate source must define {class_name}")

    model = namespace[class_name](configs)
    if not isinstance(model, torch.nn.Module):
        raise ModelInterfaceError(f"{class_name} must inherit torch.nn.Module")

    model.eval()
    x = torch.randn(2, int(configs.seq_len), int(feature_dim))
    expected = (2, int(configs.pred_len), int(configs.enc_in))
    with torch.no_grad():
        try:
            y = model(x)
        except Exception as exc:
            raise ModelInterfaceError(
                "Candidate forward failed during interface validation: "
                f"input_shape={tuple(x.shape)}, expected_output_shape={expected}, "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
    if tuple(y.shape) != expected:
        raise ModelInterfaceError(f"Forward output shape {tuple(y.shape)} != expected {expected}")
    if not torch.isfinite(y).all():
        raise ModelInterfaceError("Forward output contains NaN or Inf")
