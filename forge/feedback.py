from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import diagnose_result
from .harness_spec import get_feedback_schema


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _target_metric(result: dict[str, Any], name: str = "mae_inverse") -> float | None:
    if not result or not result.get("success"):
        return None
    return result.get("metrics", {}).get("target", {}).get(name)


def _load_curve(result: dict[str, Any]) -> list[dict[str, Any]]:
    curve_path = result.get("paths", {}).get("curve") if result else None
    if not curve_path or not Path(curve_path).exists():
        return []
    rows = []
    with Path(curve_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def encode_feedback(
    result: dict[str, Any],
    previous_result: dict[str, Any] | None = None,
    best_result: dict[str, Any] | None = None,
    target_metric: str = "mae_inverse",
) -> dict[str, Any]:
    success = bool(result.get("success"))
    error_type = str(result.get("error_type") or "")
    error_text = f"{result.get('error_message', '')}\n{result.get('traceback', '')}".lower()
    curve = _load_curve(result)

    train_loss = _safe_float(result.get("metrics", {}).get("train", {}).get("final_train_loss"))
    val_loss = _safe_float(result.get("metrics", {}).get("train", {}).get("final_val_loss"))
    val_values = [_safe_float(row.get("val_loss")) for row in curve]
    train_values = [_safe_float(row.get("train_loss")) for row in curve]

    gen_gap = max(0.0, val_loss - train_loss)
    overfit_score = gen_gap / (abs(train_loss) + 1e-8) if success else 0.0
    underfit_score = min(train_loss, val_loss) if success else 0.0
    val_slope = _slope(val_values[-8:])
    val_volatility = float(np.std(val_values[-8:])) if len(val_values) >= 2 else 0.0

    inv = result.get("metrics", {}).get("inverse", {}) if success else {}
    mae_inv = _safe_float(inv.get("mae"))
    rmse_inv = _safe_float(inv.get("rmse"))
    mape_inv = _safe_float(inv.get("mape"))

    current_target = _target_metric(result, target_metric)
    previous_target = _target_metric(previous_result or {}, target_metric)
    best_target = _target_metric(best_result or {}, target_metric)
    degraded = 0.0
    improved_best = 0.0
    if current_target is not None and previous_target is not None:
        degraded = max(0.0, (current_target - previous_target) / (abs(previous_target) + 1e-8))
    if current_target is not None and best_target is not None:
        improved_best = max(0.0, (best_target - current_target) / (abs(best_target) + 1e-8))

    diagnostics = diagnose_result(result, previous_result, best_result, target_metric=target_metric)
    diagnostic_features = {
        f"diag_{item['name']}": float(item.get("severity", 0.0)) * float(item.get("confidence", 0.0))
        for item in diagnostics
        if item.get("name")
    }

    features = {
        "run_success": 1.0 if success else 0.0,
        "has_exception": 0.0 if success else 1.0,
        "syntax_error": 1.0 if error_type == "syntax" else 0.0,
        "import_error": 1.0 if error_type == "import" else 0.0,
        "shape_error": 1.0 if error_type == "shape" else 0.0,
        "oom_error": 1.0 if error_type == "oom" else 0.0,
        "nan_or_inf": 1.0 if ("nan" in error_text or "inf" in error_text or "non-finite" in error_text) else 0.0,
        "mae_inverse_log": math.log1p(mae_inv),
        "rmse_inverse_log": math.log1p(rmse_inv),
        "mape_inverse_log": math.log1p(mape_inv),
        "train_final_loss": train_loss,
        "val_final_loss": val_loss,
        "generalization_gap": gen_gap,
        "overfit_score": overfit_score,
        "underfit_score": underfit_score,
        "val_slope": val_slope,
        "val_volatility": val_volatility,
        "early_stopped": 1.0 if result.get("metrics", {}).get("train", {}).get("early_stopped") else 0.0,
        "degraded_vs_previous": degraded,
        "improved_vs_best": improved_best,
        "cold_start": 1.0 if previous_result is None else 0.0,
    }
    features.update(diagnostic_features)
    schema = get_feedback_schema()
    vector = [float(features.get(name, 0.0)) for name in schema]

    return {
        "schema": schema,
        "vector": vector,
        "features": features,
        "diagnostics": diagnostics,
        "target_metric": target_metric,
        "current_target": current_target,
        "previous_target": previous_target,
        "best_target": best_target,
        "curve_summary": {
            "epochs": len(curve),
            "last_train_losses": train_values[-5:],
            "last_val_losses": val_values[-5:],
            "val_slope_last8": val_slope,
            "val_volatility_last8": val_volatility,
        },
        "error": {
            "type": error_type,
            "message": result.get("error_message"),
            "traceback_tail": (result.get("traceback") or "")[-2000:],
        },
        "notes": [
            "Lower target metric is better.",
            "The vector encodes failures, degradation, train/val dynamics, and noisy curve statistics.",
        ],
    }
