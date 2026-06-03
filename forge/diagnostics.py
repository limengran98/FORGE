from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ratio_severity(numerator: float, denominator: float, scale: float = 1.0) -> float:
    if denominator <= 1e-12:
        return 0.0
    return _clip01(((numerator / denominator) - 1.0) / max(scale, 1e-8))


def _load_curve(result: dict[str, Any]) -> list[dict[str, Any]]:
    curve_path = result.get("paths", {}).get("curve") if result else None
    if not curve_path or not Path(curve_path).exists():
        return []
    rows: list[dict[str, Any]] = []
    with Path(curve_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_predictions(result: dict[str, Any]) -> dict[str, np.ndarray]:
    prediction_path = result.get("paths", {}).get("predictions") if result else None
    if not prediction_path or not Path(prediction_path).exists():
        return {}
    with np.load(prediction_path) as data:
        return {key: data[key] for key in data.files}


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    return float(np.polyfit(x, values.astype(np.float64), 1)[0])


def _residual_autocorrelation(residual: np.ndarray) -> float:
    flat = residual.reshape(-1, residual.shape[-1])
    vals: list[float] = []
    for channel in range(flat.shape[-1]):
        x = flat[:-1, channel]
        y = flat[1:, channel]
        if x.size > 2 and float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12:
            corr = float(np.corrcoef(x, y)[0, 1])
            if not math.isnan(corr) and not math.isinf(corr):
                vals.append(abs(corr))
    return float(np.mean(vals)) if vals else 0.0


def _diagnostic(name: str, severity: float, confidence: float, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "severity": round(_clip01(severity), 6),
        "confidence": round(_clip01(confidence), 6),
        "evidence": evidence,
    }


def diagnose_result(
    result: dict[str, Any],
    previous_result: dict[str, Any] | None = None,
    best_result: dict[str, Any] | None = None,
    target_metric: str = "mae_inverse",
) -> list[dict[str, Any]]:
    """Build PEMFC diagnostic feedback nodes from executable harness artifacts."""
    diagnostics: list[dict[str, Any]] = []
    success = bool(result.get("success"))
    if not success:
        error_type = str(result.get("error_type") or "")
        diagnostics.append(
            _diagnostic(
                "harness_failure",
                1.0,
                1.0,
                {
                    "error_type": error_type,
                    "error_message": result.get("error_message"),
                },
            )
        )
        if error_type == "shape":
            diagnostics.append(_diagnostic("shape_contract_error", 1.0, 1.0, {"error_type": error_type}))
        if "nan" in str(result.get("error_message") or "").lower():
            diagnostics.append(_diagnostic("nan_instability", 1.0, 1.0, {"error_type": error_type}))
        return diagnostics

    metrics = result.get("metrics", {})
    train_metrics = metrics.get("train", {})
    train_loss = _safe_float(train_metrics.get("final_train_loss"))
    val_loss = _safe_float(train_metrics.get("final_val_loss"))
    gap = max(0.0, val_loss - train_loss)
    gap_ratio = gap / (abs(train_loss) + 1e-8) if train_loss else 0.0
    diagnostics.append(
        _diagnostic(
            "train_val_gap",
            gap_ratio / 1.0,
            0.9,
            {"train_loss": train_loss, "val_loss": val_loss, "gap_ratio": gap_ratio},
        )
    )

    curve = _load_curve(result)
    val_values = np.asarray([_safe_float(row.get("val_loss")) for row in curve], dtype=np.float64)
    if val_values.size >= 2:
        recent = val_values[-min(8, val_values.size) :]
        volatility = float(np.std(recent))
        slope = _slope(recent)
        diagnostics.append(
            _diagnostic(
                "val_curve_instability",
                max(volatility / 0.05, slope / 0.01),
                0.75,
                {"volatility": volatility, "slope": slope, "epochs": int(val_values.size)},
            )
        )

    current_target = metrics.get("target", {}).get(target_metric)
    prev_target = (previous_result or {}).get("metrics", {}).get("target", {}).get(target_metric)
    best_target = (best_result or {}).get("metrics", {}).get("target", {}).get(target_metric)
    reference = prev_target if prev_target is not None else best_target
    if current_target is not None and reference is not None:
        degradation = max(0.0, (_safe_float(current_target) - _safe_float(reference)) / (abs(_safe_float(reference)) + 1e-8))
        diagnostics.append(
            _diagnostic(
                "target_degradation",
                degradation / 0.1,
                0.95,
                {"current": current_target, "reference": reference, "relative_degradation": degradation},
            )
        )

    pred = _load_predictions(result)
    y_pred = pred.get("y_pred_inverse")
    y_true = pred.get("y_true_inverse")
    if y_pred is None or y_true is None or y_pred.size == 0 or y_true.size == 0:
        return diagnostics

    abs_err = np.abs(y_pred - y_true)
    residual = y_pred - y_true
    horizon_mae = abs_err.mean(axis=(0, 2))
    channel_mae = abs_err.mean(axis=(0, 1))
    short_count = max(1, abs_err.shape[1] // 3)
    long_count = max(1, abs_err.shape[1] // 3)
    short_mae = float(abs_err[:, :short_count, :].mean())
    long_mae = float(abs_err[:, -long_count:, :].mean())
    diagnostics.append(
        _diagnostic(
            "long_horizon_error",
            _ratio_severity(long_mae, short_mae, scale=0.5),
            0.9,
            {
                "short_horizon_mae": short_mae,
                "long_horizon_mae": long_mae,
                "worst_horizon": int(np.argmax(horizon_mae) + 1),
                "worst_horizon_mae": float(horizon_mae.max()),
            },
        )
    )

    n_windows = abs_err.shape[0]
    early = float(abs_err[: max(1, n_windows // 3)].mean())
    late = float(abs_err[int(n_windows * 2 / 3) :].mean())
    diagnostics.append(
        _diagnostic(
            "late_life_error",
            _ratio_severity(late, early, scale=0.5),
            0.8,
            {"early_test_mae": early, "late_test_mae": late},
        )
    )

    residual_mean_by_window = residual.mean(axis=(1, 2))
    drift_slope = _slope(residual_mean_by_window)
    drift_norm = abs(drift_slope) * max(1, residual_mean_by_window.size) / (float(np.std(residual_mean_by_window)) + 1e-8)
    diagnostics.append(
        _diagnostic(
            "residual_drift",
            drift_norm / 1.0,
            0.8,
            {"slope": drift_slope, "normalized_drift": drift_norm},
        )
    )

    autocorr = _residual_autocorrelation(residual)
    diagnostics.append(
        _diagnostic(
            "residual_autocorrelation",
            max(0.0, (autocorr - 0.35) / 0.65),
            0.85,
            {"lag1_abs_autocorrelation": autocorr},
        )
    )

    channel_ratio = float(channel_mae.max() / (channel_mae.mean() + 1e-8))
    diagnostics.append(
        _diagnostic(
            "channel_imbalance",
            (channel_ratio - 1.0) / 0.75,
            0.75,
            {
                "channel_mae": [float(value) for value in channel_mae],
                "worst_channel": int(np.argmax(channel_mae)),
                "max_to_mean_ratio": channel_ratio,
            },
        )
    )

    x_test = pred.get("x_test")
    if x_test is not None and x_test.ndim == 3 and x_test.shape[0] == abs_err.shape[0]:
        enc_in = y_true.shape[-1]
        factor = x_test[:, :, enc_in:]
        if factor.size:
            load_score = factor.std(axis=1).mean(axis=1)
            threshold = float(np.quantile(load_score, 0.7))
            high_mask = load_score >= threshold
            low_mask = load_score < threshold
            if bool(high_mask.any()) and bool(low_mask.any()):
                high_mae = float(abs_err[high_mask].mean())
                low_mae = float(abs_err[low_mask].mean())
                diagnostics.append(
                    _diagnostic(
                        "dynamic_load_error",
                        _ratio_severity(high_mae, low_mae, scale=0.5),
                        0.75,
                        {"high_dynamic_mae": high_mae, "low_dynamic_mae": low_mae, "load_threshold": threshold},
                    )
                )

    return diagnostics


def diagnostics_by_name(feedback: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("name")): item for item in feedback.get("diagnostics", []) if item.get("name")}
