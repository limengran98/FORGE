from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(100 * (pred - true) / (true + 1e-8))))


def smape(pred: np.ndarray, true: np.ndarray) -> float:
    denom = np.abs(pred) + np.abs(true) + 1e-8
    return float(np.mean(200 * np.abs(pred - true) / denom))


def nd(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)) / (np.mean(np.abs(true)) + 1e-8))


def metric_dict(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(pred, true),
        "mse": mse(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true),
        "smape": smape(pred, true),
        "nd": nd(pred, true),
    }

