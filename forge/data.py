from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .assets import resolve_data_path
from .harness_spec import get_default_dataset_name, get_feature_groups, get_split_ratios


@dataclass
class WindowedPEMFCData:
    data_name: str
    data_path: Path
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler_x_volt: StandardScaler
    scaler_x_factor: StandardScaler
    scaler_y: StandardScaler
    seq_len: int
    pred_len: int
    enc_in: int
    feature_dim: int
    split_sizes: dict[str, int]


def _create_windows(X: np.ndarray, y: np.ndarray, seq_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    X_windows = []
    y_windows = []
    max_start = len(X) - seq_len - pred_len + 1
    for i in range(max_start):
        X_windows.append(X[i : i + seq_len])
        y_windows.append(y[i + seq_len : i + seq_len + pred_len])
    return np.asarray(X_windows, dtype=np.float32), np.asarray(y_windows, dtype=np.float32)


def load_pemfc_data(
    data_name: str | None = None,
    data_path: str | Path | None = None,
    seq_len: int = 24,
    pred_len: int = 12,
    limit_rows: int | None = None,
    scaling: str = "baseline",
) -> WindowedPEMFCData:
    """Load PEMFC data using the Ms-AeDNet chronological 6:2:2 protocol."""
    data_name = data_name or get_default_dataset_name()
    resolved_path = resolve_data_path(data_name, data_path)
    data = pd.read_csv(resolved_path)
    if limit_rows:
        data = data.iloc[: int(limit_rows)].copy()

    features = get_feature_groups()
    input_volt_features = features["voltage_inputs"]
    input_factor_features = features["factor_inputs"]
    output_volt_features = features["targets"]

    required = list(dict.fromkeys(input_volt_features + input_factor_features + output_volt_features))
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in {resolved_path}: {missing}")

    total_samples = len(data)
    train_ratio, val_ratio, test_ratio = get_split_ratios()
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_end = int((train_ratio / ratio_sum) * total_samples)
    val_end = train_end + int((val_ratio / ratio_sum) * total_samples)
    test_end = val_end + int((test_ratio / ratio_sum) * total_samples)
    if min(train_end, val_end - train_end, test_end - val_end) < seq_len + pred_len:
        raise ValueError(
            "Not enough rows after chronological split for windowing: "
            f"rows={total_samples}, seq_len={seq_len}, pred_len={pred_len}"
        )

    train_data = data.iloc[:train_end]
    scaler_x_volt = StandardScaler()
    scaler_x_factor = StandardScaler()
    scaler_y = StandardScaler()

    if scaling == "baseline":
        scaler_x_volt.fit(data[input_volt_features].values)
        scaler_y.fit(data[output_volt_features].values)
    elif scaling == "train":
        scaler_x_volt.fit(train_data[input_volt_features].values)
        scaler_y.fit(train_data[output_volt_features].values)
    else:
        raise ValueError("scaling must be 'baseline' or 'train'")

    scaler_x_factor.fit(train_data[input_factor_features].values)

    X_all_volt = scaler_x_volt.transform(data[input_volt_features].values)
    X_all_factor = scaler_x_factor.transform(data[input_factor_features].values)
    X_all = np.hstack([X_all_volt, X_all_factor]).astype(np.float32)
    y_all = scaler_y.transform(data[output_volt_features].values).astype(np.float32)

    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:test_end], y_all[val_end:test_end]

    X_train_w, y_train_w = _create_windows(X_train, y_train, seq_len, pred_len)
    X_val_w, y_val_w = _create_windows(X_val, y_val, seq_len, pred_len)
    X_test_w, y_test_w = _create_windows(X_test, y_test, seq_len, pred_len)

    return WindowedPEMFCData(
        data_name=data_name.upper(),
        data_path=resolved_path,
        X_train=X_train_w,
        y_train=y_train_w,
        X_val=X_val_w,
        y_val=y_val_w,
        X_test=X_test_w,
        y_test=y_test_w,
        scaler_x_volt=scaler_x_volt,
        scaler_x_factor=scaler_x_factor,
        scaler_y=scaler_y,
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=len(output_volt_features),
        feature_dim=len(input_volt_features) + len(input_factor_features),
        split_sizes={
            "raw_total": total_samples,
            "train_rows": train_end,
            "val_rows": val_end - train_end,
            "test_rows": test_end - val_end,
            "train_windows": int(X_train_w.shape[0]),
            "val_windows": int(X_val_w.shape[0]),
            "test_windows": int(X_test_w.shape[0]),
        },
    )
