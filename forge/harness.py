from __future__ import annotations

import json
import random
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_pemfc_data
from .harness_spec import get_default_dataset_name
from .metrics import metric_dict
from .model_io import instantiate_model


@dataclass
class HarnessConfig:
    data_name: str = field(default_factory=get_default_dataset_name)
    data_path: str | None = None
    seq_len: int = 24
    pred_len: int = 12
    scaling: str = "baseline"
    limit_rows: int | None = None
    enc_in: int = 5
    hidden_dim: int = 256
    layer: int = 2
    dropout: float = 0.1
    batch_size: int = 128
    lr: float = 0.001
    epochs: int = 200
    patience: int = 5
    seed: int = 2025
    device: str = "cuda"
    cuda_id: int = 0
    num_workers: int = 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def _append_jsonl(row: dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _resolve_device(device_name: str, cuda_id: int = 0) -> torch.device:
    name = str(device_name or "cuda").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "auto":
        return torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        count = torch.cuda.device_count()
        if cuda_id < 0 or (count and cuda_id >= count):
            raise RuntimeError(f"CUDA device index {cuda_id} is unavailable; visible device count is {count}")
        return torch.device(f"cuda:{cuda_id}")
    if name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{device_name} was requested but torch.cuda.is_available() is False")
        return torch.device(name)
    raise ValueError("device must be one of: cuda, cpu, auto, cuda:<id>")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _categorize_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "cuda out of memory" in text or "outofmemory" in text:
        return "oom"
    if "shape" in text or "size mismatch" in text or "mat1 and mat2" in text:
        return "shape"
    if isinstance(exc, (SyntaxError, IndentationError)):
        return "syntax"
    if isinstance(exc, ImportError):
        return "import"
    return "runtime"


def _inverse_scale(data: np.ndarray, scaler_y: Any, enc_in: int) -> np.ndarray:
    original_shape = data.shape
    return scaler_y.inverse_transform(data.reshape(-1, enc_in)).reshape(original_shape)


def run_harness(model_path: str | Path, run_dir: str | Path, cfg: HarnessConfig) -> dict[str, Any]:
    """Run fixed PEMFC training and evaluation against one model source file."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    local_model_path = run_dir / "model.py"
    if model_path.resolve() != local_model_path.resolve():
        shutil.copy2(model_path, local_model_path)

    curve_path = run_dir / "train_curve.jsonl"
    metrics_path = run_dir / "metrics.json"
    result_path = run_dir / "result.json"
    best_model_path = run_dir / "best_model.pt"
    prediction_path = run_dir / "predictions.npz"

    started = time.time()
    result: dict[str, Any] = {
        "success": False,
        "run_dir": str(run_dir),
        "model_path": str(local_model_path),
        "config": asdict(cfg),
        "paths": {
            "curve": str(curve_path),
            "metrics": str(metrics_path),
            "result": str(result_path),
            "best_model": str(best_model_path),
            "predictions": str(prediction_path),
        },
    }

    try:
        _set_seed(cfg.seed)
        pemfc = load_pemfc_data(
            data_name=cfg.data_name,
            data_path=cfg.data_path,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            limit_rows=cfg.limit_rows,
            scaling=cfg.scaling,
        )
        result["data"] = {
            "data_name": pemfc.data_name,
            "data_path": str(pemfc.data_path),
            "split_sizes": pemfc.split_sizes,
            "feature_dim": pemfc.feature_dim,
        }

        model_cfg = SimpleNamespace(
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            enc_in=pemfc.enc_in,
            hidden_dim=cfg.hidden_dim,
            layer=cfg.layer,
            dropout=cfg.dropout,
            feature_dim=pemfc.feature_dim,
            input_dim=pemfc.feature_dim,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            epochs=cfg.epochs,
            patience=cfg.patience,
            data=cfg.data_name,
        )
        model = instantiate_model(local_model_path, model_cfg)
        device = _resolve_device(cfg.device, cfg.cuda_id)
        model.to(device)

        train_dataset = TensorDataset(torch.FloatTensor(pemfc.X_train), torch.FloatTensor(pemfc.y_train))
        val_dataset = TensorDataset(torch.FloatTensor(pemfc.X_val), torch.FloatTensor(pemfc.y_val))
        test_dataset = TensorDataset(torch.FloatTensor(pemfc.X_test), torch.FloatTensor(pemfc.y_test))

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        criterion = nn.L1Loss()
        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        train_curve: list[dict[str, Any]] = []

        if curve_path.exists():
            curve_path.unlink()

        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0.0
            batch_count = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_train_loss += float(loss.item())
                batch_count += 1

            train_loss = total_train_loss / max(batch_count, 1)
            model.eval()
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
                    total_val_loss += float(loss.item())
                    val_batches += 1
            val_loss = total_val_loss / max(val_batches, 1)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_sec": time.time() - epoch_start,
            }
            train_curve.append(row)
            _append_jsonl(row, curve_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    break

        if not best_model_path.exists():
            torch.save(model.state_dict(), best_model_path)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                if not torch.isfinite(outputs).all():
                    raise FloatingPointError("Non-finite model outputs during evaluation")
                y_pred_list.append(outputs.cpu().numpy())
                y_true_list.append(targets.numpy())

        y_pred = np.concatenate(y_pred_list, axis=0).reshape(-1, cfg.pred_len, pemfc.enc_in)
        y_true = np.concatenate(y_true_list, axis=0).reshape(-1, cfg.pred_len, pemfc.enc_in)
        y_pred_inv = _inverse_scale(y_pred, pemfc.scaler_y, pemfc.enc_in)
        y_true_inv = _inverse_scale(y_true, pemfc.scaler_y, pemfc.enc_in)
        np.savez_compressed(
            prediction_path,
            y_pred=y_pred,
            y_true=y_true,
            y_pred_inverse=y_pred_inv,
            y_true_inverse=y_true_inv,
        )

        metrics = {
            "normalized": metric_dict(y_pred, y_true),
            "inverse": metric_dict(y_pred_inv, y_true_inv),
            "train": {
                "best_val_loss": float(best_val_loss),
                "best_epoch": int(best_epoch),
                "epochs_run": int(len(train_curve)),
                "early_stopped": bool(len(train_curve) < cfg.epochs),
                "final_train_loss": float(train_curve[-1]["train_loss"]) if train_curve else None,
                "final_val_loss": float(train_curve[-1]["val_loss"]) if train_curve else None,
            },
        }
        metrics["target"] = {
            "mae_inverse": metrics["inverse"]["mae"],
            "rmse_inverse": metrics["inverse"]["rmse"],
            "mape_inverse": metrics["inverse"]["mape"],
            "mae_normalized": metrics["normalized"]["mae"],
        }
        _write_json(metrics, metrics_path)

        result.update(
            {
                "success": True,
                "metrics": metrics,
                "duration_sec": time.time() - started,
                "device": str(device),
                "requested_device": cfg.device,
                "cuda_id": cfg.cuda_id,
            }
        )
    except BaseException as exc:
        result.update(
            {
                "success": False,
                "duration_sec": time.time() - started,
                "error_type": _categorize_exception(exc),
                "error_message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )

    _write_json(result, result_path)
    return result
