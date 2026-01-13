"""Evaluation Metrics for DC power prediction."""

import numpy as np
import torch
from typing import Dict, Union


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def RMSE(y_true, y_pred) -> float:
    y_true, y_pred = to_numpy(y_true).flatten(), to_numpy(y_pred).flatten()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def MAE(y_true, y_pred) -> float:
    y_true, y_pred = to_numpy(y_true).flatten(), to_numpy(y_pred).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))


def MAPE(y_true, y_pred, epsilon: float = 1e-8) -> float:
    y_true, y_pred = to_numpy(y_true).flatten(), to_numpy(y_pred).flatten()
    mask = np.abs(y_true) > epsilon
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def R2Score(y_true, y_pred) -> float:
    y_true, y_pred = to_numpy(y_true).flatten(), to_numpy(y_pred).flatten()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - (ss_res / ss_tot))


def nRMSE(y_true, y_pred, normalization: str = "range") -> float:
    y_true, y_pred = to_numpy(y_true).flatten(), to_numpy(y_pred).flatten()
    rmse = RMSE(y_true, y_pred)
    
    if normalization == "range":
        norm_factor = y_true.max() - y_true.min()
    elif normalization == "mean":
        norm_factor = np.mean(y_true)
    else:
        norm_factor = np.std(y_true)
    
    return float((rmse / norm_factor) * 100) if norm_factor > 0 else 0.0


def calculate_metrics(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    metrics = {
        "rmse": RMSE(y_true, y_pred),
        "mae": MAE(y_true, y_pred),
        "mape": MAPE(y_true, y_pred),
        "r2": R2Score(y_true, y_pred),
        "nrmse": nRMSE(y_true, y_pred)
    }
    
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    
    return metrics
