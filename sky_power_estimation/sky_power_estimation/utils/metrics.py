"""
Metrics Module

Evaluation metrics for DC power prediction.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def RMSE(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAE(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    
    return np.mean(np.abs(y_true - y_pred))


def MAPE(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    epsilon: float = 1e-8
) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value (as percentage)
    """
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    
    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    
    if not mask.any():
        return 0.0
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def R2Score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    R-squared (coefficient of determination).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² value
    """
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def nRMSE(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    normalization: str = "range"
) -> float:
    """
    Normalized Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        normalization: Normalization method ('range', 'mean', 'std')
        
    Returns:
        nRMSE value (as percentage)
    """
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    
    rmse = RMSE(y_true, y_pred)
    
    if normalization == "range":
        norm_factor = y_true.max() - y_true.min()
    elif normalization == "mean":
        norm_factor = np.mean(y_true)
    elif normalization == "std":
        norm_factor = np.std(y_true)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    
    if norm_factor == 0:
        return 0.0
    
    return (rmse / norm_factor) * 100


def skill_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_baseline: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Forecast Skill Score.
    
    Compares model performance to a baseline (e.g., persistence).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_baseline: Baseline predictions
        
    Returns:
        Skill score (1 = perfect, 0 = same as baseline, negative = worse)
    """
    rmse_model = RMSE(y_true, y_pred)
    rmse_baseline = RMSE(y_true, y_baseline)
    
    if rmse_baseline == 0:
        return 0.0
    
    return 1 - (rmse_model / rmse_baseline)


def MBE(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Mean Bias Error.
    
    Indicates systematic over/under-prediction.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MBE value (positive = over-prediction)
    """
    y_true = to_numpy(y_true).flatten()
    y_pred = to_numpy(y_pred).flatten()
    
    return np.mean(y_pred - y_true)


def calculate_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate all metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        prefix: Optional prefix for metric names
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        "rmse": RMSE(y_true, y_pred),
        "mae": MAE(y_true, y_pred),
        "mape": MAPE(y_true, y_pred),
        "r2": R2Score(y_true, y_pred),
        "nrmse": nRMSE(y_true, y_pred),
        "mbe": MBE(y_true, y_pred)
    }
    
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    
    return metrics


def calculate_metrics_by_condition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conditions: np.ndarray,
    condition_names: Optional[Dict[int, str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different conditions (e.g., clear, cloudy).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        conditions: Condition labels for each sample
        condition_names: Mapping from condition values to names
        
    Returns:
        Dictionary of metrics for each condition
    """
    unique_conditions = np.unique(conditions)
    results = {}
    
    for cond in unique_conditions:
        mask = conditions == cond
        
        if not mask.any():
            continue
        
        cond_name = condition_names.get(cond, str(cond)) if condition_names else str(cond)
        results[cond_name] = calculate_metrics(y_true[mask], y_pred[mask])
    
    return results


def calculate_metrics_by_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics grouped by hour of day.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        hours: Hour of day for each sample
        
    Returns:
        Dictionary of metrics for each hour
    """
    unique_hours = np.unique(hours)
    results = {}
    
    for hour in unique_hours:
        mask = hours == hour
        
        if mask.sum() < 5:  # Skip if too few samples
            continue
        
        results[f"hour_{int(hour):02d}"] = calculate_metrics(
            y_true[mask], 
            y_pred[mask]
        )
    
    return results


class MetricTracker:
    """
    Track metrics during training.
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(
        self,
        metrics: Dict[str, float],
        n: int = 1
    ):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value * n
            self.counts[key] += n
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
            if self.counts[key] > 0
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


if __name__ == "__main__":
    # Test metrics
    print("Testing Metrics...")
    
    np.random.seed(42)
    n = 100
    
    # Generate test data
    y_true = np.random.uniform(0, 1000, n)
    noise = np.random.normal(0, 50, n)
    y_pred = y_true + noise
    
    # Calculate all metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test skill score
    y_baseline = np.roll(y_true, 1)  # Persistence model
    ss = skill_score(y_true, y_pred, y_baseline)
    print(f"\nSkill score vs persistence: {ss:.4f}")
    
    # Test by condition
    conditions = np.random.randint(0, 3, n)
    condition_names = {0: "clear", 1: "partly_cloudy", 2: "cloudy"}
    
    metrics_by_cond = calculate_metrics_by_condition(
        y_true, y_pred, conditions, condition_names
    )
    
    print("\nMetrics by condition:")
    for cond, cond_metrics in metrics_by_cond.items():
        print(f"  {cond}:")
        print(f"    RMSE: {cond_metrics['rmse']:.4f}")
        print(f"    MAE: {cond_metrics['mae']:.4f}")
