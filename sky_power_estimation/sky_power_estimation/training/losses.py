"""
Loss Functions Module

Custom loss functions for DC power prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss."""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(pred, target) + self.eps)


class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error Loss."""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # Avoid division by zero
        mask = target.abs() > self.eps
        
        if not mask.any():
            return torch.tensor(0.0, device=pred.device)
        
        error = torch.abs((pred[mask] - target[mask]) / target[mask])
        return error.mean() * 100


class HuberLoss(nn.Module):
    """Smooth L1 (Huber) Loss."""
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target, beta=self.delta)


class QuantileLoss(nn.Module):
    """
    Quantile Loss for probabilistic predictions.
    
    Useful for estimating prediction intervals.
    """
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions [batch, num_quantiles]
            target: Targets [batch]
        """
        losses = []
        target = target.unsqueeze(-1)
        
        for i, q in enumerate(self.quantiles):
            error = target - pred[:, i:i+1]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple components.
    
    Combines MSE, MAE, and optional auxiliary losses.
    """
    
    def __init__(
        self,
        mse_weight: float = 0.5,
        mae_weight: float = 0.3,
        huber_weight: float = 0.2,
        huber_delta: float = 1.0
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.huber_weight = huber_weight
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.huber = HuberLoss(delta=huber_delta)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Returns dictionary with individual loss components.
        """
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        huber_loss = self.huber(pred, target)
        
        total_loss = (
            self.mse_weight * mse_loss +
            self.mae_weight * mae_loss +
            self.huber_weight * huber_loss
        )
        
        return {
            "total": total_loss,
            "mse": mse_loss,
            "mae": mae_loss,
            "huber": huber_loss
        }


class WeightedMSELoss(nn.Module):
    """
    MSE Loss with sample weights.
    
    Can weight recent samples higher or weight by power magnitude.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = (pred - target) ** 2
        
        if weights is not None:
            loss = loss * weights
        
        if self.reduction == "mean":
            if weights is not None:
                return loss.sum() / weights.sum()
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class GradientMatchingLoss(nn.Module):
    """
    Loss that encourages matching gradients (trends) in addition to values.
    
    Useful for capturing ramp events.
    """
    
    def __init__(self, value_weight: float = 0.7, gradient_weight: float = 0.3):
        super().__init__()
        self.value_weight = value_weight
        self.gradient_weight = gradient_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # Value loss
        value_loss = F.mse_loss(pred, target)
        
        # Gradient loss (difference between consecutive predictions)
        if pred.dim() > 1 and pred.size(0) > 1:
            pred_grad = pred[1:] - pred[:-1]
            target_grad = target[1:] - target[:-1]
            gradient_loss = F.mse_loss(pred_grad, target_grad)
        else:
            gradient_loss = torch.tensor(0.0, device=pred.device)
        
        return self.value_weight * value_loss + self.gradient_weight * gradient_loss


def get_loss_function(
    loss_type: str = "mse",
    **kwargs
) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: Type of loss ('mse', 'mae', 'huber', 'rmse', 'combined')
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function module
    """
    loss_type = loss_type.lower()
    
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "huber":
        return HuberLoss(delta=kwargs.get("huber_delta", 1.0))
    elif loss_type == "rmse":
        return RMSELoss()
    elif loss_type == "mape":
        return MAPELoss()
    elif loss_type == "combined":
        return CombinedLoss(**kwargs)
    elif loss_type == "gradient":
        return GradientMatchingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test losses
    print("Testing Loss Functions...")
    
    pred = torch.randn(32)
    target = torch.randn(32)
    
    for loss_type in ["mse", "mae", "huber", "rmse", "mape", "combined"]:
        loss_fn = get_loss_function(loss_type)
        
        if loss_type == "combined":
            losses = loss_fn(pred, target)
            print(f"{loss_type}: total={losses['total']:.4f}, mse={losses['mse']:.4f}")
        else:
            loss = loss_fn(pred, target)
            print(f"{loss_type}: {loss:.4f}")
