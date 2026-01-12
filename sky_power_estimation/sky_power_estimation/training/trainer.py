"""
Trainer Module

Training loop and utilities for sky power estimation.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    StepLR, 
    ReduceLROnPlateau,
    OneCycleLR
)
from tqdm import tqdm

from .losses import get_loss_function
from ..utils.metrics import calculate_metrics, MetricTracker
from ..utils.logger import setup_logger, TensorBoardLogger


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Trainer class for sky power estimation model.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        device: Device to train on
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        
        # Training config
        training_cfg = config.get("training", {})
        self.epochs = training_cfg.get("epochs", 100)
        self.learning_rate = training_cfg.get("learning_rate", 0.001)
        self.weight_decay = training_cfg.get("weight_decay", 0.0001)
        self.gradient_clip = training_cfg.get("gradient_clip", 1.0)
        self.use_amp = training_cfg.get("mixed_precision", True) and torch.cuda.is_available()
        
        # Loss function
        loss_cfg = config.get("loss", {})
        self.loss_fn = get_loss_function(
            loss_cfg.get("type", "mse"),
            **loss_cfg
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler(training_cfg.get("scheduler", {}))
        
        # Early stopping
        es_cfg = training_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 15),
            min_delta=es_cfg.get("min_delta", 0.0001)
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Logging
        paths_cfg = config.get("paths", {})
        self.output_dir = Path(paths_cfg.get("output_dir", "./outputs"))
        self.checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "./checkpoints"))
        self.log_dir = Path(paths_cfg.get("log_dir", "./logs"))
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = setup_logger(
            "trainer",
            log_dir=str(self.log_dir),
            log_to_console=True,
            log_to_file=True
        )
        
        # TensorBoard
        logging_cfg = config.get("logging", {})
        if logging_cfg.get("use_tensorboard", True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tb_dir = self.log_dir / f"tensorboard_{timestamp}"
            self.tb_logger = TensorBoardLogger(str(tb_dir))
        else:
            self.tb_logger = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history = {"train": [], "val": []}
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _create_scheduler(self, scheduler_cfg: Dict) -> Optional[Any]:
        """Create learning rate scheduler."""
        scheduler_type = scheduler_cfg.get("type", "cosine")
        
        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=scheduler_cfg.get("min_lr", 1e-6)
            )
        elif scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=scheduler_cfg.get("step_size", 30),
                gamma=scheduler_cfg.get("gamma", 0.1)
            )
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_cfg.get("factor", 0.5),
                patience=scheduler_cfg.get("patience", 5)
            )
        elif scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                epochs=self.epochs,
                steps_per_epoch=len(self.train_loader)
            )
        else:
            return None
    
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(
            current_image=batch["current_image"],
            image_sequence=batch.get("image_sequence"),
            current_weather=batch["current_weather"],
            weather_sequence=batch.get("weather_sequence"),
            current_sun_position=batch["current_sun_position"],
            sun_position_sequence=batch.get("sun_position_sequence")
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metric_tracker = MetricTracker()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            leave=True
        )
        
        for batch in pbar:
            batch = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    predictions = self._forward_pass(batch)
                    loss = self.loss_fn(predictions, batch["target"])
                    
                    if isinstance(loss, dict):
                        loss_value = loss["total"]
                    else:
                        loss_value = loss
                
                # Backward pass
                self.scaler.scale(loss_value).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self._forward_pass(batch)
                loss = self.loss_fn(predictions, batch["target"])
                
                if isinstance(loss, dict):
                    loss_value = loss["total"]
                else:
                    loss_value = loss
                
                loss_value.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(
                    batch["target"].cpu(),
                    predictions.cpu()
                )
                metrics["loss"] = loss_value.item()
            
            metric_tracker.update(metrics, n=len(batch["target"]))
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_value.item():.4f}",
                "rmse": f"{metrics['rmse']:.2f}"
            })
            
            # Step scheduler if per-batch
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            self.global_step += 1
            
            # Log to tensorboard
            if self.tb_logger and self.global_step % 10 == 0:
                self.tb_logger.log_scalar("train/loss", loss_value.item(), self.global_step)
        
        return metric_tracker.compute()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metric_tracker = MetricTracker()
        
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch = self._prepare_batch(batch)
            
            predictions = self._forward_pass(batch)
            loss = self.loss_fn(predictions, batch["target"])
            
            if isinstance(loss, dict):
                loss_value = loss["total"]
            else:
                loss_value = loss
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch["target"].cpu())
            
            metrics = calculate_metrics(
                batch["target"].cpu(),
                predictions.cpu()
            )
            metrics["loss"] = loss_value.item()
            
            metric_tracker.update(metrics, n=len(batch["target"]))
        
        # Aggregate predictions for overall metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        overall_metrics = calculate_metrics(all_targets, all_predictions, prefix="val")
        overall_metrics["val_loss"] = metric_tracker.compute()["loss"]
        
        return overall_metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Training history and best metrics
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val RMSE: {val_metrics['val_rmse']:.2f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save history
            self.training_history["train"].append(train_metrics)
            self.training_history["val"].append(val_metrics)
            
            # TensorBoard logging
            if self.tb_logger:
                for key, value in train_metrics.items():
                    self.tb_logger.log_scalar(f"train/{key}", value, epoch)
                for key, value in val_metrics.items():
                    self.tb_logger.log_scalar(f"val/{key}", value, epoch)
                self.tb_logger.log_scalar(
                    "lr",
                    self.optimizer.param_groups[0]["lr"],
                    epoch
                )
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                elif not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
            
            # Save best model
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best_model.pt", val_metrics)
                self.logger.info(f"New best model saved! Val Loss: {self.best_val_loss:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", val_metrics)
            
            # Early stopping
            if self.early_stopping(val_metrics["val_loss"]):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        
        # Save final model
        self.save_checkpoint("final_model.pt", val_metrics)
        
        # Close tensorboard
        if self.tb_logger:
            self.tb_logger.close()
        
        return {
            "history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "total_epochs": self.current_epoch + 1,
            "total_time": total_time
        }
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


if __name__ == "__main__":
    # Test trainer
    print("Testing Trainer...")
    
    from ..models import SkyPowerModel
    from ..data import create_dataloaders
    
    # Create model
    model = SkyPowerModel(
        image_backbone="resnet18",
        image_pretrained=False,
        temporal_type="lstm"
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        use_synthetic=True,
        synthetic_samples=100,
        batch_size=4,
        sequence_length=6,
        image_size=224,
        num_workers=0
    )
    
    # Config
    config = {
        "training": {
            "epochs": 2,
            "learning_rate": 0.001,
            "gradient_clip": 1.0,
            "mixed_precision": False
        },
        "loss": {"type": "mse"},
        "paths": {
            "output_dir": "./test_outputs",
            "checkpoint_dir": "./test_checkpoints",
            "log_dir": "./test_logs"
        },
        "logging": {"use_tensorboard": False}
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=config
    )
    
    # Train
    results = trainer.train()
    print(f"Training completed in {results['total_epochs']} epochs")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
