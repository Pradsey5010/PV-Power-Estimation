#!/usr/bin/env python
"""
Evaluation Script

Evaluate a trained model on test data.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from sky_power_estimation.models import SkyPowerModel
from sky_power_estimation.data import create_dataloaders
from sky_power_estimation.utils.metrics import (
    calculate_metrics,
    calculate_metrics_by_condition,
    skill_score
)
from sky_power_estimation.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sky Power Estimation Model")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory"
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Sky Power Estimation Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    
    # Load checkpoint
    print("\nLoading model...")
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint.get("config", {})
    
    # Create model
    model = SkyPowerModel.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    # Create dataloader
    print("\nLoading data...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        use_synthetic=args.use_synthetic or args.data_dir is None,
        batch_size=args.batch_size,
        sequence_length=config.get("data", {}).get("sequence_length", 12),
        image_size=config.get("data", {}).get("image_size", 224),
        num_workers=4
    )
    
    test_loader = dataloaders["test"]
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    print("\nRunning evaluation...")
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            predictions = model(
                current_image=batch["current_image"],
                image_sequence=batch.get("image_sequence"),
                current_weather=batch["current_weather"],
                weather_sequence=batch.get("weather_sequence"),
                current_sun_position=batch["current_sun_position"],
                sun_position_sequence=batch.get("sun_position_sequence")
            )
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch["target"].cpu().numpy())
    
    # Concatenate
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    metrics = calculate_metrics(all_targets, all_predictions)
    
    print("\nOverall Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  nRMSE: {metrics['nrmse']:.2f}%")
    print(f"  MBE: {metrics['mbe']:.4f}")
    
    # Skill score vs persistence
    persistence = np.roll(all_targets, 1)
    persistence[0] = all_targets[0]
    ss = skill_score(all_targets, all_predictions, persistence)
    print(f"  Skill Score (vs persistence): {ss:.4f}")
    
    # Error statistics
    errors = all_predictions - all_targets
    print("\nError Statistics:")
    print(f"  Mean error: {np.mean(errors):.4f}")
    print(f"  Std error: {np.std(errors):.4f}")
    print(f"  Min error: {np.min(errors):.4f}")
    print(f"  Max error: {np.max(errors):.4f}")
    print(f"  Median error: {np.median(errors):.4f}")
    
    # Error percentiles
    print("\nError Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(np.abs(errors), p):.4f}")
    
    # Save predictions if requested
    if args.output:
        import pandas as pd
        
        df = pd.DataFrame({
            "target": all_targets,
            "prediction": all_predictions,
            "error": errors,
            "abs_error": np.abs(errors)
        })
        df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")


if __name__ == "__main__":
    main()
