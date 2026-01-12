#!/usr/bin/env python
"""
Training Script

Train the Sky Power Estimation model.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from sky_power_estimation.models import SkyPowerModel
from sky_power_estimation.data import create_dataloaders
from sky_power_estimation.training import Trainer
from sky_power_estimation.utils.config import load_config, get_default_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sky Power Estimation Model")
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
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
        help="Use synthetic dataset for testing"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["resnet18", "resnet34", "resnet50", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"],
        help="Image encoder backbone"
    )
    parser.add_argument(
        "--temporal-encoder",
        type=str,
        default=None,
        choices=["lstm", "gru", "transformer"],
        help="Temporal encoder type"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Sequence length for temporal modeling"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.backbone:
        config["model"]["image_encoder"]["backbone"] = args.backbone
    if args.temporal_encoder:
        config["model"]["temporal_encoder"]["type"] = args.temporal_encoder
    if args.sequence_length:
        config["data"]["sequence_length"] = args.sequence_length
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
        config["paths"]["checkpoint_dir"] = f"{args.output_dir}/checkpoints"
        config["paths"]["log_dir"] = f"{args.output_dir}/logs"
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Sky Power Estimation Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Backbone: {config['model']['image_encoder']['backbone']}")
    print(f"Temporal Encoder: {config['model']['temporal_encoder']['type']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print("=" * 60)
    
    # Create dataloaders
    print("\nLoading data...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        use_synthetic=args.use_synthetic or args.data_dir is None,
        batch_size=config["training"]["batch_size"],
        sequence_length=config["data"]["sequence_length"],
        image_size=config["data"]["image_size"],
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Val samples: {len(dataloaders['val'].dataset)}")
    print(f"Test samples: {len(dataloaders['test'].dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = SkyPowerModel.from_config(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    results = trainer.train()
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total epochs: {results['total_epochs']}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Total time: {results['total_time'] / 60:.1f} minutes")
    print(f"Checkpoints saved to: {config['paths']['checkpoint_dir']}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    trainer.model.eval()
    
    from sky_power_estimation.utils.metrics import calculate_metrics
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloaders["test"]:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            preds = trainer._forward_pass(batch)
            all_preds.append(preds.cpu())
            all_targets.append(batch["target"].cpu())
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    test_metrics = calculate_metrics(all_targets, all_preds, prefix="test")
    
    print("\nTest Set Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
