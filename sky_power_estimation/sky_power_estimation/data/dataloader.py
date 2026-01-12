"""
DataLoader Module

DataLoader creation and utilities for sky power estimation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from .dataset import SkyPowerDataset, SyntheticSkyDataset


def create_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    sequence_length: int = 12,
    image_size: int = 224,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    use_synthetic: bool = False,
    synthetic_samples: int = 1000,
    pin_memory: bool = True,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
        sequence_length: Number of historical timesteps
        image_size: Target image size
        num_workers: Number of data loading workers
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        use_synthetic: Whether to use synthetic dataset
        synthetic_samples: Number of synthetic samples
        pin_memory: Whether to pin memory for GPU
        **kwargs: Additional arguments for dataset
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    if use_synthetic or data_dir is None:
        # Use synthetic dataset
        train_dataset = SyntheticSkyDataset(
            num_samples=synthetic_samples,
            sequence_length=sequence_length,
            image_size=image_size,
            mode="train"
        )
        
        val_dataset = SyntheticSkyDataset(
            num_samples=int(synthetic_samples * 0.2),
            sequence_length=sequence_length,
            image_size=image_size,
            mode="val"
        )
        
        test_dataset = SyntheticSkyDataset(
            num_samples=int(synthetic_samples * 0.2),
            sequence_length=sequence_length,
            image_size=image_size,
            mode="test"
        )
    else:
        # Use real dataset
        full_dataset = SkyPowerDataset(
            data_dir=data_dir,
            mode="train",
            sequence_length=sequence_length,
            image_size=image_size,
            **kwargs
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for sky power dataset.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    collated = {}
    
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        else:
            collated[key] = [sample[key] for sample in batch]
    
    return collated


class InfiniteDataLoader:
    """
    DataLoader that loops infinitely.
    
    Useful for training when you want to specify iterations instead of epochs.
    """
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
    
    def __len__(self):
        return len(self.dataloader)


class BalancedSampler(torch.utils.data.Sampler):
    """
    Sampler that balances samples across different conditions.
    
    Useful when you have imbalanced data (e.g., more clear sky than cloudy).
    """
    
    def __init__(
        self,
        dataset: Dataset,
        condition_labels: np.ndarray,
        replacement: bool = True
    ):
        self.dataset = dataset
        self.condition_labels = condition_labels
        self.replacement = replacement
        
        # Compute class weights
        unique_labels, counts = np.unique(condition_labels, return_counts=True)
        weights = 1.0 / counts
        self.sample_weights = torch.tensor(
            [weights[label] for label in condition_labels],
            dtype=torch.float32
        )
    
    def __iter__(self):
        indices = torch.multinomial(
            self.sample_weights,
            len(self.dataset),
            replacement=self.replacement
        )
        return iter(indices.tolist())
    
    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    # Test dataloaders
    print("Testing DataLoaders...")
    
    dataloaders = create_dataloaders(
        use_synthetic=True,
        synthetic_samples=200,
        batch_size=8,
        sequence_length=6,
        image_size=224,
        num_workers=0
    )
    
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    print(f"Test batches: {len(dataloaders['test'])}")
    
    # Test a batch
    batch = next(iter(dataloaders['train']))
    print(f"\nBatch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test infinite loader
    print("\nTesting InfiniteDataLoader...")
    infinite_loader = InfiniteDataLoader(dataloaders['train'])
    for i, batch in enumerate(infinite_loader):
        if i >= 3:
            break
        print(f"  Batch {i}: target shape = {batch['target'].shape}")
