"""
FI-2010 LOB Dataset — PyTorch DataLoader

Dataset format (HuggingFace shanehans/FI2010):
  - Columns "0"–"143": pre-normalized features (z-scored)
  - Columns "144"–"148": labels for horizons k=1,2,3,5,10 (values 1/2/3 = down/stable/up)
  - Train split: 362,400 samples (days 1-7)
  - Test split: 31,937 samples (day 10)

We use only first 40 features (raw 10-level LOB: prices + volumes).
Labels are shifted to 0-indexed (0/1/2) for PyTorch CrossEntropyLoss.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class FI2010Dataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, window: int = 50):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.window = window

    def __len__(self):
        return len(self.features) - self.window + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window]  # (window, 40)
        y = self.labels[idx + self.window - 1]       # scalar
        return x, y


def load_fi2010(horizon: int = 3, window: int = 50, batch_size: int = 256, n_features: int = 40):
    """
    Args:
        horizon: label column index (0=k1, 1=k2, 2=k3, 3=k5, 4=k10)
        window:  sliding window size
        batch_size: batch size for DataLoaders
        n_features: number of features to use (40 = raw LOB only)

    Returns:
        train_loader, val_loader, test_loader
    """

    print("Loading FI-2010 from HuggingFace...")
    ds = load_dataset("shanehans/FI2010")

    train_data = ds["train"].to_pandas()
    test_data = ds["test"].to_pandas()

    feature_cols = [str(i) for i in range(n_features)]
    label_col = str(144 + horizon)

    n_train = int(len(train_data) * 0.8)

    train_features = train_data[feature_cols].values.astype(np.float32)
    train_labels = train_data[label_col].values.astype(np.int64) - 1  # shift 1/2/3 → 0/1/2 to work with pytorch cross-entropy loss

    val_features = train_features[n_train:]
    val_labels = train_labels[n_train:]

    train_features = train_features[:n_train]
    train_labels = train_labels[:n_train]

    test_features = test_data[feature_cols].values.astype(np.float32)
    test_labels = test_data[label_col].values.astype(np.int64) - 1

    train_dataset = FI2010Dataset(train_features, train_labels, window)
    val_dataset = FI2010Dataset(val_features, val_labels, window)
    test_dataset = FI2010Dataset(test_features, test_labels, window)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset):,} samples")
    print(f"Val:   {len(val_dataset):,} samples")
    print(f"Test:  {len(test_dataset):,} samples")
    print(f"Features: {n_features} (raw LOB)")
    print(f"Window: {window}")
    print(f"Horizon: k={[1,2,3,5,10][horizon]}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_fi2010()

    # Verify shapes
    for x, y in train_loader:
        print(f"\nBatch input shape:  {x.shape}")   # (256, 50, 40)
        print(f"Batch label shape:  {y.shape}")     # (256,)
        print(f"Label values: {y[:10]}")
        print(f"Label distribution: {torch.bincount(y)}")  # [down, stable, up]
        break
