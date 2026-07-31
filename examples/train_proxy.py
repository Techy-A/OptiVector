"""
Training script for the real-time Neural Proxy in CurveSplat.
Loads the collected .npz samples and trains the LightweightUNet model.
"""

import sys
import os
import argparse
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from curvesplat.proxy import LightweightUNet


class ProxyDataset(Dataset):
    """
    Loads CurveSplat proxy training samples from .npz files.
    """

    def __init__(self, file_paths: list) -> None:
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        path = self.file_paths[idx]
        with np.load(path) as data:
            target = torch.from_numpy(data["target"]).float()
            current_render = torch.from_numpy(data["current_render"]).float()
            final_render = torch.from_numpy(data["final_render"]).float()

        # Concatenate target image and current render as input
        # target is (3, H, W), current_render is (3, H, W) -> input is (6, H, W)
        x = torch.cat([target, current_render], dim=0)
        return x, final_render


def train_proxy(
    dataset_dir: str,
    output_path: str,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = "cpu",
):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Find all training samples
    file_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))

    if not file_paths:
        print(f"Warning: No dataset samples found in '{dataset_dir}'.")
        print("Generating a synthetic dataset for verification/dummy testing...")
        
        # Create a synthetic dataset folder
        os.makedirs(dataset_dir, exist_ok=True)
        for i in range(10):
            mock_target = np.random.rand(3, 256, 256).astype(np.float32)
            mock_current = np.random.rand(3, 256, 256).astype(np.float32)
            mock_final = np.random.rand(3, 256, 256).astype(np.float32)
            np.savez_compressed(
                os.path.join(dataset_dir, f"sample_{i:06d}.npz"),
                target=mock_target,
                current_render=mock_current,
                final_render=mock_final,
            )
        file_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.npz")))

    # Split train/val
    np.random.seed(42)
    indices = np.random.permutation(len(file_paths))
    val_size = int(len(file_paths) * val_split)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_files = [file_paths[i] for i in train_indices]
    val_files = [file_paths[i] for i in val_indices]

    train_dataset = ProxyDataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_loader = None
    if val_files:
        val_dataset = ProxyDataset(val_files)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("=" * 65)
    print("  CurveSplat — Neural Proxy Training Pipeline")
    print("=" * 65)
    print(f"  Dataset directory: {dataset_dir}")
    print(f"  Total samples:     {len(file_paths)}")
    print(f"  Train samples:     {len(train_dataset)}")
    print(f"  Val samples:       {len(val_dataset) if val_loader else 0}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Epochs:            {epochs}")
    print(f"  Learning rate:     {lr}")
    print(f"  Device:            {device.upper()}")
    print("=" * 65)

    # Initialize model
    model = LightweightUNet(in_channels=6, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t_start = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_dataset)
        t_epoch = time.time() - t_start

        # Validation phase
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_dataset)

            print(f"Epoch {epoch:02d}/{epochs:02d} | Train L1: {train_loss:.5f} | Val L1: {val_loss:.5f} | Time: {t_epoch:.2f}s")
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_path)
                print(f"  → Best model checkpoint saved to: {output_path}")
        else:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train L1: {train_loss:.5f} | Time: {t_epoch:.2f}s")
            torch.save(model.state_dict(), output_path)

    print("=" * 65)
    print(f"✓ Training complete! Weights exported to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CurveSplat neural proxy model")
    parser.add_argument("--dataset_dir", type=str, default=".tmp/proxy_dataset",
                        help="Directory containing compressed dataset .npz samples")
    parser.add_argument("--output_path", type=str, default="proxy_weights.pt",
                        help="Output path to save trained PyTorch model weights")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Hardware device (cpu or cuda)")
    args = parser.parse_args()

    train_proxy(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
