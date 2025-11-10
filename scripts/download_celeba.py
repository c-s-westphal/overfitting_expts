#!/usr/bin/env python3
"""
Pre-download CelebA dataset before running experiments.
This avoids multiple jobs competing to download the same data.
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
import torchvision.datasets

print("Downloading CelebA dataset...")
print("This may take 10-20 minutes depending on your connection.")
print("=" * 80)

# Simple transform just for downloading
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
])

# Download train split
print("\nDownloading training split...")
train_dataset = torchvision.datasets.CelebA(
    root='./data',
    split='train',
    target_type='attr',
    download=True,
    transform=transform
)
print(f"Train set size: {len(train_dataset)}")

# Download validation split
print("\nDownloading validation split...")
val_dataset = torchvision.datasets.CelebA(
    root='./data',
    split='valid',
    target_type='attr',
    download=True,
    transform=transform
)
print(f"Validation set size: {len(val_dataset)}")

print("\n" + "=" * 80)
print("CelebA download complete!")
print(f"Data stored in: {os.path.abspath('./data/celeba')}")
print("You can now run your experiments.")
