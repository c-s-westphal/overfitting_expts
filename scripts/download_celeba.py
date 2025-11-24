#!/usr/bin/env python3
"""
Pre-download CelebA dataset before running experiments.
This avoids multiple jobs competing to download the same data.
"""
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
import torchvision.datasets

print("=" * 80)
print("CelebA Dataset Download")
print("=" * 80)
print("This may take 10-30 minutes depending on your connection.")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("\nNOTE: The download will appear to hang at times - this is normal.")
print("CelebA downloads multiple large zip files sequentially.")
print("Please be patient...\n")

start_time = time.time()

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

elapsed_time = time.time() - start_time
elapsed_mins = elapsed_time / 60

print("\n" + "=" * 80)
print("CelebA download complete!")
print("=" * 80)
print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total time: {elapsed_mins:.1f} minutes ({elapsed_time:.0f} seconds)")
print(f"Data stored in: {os.path.abspath('./data/celeba')}")
print("\nYou can now run your experiments.")
print("=" * 80)
