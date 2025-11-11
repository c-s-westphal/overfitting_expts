#!/usr/bin/env python3
"""
Create identity_CelebA.txt file for torchvision.
Kaggle doesn't provide this, so we create a dummy one.
"""
import os

celeba_dir = './data/celeba'
img_dir = os.path.join(celeba_dir, 'img_align_celeba')
identity_file = os.path.join(celeba_dir, 'identity_CelebA.txt')

print("Creating identity_CelebA.txt...")

# Get all image filenames
images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

# Create identity file with dummy identity IDs (just sequential numbers)
with open(identity_file, 'w') as f:
    for i, img in enumerate(images):
        # Format: image_name identity_id
        # We'll just use sequential IDs since we don't actually need real identities
        identity_id = i + 1
        f.write(f"{img} {identity_id}\n")

print(f"Created {identity_file} with {len(images)} entries")
print("You can now run your experiments!")
