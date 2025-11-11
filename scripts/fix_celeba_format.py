#!/usr/bin/env python3
"""
Convert Kaggle CelebA CSV files to TXT format expected by torchvision.
"""
import os
import pandas as pd

celeba_dir = './data/celeba'

print("Converting CSV files to TXT format for torchvision...")

# Convert list_attr_celeba.csv to list_attr_celeba.txt
csv_file = os.path.join(celeba_dir, 'list_attr_celeba.csv')
txt_file = os.path.join(celeba_dir, 'list_attr_celeba.txt')
if os.path.exists(csv_file):
    print(f"Converting {csv_file}...")
    df = pd.read_csv(csv_file)
    # Write in the format torchvision expects (space-separated, with header)
    with open(txt_file, 'w') as f:
        # First line: number of images
        f.write(f"{len(df)}\n")
        # Second line: attribute names
        attrs = ' '.join(df.columns[1:])  # Skip image_id column
        f.write(f"{attrs}\n")
        # Data lines: image_id followed by attributes (-1 or 1)
        for _, row in df.iterrows():
            img_id = row['image_id']
            attrs = ' '.join([str(int(row[col])) if row[col] in [-1, 1] else str(row[col]) for col in df.columns[1:]])
            f.write(f"{img_id} {attrs}\n")
    print(f"Created {txt_file}")

# Convert list_bbox_celeba.csv to list_bbox_celeba.txt
csv_file = os.path.join(celeba_dir, 'list_bbox_celeba.csv')
txt_file = os.path.join(celeba_dir, 'list_bbox_celeba.txt')
if os.path.exists(csv_file):
    print(f"Converting {csv_file}...")
    df = pd.read_csv(csv_file)
    with open(txt_file, 'w') as f:
        f.write(f"{len(df)}\n")
        # Header
        f.write("image_id x_1 y_1 width height\n")
        # Data
        for _, row in df.iterrows():
            f.write(f"{row['image_id']} {row['x_1']} {row['y_1']} {row['width']} {row['height']}\n")
    print(f"Created {txt_file}")

# Convert list_landmarks_align_celeba.csv to list_landmarks_align_celeba.txt
csv_file = os.path.join(celeba_dir, 'list_landmarks_align_celeba.csv')
txt_file = os.path.join(celeba_dir, 'list_landmarks_align_celeba.txt')
if os.path.exists(csv_file):
    print(f"Converting {csv_file}...")
    df = pd.read_csv(csv_file)
    with open(txt_file, 'w') as f:
        f.write(f"{len(df)}\n")
        # Write header
        cols = ' '.join(df.columns)
        f.write(f"{cols}\n")
        # Data
        for _, row in df.iterrows():
            line = ' '.join([str(row[col]) for col in df.columns])
            f.write(f"{line}\n")
    print(f"Created {txt_file}")

# Convert list_eval_partition.csv to list_eval_partition.txt
csv_file = os.path.join(celeba_dir, 'list_eval_partition.csv')
txt_file = os.path.join(celeba_dir, 'list_eval_partition.txt')
if os.path.exists(csv_file):
    print(f"Converting {csv_file}...")
    df = pd.read_csv(csv_file)
    with open(txt_file, 'w') as f:
        # No header count for this file
        for _, row in df.iterrows():
            f.write(f"{row['image_id']} {row['partition']}\n")
    print(f"Created {txt_file}")

print("\nConversion complete! Files created:")
for fname in ['list_attr_celeba.txt', 'list_bbox_celeba.txt',
              'list_landmarks_align_celeba.txt', 'list_eval_partition.txt']:
    fpath = os.path.join(celeba_dir, fname)
    if os.path.exists(fpath):
        print(f"  ✓ {fname}")
    else:
        print(f"  ✗ {fname} (MISSING)")

print("\nYou can now run your experiments!")
