#!/usr/bin/env python3
"""
Download CelebA from Kaggle instead of Google Drive.
Requires Kaggle API credentials.

Setup:
1. Create Kaggle account and get API token from https://www.kaggle.com/settings
2. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars
"""
import os
import sys
import time
import zipfile
import shutil

print("=" * 80)
print("CelebA Dataset Download from Kaggle")
print("=" * 80)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

start_time = time.time()

# Target directory
data_dir = './data'
celeba_dir = os.path.join(data_dir, 'celeba')
os.makedirs(celeba_dir, exist_ok=True)

print("\nDownloading CelebA from Kaggle...")
print("This may take 10-20 minutes.\n")

try:
    import subprocess

    # Use kaggle CLI to download (without --unzip, we'll handle that manually)
    print("Downloading dataset archive...")
    result = subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', 'jessicali9530/celeba-dataset', '-p', celeba_dir],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise Exception(f"Kaggle download failed: {result.stderr}")

    print("Download complete. Extracting files...")

    # Extract the zip file
    zip_path = os.path.join(celeba_dir, 'celeba-dataset.zip')
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(celeba_dir)
        os.remove(zip_path)
        print("Extraction complete.")

    print("\nReorganizing files to match torchvision CelebA structure...")

    # Handle nested directory structure from Kaggle
    kaggle_img_dir = os.path.join(celeba_dir, 'img_align_celeba', 'img_align_celeba')
    target_img_dir = os.path.join(celeba_dir, 'img_align_celeba')

    if os.path.exists(kaggle_img_dir):
        print("Moving images to correct location...")
        for file in os.listdir(kaggle_img_dir):
            src = os.path.join(kaggle_img_dir, file)
            dst = os.path.join(target_img_dir, file)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        # Remove empty directory
        try:
            os.rmdir(kaggle_img_dir)
        except:
            pass

    elapsed_time = time.time() - start_time
    elapsed_mins = elapsed_time / 60

    print("\n" + "=" * 80)
    print("CelebA download complete!")
    print("=" * 80)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed_mins:.1f} minutes ({elapsed_time:.0f} seconds)")
    print(f"Data stored in: {os.path.abspath(celeba_dir)}")
    print("\nYou can now run your experiments.")
    print("=" * 80)

except Exception as e:
    print(f"\nERROR: {e}")
    print("\nMake sure you have:")
    print("1. Kaggle account")
    print("2. API credentials in ~/.kaggle/kaggle.json")
    print("3. Accepted the dataset terms at: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
    sys.exit(1)
