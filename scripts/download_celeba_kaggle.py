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
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Initialize Kaggle API
    print("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    # Use kaggle API to download
    print("Downloading dataset archive...")
    api.dataset_download_files('jessicali9530/celeba-dataset', path=celeba_dir, unzip=False)
    print("Download complete!")

    print("Download complete. Extracting files...")

    # Extract the zip file
    zip_path = os.path.join(celeba_dir, 'celeba-dataset.zip')
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(celeba_dir)
        os.remove(zip_path)
        print("Extraction complete.")

    print("\nReorganizing files to match torchvision CelebA structure...")

    # torchvision expects: data/celeba/img_align_celeba/, data/celeba/list_attr_celeba.txt, etc.
    # Kaggle gives us various nested structures

    # Move all annotation files to celeba root
    for root, dirs, files in os.walk(celeba_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.csv'):
                src = os.path.join(root, file)
                dst = os.path.join(celeba_dir, file)
                if src != dst and not os.path.exists(dst):
                    print(f"Moving {file} to celeba root...")
                    shutil.copy2(src, dst)

    # Handle nested img_align_celeba directory structure
    kaggle_img_dir = os.path.join(celeba_dir, 'img_align_celeba', 'img_align_celeba')
    target_img_dir = os.path.join(celeba_dir, 'img_align_celeba')

    if os.path.exists(kaggle_img_dir):
        print("Moving images to correct location...")
        os.makedirs(target_img_dir, exist_ok=True)
        for file in os.listdir(kaggle_img_dir):
            if file.endswith('.jpg'):
                src = os.path.join(kaggle_img_dir, file)
                dst = os.path.join(target_img_dir, file)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
        # Remove empty nested directory
        try:
            os.rmdir(kaggle_img_dir)
        except:
            pass

    # List what we have
    print("\nChecking downloaded files:")
    print(f"  Images in {target_img_dir}: {len([f for f in os.listdir(target_img_dir) if f.endswith('.jpg')])}")
    annotation_files = [f for f in os.listdir(celeba_dir) if f.endswith('.txt')]
    print(f"  Annotation files: {annotation_files}")

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
