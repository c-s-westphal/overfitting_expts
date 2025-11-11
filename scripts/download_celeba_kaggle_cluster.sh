#!/bin/bash
#$ -l tmem=8G
#$ -l h_rt=2:00:00
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N download_celeba_kaggle
set -euo pipefail

hostname
date

echo "Downloading CelebA from Kaggle..."

# Load toolchains and activate virtual-env
if command -v source >/dev/null 2>&1; then
  source /share/apps/source_files/python/python-3.9.5.source || true
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
else
  if [[ -f /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate ]]; then
    source /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate
  fi
fi

# Install kaggle package if not present
pip list | grep -q kaggle || pip install --target=/SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/lib/python3.9/site-packages kaggle

# Create data directory
mkdir -p data

# Set up Kaggle credentials directory in TMPDIR (to avoid home quota)
export KAGGLE_CONFIG_DIR="$TMPDIR/kaggle"
mkdir -p "$KAGGLE_CONFIG_DIR"

# Copy kaggle.json from project directory or home
if [[ -f ".kaggle/kaggle.json" ]]; then
  cp ".kaggle/kaggle.json" "$KAGGLE_CONFIG_DIR/"
  chmod 600 "$KAGGLE_CONFIG_DIR/kaggle.json"
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  cp "$HOME/.kaggle/kaggle.json" "$KAGGLE_CONFIG_DIR/"
  chmod 600 "$KAGGLE_CONFIG_DIR/kaggle.json"
fi

# Redirect HOME to TMPDIR to avoid quota issues
export HOME="$TMPDIR/home"
mkdir -p "$HOME"

# Run Kaggle download script
python3.9 -u scripts/download_celeba_kaggle.py

date
echo "Download complete!"
echo ""
echo "You can now submit your exp7 jobs:"
echo "  qsub scripts/job_manager_exp7.sh"
echo "  qsub scripts/job_manager_exp7_5epochs.sh"
