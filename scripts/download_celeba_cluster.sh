#!/bin/bash
#$ -l tmem=8G
#$ -l h_rt=2:00:00
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N download_celeba
set -euo pipefail

hostname
date

echo "Pre-downloading CelebA dataset..."

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

# Create data directory
mkdir -p data

# Redirect HOME to TMPDIR to avoid home quota issues with gdown cookies
export HOME="$TMPDIR/home"
mkdir -p "$HOME/.cache/gdown"

# Run download script with unbuffered output so we can see progress
python3.9 -u scripts/download_celeba.py

date
echo "Download complete!"
echo ""
echo "You can now submit your exp7 jobs:"
echo "  qsub scripts/job_manager_exp7.sh"
echo "  qsub scripts/job_manager_exp7_5epochs.sh"
