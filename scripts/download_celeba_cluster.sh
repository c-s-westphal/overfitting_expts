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

# Run download script
python3.9 scripts/download_celeba.py

date
echo "Download complete!"
