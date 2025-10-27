#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=48:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N exp6_vgg_5ep
#$ -t 1-15
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_exp6_5epochs.txt"

# ---------------------------------------------------------------------
# 1.  Load toolchains and activate virtual-env
# ---------------------------------------------------------------------
if command -v source >/dev/null 2>&1; then
  source /share/apps/source_files/python/python-3.9.5.source || true
  source /share/apps/source_files/cuda/cuda-11.8.source || true
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
else
  if [[ -f /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate ]]; then
    source /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate
    export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
  fi
fi

# ---------------------------------------------------------------------
# 2.  Keep Matplotlib out of home quota
# ---------------------------------------------------------------------
export MPLCONFIGDIR="$TMPDIR/mplcache"
mkdir -p "$MPLCONFIGDIR"

# ---------------------------------------------------------------------
# 3.  Create output directories
# ---------------------------------------------------------------------
mkdir -p results/exp6
mkdir -p logs/exp6
mkdir -p data

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line in jobs_exp6_5epochs.txt:
#   <arch> <seed>
arch=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $2}')

if [[ -z "$arch" || -z "$seed" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running exp6 Full-Depth VGG (5 epochs): arch=$arch, seed=$seed"

# ---------------------------------------------------------------------
# 5.  Check GPU availability before running
# ---------------------------------------------------------------------
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
  nvidia-smi
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
else
  echo "nvidia-smi not available"
fi

# ---------------------------------------------------------------------
# 6.  Run single experiment with 5 epochs
# ---------------------------------------------------------------------
echo "Starting training for 5 epochs..."
python3.9 -u experiments/exp6_single_run.py \
    --arch "$arch" \
    --seed "$seed" \
    --epochs 5 \
    --batch_size 128 \
    --device cuda \
    --output_dir results/exp6

date
echo "Training completed: $arch seed=$seed (5 epochs)"
