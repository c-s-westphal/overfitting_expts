#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N exp3_vgg19var
#$ -t 1-39
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_exp3_vgg19var.txt"

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
mkdir -p results/exp3
mkdir -p logs/exp3
mkdir -p data

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line in jobs_exp3_vgg19var.txt:
#   <n_layers> <seed>
n_layers=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $2}')

if [[ -z "$n_layers" || -z "$seed" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running exp3 VGG19 Variable: n_layers=$n_layers, seed=$seed"

# ---------------------------------------------------------------------
# 5.  Run single experiment
# ---------------------------------------------------------------------
echo "Starting training..."
python3.9 -u experiments/exp3_single_run.py \
    --arch vgg19var \
    --n_layers "$n_layers" \
    --mi_bits 2.5 \
    --seed "$seed" \
    --batch_size 128 \
    --device cuda \
    --output_dir results/exp3

date
echo "Training completed: VGG19 n_layers=$n_layers seed=$seed"
