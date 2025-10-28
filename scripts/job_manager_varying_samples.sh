#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=48:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N varying_samples_exp
#$ -t 1-165
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_varying_samples.txt"

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
mkdir -p results/exp1
mkdir -p logs/exp1
mkdir -p data

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line in jobs_varying_samples.txt:
#   <model> <dataset_size> <seed>
model=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
dataset_size=$(sed -n ${number}p "$paramfile" | awk '{print $2}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $3}')

if [[ -z "$model" || -z "$dataset_size" || -z "$seed" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running varying samples experiment: model=$model, dataset_size=$dataset_size, seed=$seed"

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
# 6.  Run single experiment
# ---------------------------------------------------------------------
echo "Starting training..."
python3.9 -u experiments/exp1_single_run.py \
    --model "$model" \
    --dataset_size "$dataset_size" \
    --seed "$seed" \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1 \
    --device cuda \
    --output_dir results/exp1 \
    --log_dir logs/exp1

date
echo "Training completed: model=$model dataset_size=$dataset_size seed=$seed"