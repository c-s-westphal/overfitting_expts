#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N noise_pixel_exp
#$ -t 1-180
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_noise_pixel.txt"

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
mkdir -p results/exp2
mkdir -p logs/exp2
mkdir -p data

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line in jobs_noise_pixel.txt:
#   <model> <noise_level> <seed>
model=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
noise_level=$(sed -n ${number}p "$paramfile" | awk '{print $2}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $3}')

if [[ -z "$model" || -z "$noise_level" || -z "$seed" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running noise pixel experiment: model=$model, noise_level=$noise_level, seed=$seed"

# ---------------------------------------------------------------------
# 5.  Run single experiment
# ---------------------------------------------------------------------
echo "Starting training..."
python3.9 -u experiments/exp2_single_run.py \
    --model "$model" \
    --noise_level "$noise_level" \
    --seed "$seed" \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1 \
    --device cuda \
    --output_dir results/exp2 \
    --log_dir logs/exp2

date
echo "Training completed: model=$model noise_level=$noise_level seed=$seed"