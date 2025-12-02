#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=12:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N exp9_pruning_retrain
#$ -t 1-80
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_exp9.txt"

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
mkdir -p results/exp9
mkdir -p logs/exp9
mkdir -p data

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line in jobs_exp9.txt:
#   <seed> <n_layers> <dropout_base> <dataset>
seed=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
n_layers=$(sed -n ${number}p "$paramfile" | awk '{print $2}')
dropout_base=$(sed -n ${number}p "$paramfile" | awk '{print $3}')
dataset=$(sed -n ${number}p "$paramfile" | awk '{print $4}')

if [[ -z "$seed" || -z "$n_layers" || -z "$dropout_base" || -z "$dataset" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running exp9 Retraining-based Pruning: seed=$seed n_layers=$n_layers dropout_base=$dropout_base dataset=$dataset"

# ---------------------------------------------------------------------
# 5.  Run single experiment
# ---------------------------------------------------------------------
echo "Starting training..."
python3.9 -u experiments/exp9_single_run.py \
    --seed "$seed" \
    --dataset "$dataset" \
    --n_layers "$n_layers" \
    --neurons_per_layer 4 \
    --batch_size 128 \
    --device cuda \
    --output_dir results/exp9 \
    --max_epochs 200 \
    --lr 0.001 \
    --dropout_base "$dropout_base"

date
echo "Training completed: exp9 seed=$seed n_layers=$n_layers dropout_base=$dropout_base dataset=$dataset"
