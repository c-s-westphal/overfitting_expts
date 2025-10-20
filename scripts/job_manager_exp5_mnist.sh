#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N exp5_mlp_mnist
#$ -t 1-45
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_exp5_mnist.txt"

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
mkdir -p results/exp5
mkdir -p logs/exp5
mkdir -p data

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line in jobs_exp5_mnist.txt:
#   <n_layers> <seed> <dataset>
n_layers=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $2}')
dataset=$(sed -n ${number}p "$paramfile" | awk '{print $3}')

if [[ -z "$n_layers" || -z "$seed" || -z "$dataset" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running exp5 MLP Variable MNIST: n_layers=$n_layers, seed=$seed, dataset=$dataset"

# ---------------------------------------------------------------------
# 5.  Run single experiment
# ---------------------------------------------------------------------
echo "Starting training..."
python3.9 -u experiments/exp5_single_run.py \
    --n_layers "$n_layers" \
    --seed "$seed" \
    --dataset "$dataset" \
    --batch_size 512 \
    --device cuda \
    --output_dir results/exp5 \
    --epochs 500 \
    --hidden_dim 8 \
    --dropout 0.3 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --warmup_epochs 5 \
    --label_smoothing 0.1 \
    --eval_interval 10 \
    --n_masks_train 20 \
    --n_masks_final 40 \
    --max_eval_batches_train 20 \
    --max_eval_batches_final 40 \
    --num_workers 0

date
echo "Training completed: MLP n_layers=$n_layers seed=$seed dataset=$dataset"
