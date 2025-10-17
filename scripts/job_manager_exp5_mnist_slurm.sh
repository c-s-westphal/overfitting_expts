#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp5_mlp_mnist
#SBATCH --array=1-50
#SBATCH --output=logs/exp5_mlp_mnist_%A_%a.out
#SBATCH --error=logs/exp5_mlp_mnist_%A_%a.err
set -euo pipefail

hostname
date

number=$SLURM_ARRAY_TASK_ID
paramfile="scripts/jobs_exp5_mnist.txt"

# ---------------------------------------------------------------------
# 1.  Activate virtual environment
# ---------------------------------------------------------------------
# Activate venv using full path
source ~/vgg11_dropout_robustness/venv/bin/activate

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

# Define expected results path for conditional training
results_file="results/exp5/mlp_${dataset}_layers${n_layers}_seed${seed}_results.npz"

# ---------------------------------------------------------------------
# 5.  Train (conditional on results existence)
# ---------------------------------------------------------------------
if [ -f "$results_file" ]; then
    echo "Results already exist; skipping training."
    echo "  Results: $results_file"
else
    echo "Starting training..."
    python -u experiments/exp5_single_run.py \
        --n_layers "$n_layers" \
        --seed "$seed" \
        --dataset "$dataset" \
        --batch_size 512 \
        --device cuda \
        --output_dir results/exp5 \
        --epochs 500 \
        --hidden_dim 256 \
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
fi

date
echo "Job completed: MLP n_layers=$n_layers seed=$seed dataset=$dataset"
