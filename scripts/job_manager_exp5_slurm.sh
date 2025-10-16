#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp5_mlp
#SBATCH --array=1-25
#SBATCH --output=logs/exp5_mlp_%A_%a.out
#SBATCH --error=logs/exp5_mlp_%A_%a.err
set -euo pipefail

hostname
date

number=$SLURM_ARRAY_TASK_ID
paramfile="scripts/jobs_exp5.txt"

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
# Format per line in jobs_exp5.txt:
#   <n_layers> <seed>
n_layers=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $2}')

if [[ -z "$n_layers" || -z "$seed" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running exp5 MLP Variable: n_layers=$n_layers, seed=$seed"

# Define expected results path for conditional training
results_file="results/exp5/mlp_layers${n_layers}_seed${seed}_results.npz"

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
        --batch_size 512 \
        --device cuda \
        --output_dir results/exp5 \
        --epochs 200 \
        --initial_width 1024 \
        --target_width 64 \
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
        --num_workers 4

    date
    echo "Training completed: MLP n_layers=$n_layers seed=$seed"
fi

date
echo "Job completed: MLP n_layers=$n_layers seed=$seed"
