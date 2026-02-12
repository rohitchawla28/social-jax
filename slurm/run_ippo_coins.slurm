#!/bin/bash
#SBATCH -J ippo_coins
#SBATCH -A MLL
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2               # good for runtime orchestration?
#SBATCH -t 08:00:00
##SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -o /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/ippo_coins.%j.out
#SBATCH -e /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/ippo_coins.%j.err

set -euo pipefail

# clears any inherited modules so we are at clean slate
# doesn't affect modules globally, just for this job
module reset

# this was the key fix: JAX's cuda plugin was searching wrong/old CUDA library locations, giving the error: "Unable to load cuSPARSE"
unset LD_LIBRARY_PATH

# pin to 1 GPU?
# export CUDA_VISIBLE_DEVICES=0

BASE="/scratch/11264/rohitchawla28/scr_reproduce_socialjax"
REPO="$BASE/SocialJax"
MINIFORGE="$BASE/miniforge3"

# so that conda executable is found
export PATH="$MINIFORGE/bin:$PATH"

# move into repo
cd "$REPO"

# added this because PYTHONPATH currently set so it might mess up imports
unset PYTHONPATH

# make sure imports work (main fix after removing the hardcoded sys.path)
export PYTHONPATH="$REPO"

# WandB local files
export WANDB_DIR="$BASE/wandb"
export WANDB_CACHE_DIR="$BASE/wandb_cache"
export WANDB_CONFIG_DIR="$BASE/wandb_config"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# will delete and make new WandB key, just need to get past the errors for now
export WANDB_API_KEY="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"

# other caches to make sure nothing going on $HOME
export XDG_CACHE_HOME="$BASE/.cache"
export HF_HOME="$BASE/hf"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME"

# run
python "$REPO/algorithms/IPPO/ippo_cnn_coins.py"
