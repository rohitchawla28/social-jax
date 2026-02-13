#!/bin/bash
#SBATCH -J socialjax_setup
#SBATCH -A MLL
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2               # good for runtime orchestration?
#SBATCH -t 01:00:00
#SBATCH -o /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/setup.%j.out
#SBATCH -e /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/setup.%j.err

set -euo pipefail

# set up paths
BASE="/scratch/11264/rohitchawla28/scr_reproduce_socialjax"
REPO="$BASE/SocialJax"
MINIFORGE="$BASE/miniforge3"

module reset
unset LD_LIBRARY_PATH

# conda
export PATH="$MINIFORGE/bin:$PATH"
source "$MINIFORGE/etc/profile.d/conda.sh"
conda activate SocialJax    # do i actually need this still? Because conda activate was being flaky, so i switched to conda run

# run from repo root so relative imports + relative output dirs work
# key thing is that it still finds "socialjax" package inside the REPO
cd "$REPO"

# added this because PYTHONPATH currently set to python 3.9.7 site-packages on lonestar6 so it might mess up imports
# unsetting it inside the job doesn't change it globally
unset PYTHONPATH

# make sure no cache data going on $HOME
export PIP_CACHE_DIR="$BASE/.pip_cache"
export XDG_CACHE_HOME="$BASE/.cache"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

# install dependencies (also switched to conda run b/c was being flaky before)
conda run -n SocialJax python -m pip install -r requirements_updated.txt

# install CUDA12-enabled JAX (socialJax used CUDA11-enabled version)
conda run -n SocialJax python -m pip install -U "jax[cuda12]"

# final checks
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
conda run -n SocialJax python -c "import jax; print('jax', jax.__version__); print(jax.devices())"
