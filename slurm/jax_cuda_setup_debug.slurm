#!/bin/bash
#SBATCH -J cuda_test
#SBATCH -A MLL
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2               # good for runtime orchestration?
#SBATCH -t 00:05:00
#SBATCH -o /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/%j.cuda_test2.out
#SBATCH -e /scratch/11264/rohitchawla28/scr_reproduce_socialjax/slurm/logs/%j.cuda_test2.err

### ======== This script is what ended up working for getting JAX to see the GPU on lonestar6 with CUDA 12.2 ========


# clears any inherited modules so we are at clean slate
# doesn't affect modules globally, just for this job
module reset

# this was the key fix: JAX's cuda plugin was searching wrong/old CUDA library locations, giving the error: "Unable to load cuSPARSE"
unset LD_LIBRARY_PATH

# commented this out to just using the pip provided CUDA lib, so they don't clash
# load cuda 12 module
# module load cuda/12.2

# check cuda version, || true b/c some CUDA modules provide runtime libraries without nvcc which is fine for JAX
# nvcc --version || true

# so that conda executable is found
export PATH="/scratch/11264/rohitchawla28/scr_reproduce_socialjax/miniforge3/bin:$PATH"

# install jax with cuda 12 support
conda run -n SocialJax python -m pip uninstall -y jax jaxlib
conda run -n SocialJax python -m pip install -U "jax[cuda12]"

# check gpu
nvidia-smi

# check  visibility (this is still returning empty though but jax.devices() is working)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# check LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

# test
conda run -n SocialJax python -c "import jax; print(jax.devices())"