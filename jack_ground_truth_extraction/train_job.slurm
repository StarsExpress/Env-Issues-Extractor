#!/bin/bash
#SBATCH -J env-extract
#SBATCH -o slurm-%j.out
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=40G
# If your cluster requires explicit partition/QoS, uncomment:
# #SBATCH -p coe-gpu
# #SBATCH --qos=coe-ice


# Navigate to project directory

# --- Conda setup ---
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate environmental-extraction

# Quick sanity checks
which python
python -V
nvidia-smi || true

# Launch training
srun --export=ALL bash -x run.sh
