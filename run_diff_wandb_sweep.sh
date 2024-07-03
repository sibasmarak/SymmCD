#!/bin/bash
#SBATCH --job-name=train_mp_20
#SBATCH --partition=long
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --reservation=ubuntu2204
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=mp_20_sweep_output/ab-initio-dummy-final-experiment-%A.%a.out

module load anaconda/3
conda activate diffcsp39

# To create a sweep use the corresp
# wandb sweep --project diffcsp_symmetry -e symmetry_group hyperparam_sweep_diffcsp_dummy_repr.yaml

# To run the agent of the created sweep
wandb agent symmetry_group/diffcsp_symmetry/vux1gk4j