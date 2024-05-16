#!/bin/bash
#SBATCH --job-name=mp20_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-20
#SBATCH --output=mp_20_eval_output/mp20-eval-experiment-%A.%a.out

module load anaconda/3
conda activate diffcsp39

model_paths=() # add model paths here

# generating eval_gen.pt (for Table 4 in DiffCSP paper, ab-initio generation) 
python scripts/generation.py --model_path ${model_paths[index]} --dataset mp
python scripts/compute_metrics.py --root_path ${model_paths[index]} --tasks gen --gt_file data/mp_20/test.csv