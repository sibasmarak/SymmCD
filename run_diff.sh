#!/bin/bash
#SBATCH --job-name=train-mp20
#SBATCH --partition=long
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --reservation=ubuntu2204
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --output=output/mp20-experiment-%A.%a.out

module load anaconda/3
conda activate diffcsp

# training perov
HYDRA_FULL_ERROR=1 python diffcsp/run.py expname=mp_20_temporary \
data=mp_20 data.use_random_representatives=True data.number_representatives=20 \
logging.wandb.mode=online logging.wandb.project=diffcsp_symmetry logging.val_check_interval=1 \
model=diffusion_w_site_symm model.use_ks=True model.ip=False model.use_gt_frac_coords=False model.use_site_symm=True

# generating eval_gen.pt (for Table 4 in DiffCSP paper)
# 2023-12-17/ contains the correct results for with origin as a dummy or using frac coords for perov 
python scripts/generation.py --model_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2024-01-11/mp_20_temporary --dataset mp --label final_num_samples_20
python scripts/compute_metrics.py --root_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2024-01-11/mp_20_temporary --tasks gen --gt_file data/mp_20/test.csv --label final_num_samples_20

# generating eval_diff.pt (for Table 1 in DiffCSP paper)
# python scripts/evaluate.py --model_path <path-of-run> --label num_samples_1
# python scripts/compute_metrics.py --root_path <path-of-run> --tasks csp --gt_file data/perov_5/test.csv --label num_samples_1

# python scripts/evaluate.py --model_path <path-of-run> --num_evals 20 --label num_samples_20
# python scripts/compute_metrics.py --root_path <path-of-run> --tasks csp --gt_file data/perov_5/test.csv --multi_eval --label num_samples_20 