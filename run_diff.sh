#!/bin/bash
#SBATCH --job-name=perov_eval_table4
#SBATCH --partition=long
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=output/correct-eval-perov-kvector-sitesymm-abgen.out

# experiment-%A.%a.out

module load anaconda/3
conda activate diffcsp

# training perov
# HYDRA_FULL_ERROR=1 python diffcsp/run.py data=perov expname=perov_w_symm_type logging.wandb.mode=offline logging.val_check_interval=1 model=diffusion_w_site_symm model.use_ks=True model.ip=False

# generating eval_diff.pt (for Table 1 in DiffCSP paper)
# python scripts/evaluate.py --model_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-11-10/perov_csp --label num_samples_1
# python scripts/compute_metrics.py --root_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-11-10/perov_csp --tasks csp --gt_file data/perov_5/test.csv --label num_samples_1

# python scripts/evaluate.py --model_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-11-10/perov_csp --num_evals 20 --label num_samples_20
# python scripts/compute_metrics.py --root_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-11-10/perov_csp --tasks csp --gt_file data/perov_5/test.csv --multi_eval --label num_samples_20

# generating eval_gen.pt (for Table 4 in DiffCSP paper)
# python scripts/generation.py --model_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-11-15/mp20_abgen --dataset mp
# python scripts/compute_metrics.py --root_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-11-15/mp20_abgen --tasks gen --gt_file data/mp_20/test.csv

# generation with site symmetries
python scripts/generation.py --model_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-12-06/perov_w_symm_type --dataset perov --label final_num_samples_20
python scripts/compute_metrics.py --root_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-12-06/perov_w_symm_type --tasks gen --gt_file data/perov_5/test.csv --label final_num_samples_20