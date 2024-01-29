#!/bin/bash
#SBATCH --job-name=mp20_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --reservation=ubuntu2204
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=mp_20_sweep_output/mp20-eval-super-hill-214-experiment-%A.%a.out

module load anaconda/3
conda activate diffcsp

# training
# HYDRA_FULL_ERROR=1 python diffcsp/run.py expname=mp_20_actual_repr data=mp_20 data.number_representatives=0 data.train_max_epochs=100000 \
# logging.wandb.mode=offline logging.wandb.project=diffcsp_symmetry logging.val_check_interval=1 \
# model=diffusion_w_site_symm model.use_ks=True model.ip=False model.use_gt_frac_coords=True model.use_site_symm=True \
# model.decoder.hidden_dim=1024 model.decoder.num_layers=10

# generating eval_gen.pt (for Table 4 in DiffCSP paper, ab-initio generation) 
python scripts/generation.py --model_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2024-01-27/mp_20_actual_repr/super-hill-214/ --dataset mp --label final_num_samples_20
python scripts/compute_metrics.py --root_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2024-01-27/mp_20_actual_repr/super-hill-214/ --tasks gen --gt_file data/mp_20/test.csv --label final_num_samples_20

# generating eval_diff.pt (for Table 1 in DiffCSP paper, crystal structure prediction (csp) task)
# python scripts/evaluate.py --model_path <path-of-run> --label num_samples_1
# python scripts/compute_metrics.py --root_path <path-of-run> --tasks csp --gt_file data/perov_5/test.csv --label num_samples_1

# python scripts/evaluate.py --model_path <path-of-run> --num_evals 20 --label num_samples_20
# python scripts/compute_metrics.py --root_path <path-of-run> --tasks csp --gt_file data/perov_5/test.csv --multi_eval --label num_samples_20 