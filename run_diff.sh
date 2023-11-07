#!/bin/bash
#SBATCH --job-name=eval_perov_w_type
#SBATCH --partition=long
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=output/experiment-%A.%a.out

module load anaconda/3
conda activate diffcsp

# training perov
python diffcsp/run.py data=perov expname=perov logging.wandb.mode=offline model.decoder.hidden_dim=256 model.decoder.num_layers=4 model.decoder.num_freqs=10 logging.val_check_interval=1 model=diffusion_w_type

# generating eval_diff.pt (for Table 1 in DiffCSP paper)
# python scripts/evaluate.py --model_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-10-25/perov_complete --num_evals 20
# python scripts/compute_metrics.py --root_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-10-25/perov_complete --tasks csp --gt_file data/perov_5/test.csv --multi_eval

# generating eval_gen.pt (for Table 4 in DiffCSP paper)
python scripts/generation.py --model_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-10-31/perov_complete_w_type --dataset perov
python scripts/compute_metrics.py --root_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2023-10-31/perov_complete_w_type --tasks gen --gt_file data/perov_5/test.csv