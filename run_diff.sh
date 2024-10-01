#!/bin/bash
#SBATCH --job-name=mp20
#SBATCH --partition=long
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=0
#SBATCH --output=mp_20_eval_output/mp20-experiment-%A.%a.out

module load anaconda/3
conda activate diffcsp39

# training
HYDRA_FULL_ERROR=1 python diffcsp/run.py expname=mp_20 data=mp_20 model=discrete_diffusion_w_site_symm \
data.number_representatives=0 data.train_max_epochs=100000 \
logging.wandb.mode=offline logging.wandb.project=diffcsp_symmetry logging.val_check_interval=1 optim.optimizer.lr=0.001 \
data.use_space_group=True data.use_asym_unit=False data.eval_every_epoch=100 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 model.decoder.use_site_symm=False model.decoder.use_site_symm=False

# generating eval_gen.pt (for Table 4 in DiffCSP paper, ab-initio generation) 
# python -m scripts.generation --model_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2024-03-17/mp_20_gnn_gt_frac_coords/ --dataset mp --label final_num_samples_20 --batch_size 5 --num_batches_to_samples 2
# python -m scripts.compute_metrics --root_path /home/mila/s/siba-smarak.panigrahi/scratch/DiffCSP/hydra/singlerun/2024-03-17/mp_20_gnn_gt_frac_coords/ --tasks gen --gt_file data/mp_20/test.csv --label final_num_samples_20

# generating eval_diff.pt (for Table 1 in DiffCSP paper, crystal structure prediction (csp) task)
# python -m scripts.evaluate --model_path <path-of-run> --label num_samples_1
# python -m scripts.compute_metrics --root_path <path-of-run> --tasks csp --gt_file data/perov_5/test.csv --label num_samples_1

# python -m scripts.evaluate --model_path <path-of-run> --num_evals 20 --label num_samples_20
# python -m scripts.compute_metrics --root_path <path-of-run> --tasks csp --gt_file data/perov_5/test.csv --multi_eval --label num_samples_20 