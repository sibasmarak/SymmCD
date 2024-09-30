#!/bin/bash
#SBATCH --job-name=mp20
#SBATCH --partition=long
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=mp_20_eval_output/mp20-experiment-%A.%a.out

module load anaconda/3
conda activate diffcsp39

site_info_dims=(2 4 8 16 32)
HYDRA_FULL_ERROR=1 python diffcsp/run.py expname=mp_20_prop_trans data=mp_20 \
data.use_space_group=True data.use_asym_unit=True data.eval_every_epoch=1 data.train_max_epochs=500 data.early_stopping_patience=100 \
logging.wandb.mode=online logging.wandb.project=crystgnn_supervise logging.val_check_interval=1 \
model=cspnet_transformer model.encoder.use_site_info=True model.encoder.site_info_dim=${site_info_dims[$SLURM_ARRAY_TASK_ID]} \
train.monitor_metric=val_mae


