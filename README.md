# SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models

This is the codebase for the the paper [SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models](https://arxiv.org/abs/2502.03638) presented at ICLR 2025.

![alt text](figure_symmcd.pdf "Title")

## Setup

#### Recommended installation method
It is recommended to install each necessary package (with appropriate versions mentioned in the `symmcd.yaml`) file following this order: `pytorch`, `pytorch-lightning`, `pyg`, `pyxtal`, `pymatgen`, `matminer`, `einops`, `hydra-core`, `symd`, `dotenv`, `wandb`,`p_tqdm`,`torch_scatter`, `torch_sparse`, `smact`,`chemparse`.  

#### Other installation method

1. Rename the `.env.template` file into `.env` and specify the following variables.
```
PROJECT_ROOT: the absolute path of this repo
HYDRA_JOBS: the absolute path to save hydra outputs
WABDB_DIR: the absolute path to save wandb outputs
WABDB_DIR: the absolute path to save wandb cache outputs
```
2. Create the anaconda environment with the `symmcd.yml` file (`conda env create -f symmcd.yml`: creates environment with name `symmcd`)
3. Install `matminer` with `cd matminer && pip install -e .`   
4. Clone [`cdvae repo`](https://github.com/txie-93/cdvae) in the same directory level as `conf` and `data` and install it    
5. You can now execute the training command [below](README.md#Training) without errors. If it throws some package error, please install those (and create a PR).  

```
SymmCD (should now look like this, and the environment should contain cdvae and matminer packages)
├── cdvae
├── conf
├── data 
├── symmcd
├── matminer
├── scripts
├── .env
├── .........
```

## Training
Before training, you should change the following in `cdvae/cdvae/pl_modules/gnn.py` (`swish` can no longer be imported from `torch_geometric.nn.acts`):  
- Comment out the import: `from torch_geometric.nn.acts import swish`
- Add the following lines to the script before the `class InteractionPPBlock`:  
```
def swish(x):
    return x * x.sigmoid()
```

### For the Ab Initio Generation task

```
python symmcd/run.py data=<dataset> model=discrete_diffusion_w_site_symm expname=<expname>
```

- The ``<dataset>`` tag can be selected from `mp_20` and `mpts_52`.   
- For multiple GPUs, please add `train.pl_trainer.devices=2` to above commands (ensure 2 gpus on machine where script launches).

## Evaluation

### Ab initio generation

```
python scripts/generation.py --model_path <model_path> --dataset <dataset>
python scripts/compute_metrics --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv
```


### Sample from arbitrary composition

```
python scripts/sample.py --model_path <model_path> --save_path <save_path> --formula <formula> --num_evals <num_evals>
```


### How to run the sweep

- Change/Add hyperparameters and their values in the `hyperparam_sweep.yaml` file.  
- `wandb sweep --project <project-name> -e <entity-name> hyperparam_sweep.yaml`.  
- `wandb agent <above-agent-id>`.  

## Citation
If you find this codebase useful, please cite the following paper:
```
@inproceedings{levysymmcd,
  title={SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models},
  author={Levy, Daniel and Panigrahi, Siba Smarak and Kaba, S{\'e}kou-Oumar and Zhu, Qiang and Lee, Kin Long Kelvin and Galkin, Mikhail and Miret, Santiago and Ravanbakhsh, Siamak},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

## Acknowledgements
This codebase was adapted from the [DiffCSP](https://github.com/jiaor17/DiffCSP) repository. We thank the authors for their work and open-sourcing their code.
