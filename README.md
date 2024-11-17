# SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models

## Setup

#### Recommended installation method
It is recommended to install each necessary package (with appropriate versions mentioned in the `diffcsp39.yaml`) file following this order:  
`pytorch`, `pytorch-lightning`, `pyg`, `pyxtal`, `pymatgen`, `matminer`, `einops`, `hydra-core`, `symd`, `dotenv`, `wandb`,`p_tqdm`,`torch_scatter`, `torch_sparse`, `smact`,`chemparse`.  

#### Other installation method

1. follow the instructions from [DiffCSP's .env setup](https://github.com/jiaor17/DiffCSP?tab=readme-ov-file#dependencies-and-setup) to setup/add `.env` file  
2. create the anaconda environment with the `symmcd.yml` file (`conda env create -f symmcd.yml`: creates environment with name `symmcd`)
3. install `matminer` with `cd matminer && pip install -e .`   
4. clone [`cdvae repo`](https://github.com/txie-93/cdvae) in the same directory level as `conf` and `data` and install it    
5. if you execute training command [below](https://github.com/sibasmarak/SymmCD/#Training), it should run without errors. If it throws some package error, please install those (and create a PR).  

```
SymmCD (should now look like this, and the environment should contain cdvae and matminer packages)
├── cdvae
├── conf
├── data 
├── diffcsp
├── matminer
├── scripts
├── .env
├── .........
```

### Training

For the Ab Initio Generation task

```
python diffcsp/run.py data=<dataset> model=diffusion_w_type expname=<expname>
```

The ``<dataset>`` tag can be selected from `mp_20` and `mpts_52`.  

For multiple GPUs, please add `train.pl_trainer.devices=2` to above commands (ensure 2 gpus on machine where script launches).

### Evaluation

#### Ab initio generation

```
python scripts/generation.py --model_path <model_path> --dataset <dataset>
python scripts/compute_metrics --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv
```


#### Sample from arbitrary composition

```
python scripts/sample.py --model_path <model_path> --save_path <save_path> --formula <formula> --num_evals <num_evals>
```


#### How to run the sweep

- Change/Add hyperparameters and their values in the `hyperparam_sweep.yaml` file.  
- `wandb sweep --project <project-name> -e <entity-name> hyperparam_sweep.yaml`.  
- `wandb agent <above-agent-id>`.  
