# DiffCSP

## Setup
1. follow the instructions from [DiffCSP's .env setup](https://github.com/jiaor17/DiffCSP?tab=readme-ov-file#dependencies-and-setup) to setup/add `.env` file  
2. create the anaconda environment with the `environment.yml` file (`conda env create -f environment.yml`: creates environment with name `diffcsp`)
3. install `matminer` with `cd matminer && pip install -e .`   
4. clone [`cdvae repo`](https://github.com/txie-93/cdvae) in the same directory level as `conf` and `data` and install it    
5. if you execute `run_diff.sh`, it should work now  
```
DiffCSP (should now look like this, and the environment should contain cdvae and matminer packages)
├── cdvae
├── conf
├── data 
├── diffcsp
├── matminer
├── scripts
├── .env
├── .........
```

Implementation codes for Crystal Structure Prediction by Joint Equivariant Diffusion.

### Dependencies

```
python==3.8.13
torch==1.9.0
torch-geometric==1.7.2
pytorch_lightning==1.3.8
pymatgen==2022.9.21 
pyxtal==0.6.0
<!-- pymatgen==2020.12.31 (old COMP760) -->
```

### Training

For the CSP task

```
python diffcsp/run.py data=<dataset> expname=<expname>
```

For the Ab Initio Generation task

```
python diffcsp/run.py data=<dataset> model=diffusion_w_type expname=<expname>
```

The ``<dataset>`` tag can be selected from perov_5, mp_20, mpts_52 and carbon_24.

### Evaluation

#### Stable structure prediction 

One sample 

```
python scripts/evaluate.py --model_path <model_path>
python scripts/compute_metrics --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv 
```

Multiple samples

```
python scripts/evaluate.py --model_path <model_path> --num_evals 20
python scripts/compute_metrics --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv --multi_eval
```

#### Ab initio generation

```
python scripts/generation.py --model_path <model_path> --dataset <dataset>
python scripts/compute_metrics --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv
```


#### Sample from arbitrary composition

```
python scripts/sample.py --model_path <model_path> --save_path <save_path> --formula <formula> --num_evals <num_evals>
```
