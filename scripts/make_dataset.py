import sys
sys.path.append('.')

import warnings
warnings.filterwarnings("ignore")

niggli = True
primitive = False
prop = 'formation_energy_per_atom'
graph_method='crystalnn'
lattice_scale_method = 'scale_length'
preprocess_workers = 10
readout = 'mean'
max_atoms = 20
otf_graph = False
eval_model_name=  "mp_20"
tolerance = 0.01
use_space_group = True
use_pos_index = False

from diffcsp.pl_data.dataset import CrystDataset
path = './data/mp_20/train.csv'
save_path ='./data/mp_20/train_ori.pt'
dataset = CrystDataset(name="Formation energy test",
                       path=path,
                       save_path=save_path,
                       prop=prop,
                       niggli=niggli,
                       primitive=primitive,
                       graph_method=graph_method,
                       preprocess_workers=preprocess_workers,
                       lattice_scale_method=lattice_scale_method,
                       tolerance=tolerance,
                       use_space_group=use_space_group,
                       use_pos_index=use_pos_index,
                       number_representatives=0,
                       use_random_representatives=False)

path = './data/mp_20/val.csv'
save_path ='./data/mp_20/val_ori.pt'
dataset = CrystDataset(name="Formation energy test",
                       path=path,
                       save_path=save_path,
                       prop=prop,
                       niggli=niggli,
                       primitive=primitive,
                       graph_method=graph_method,
                       preprocess_workers=preprocess_workers,
                       lattice_scale_method=lattice_scale_method,
                       tolerance=tolerance,
                       use_space_group=use_space_group,
                       use_pos_index=use_pos_index,
                       number_representatives=0,
                       use_random_representatives=False)


path = './data/mp_20/test.csv'
save_path ='./data/mp_20/test_ori.pt'
dataset = CrystDataset(name="Formation energy test",
                       path=path,
                       save_path=save_path,
                       prop=prop,
                       niggli=niggli,
                       primitive=primitive,
                       graph_method=graph_method,
                       preprocess_workers=preprocess_workers,
                       lattice_scale_method=lattice_scale_method,
                       tolerance=tolerance,
                       use_space_group=use_space_group,
                       use_pos_index=use_pos_index,
                       number_representatives=0,
                       use_random_representatives=False)
