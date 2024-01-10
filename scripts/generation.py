"""
NOTE: 
this script should be used for diffusion with atom types or "ab-initio generation task"
please do not use this for model trained on diffusion without atom types

for diffusion without atom types, use evaluate.py only 
for "CSP task", we care about MR and RMSE, see Table 1 of https://arxiv.org/pdf/2309.04475.pdf
"""

import time
import argparse
import torch

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list
from diffcsp.common.constants import SpaceGroupDist

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

import pdb

import os

def get_num_atoms_per_sg(dataset):
    if dataset == 'perov':
        dataset_name = 'perov_4'
    elif dataset == 'carbon':
        dataset_name = 'carbon_24'
    elif dataset == 'mp':
        dataset_name = 'mp_20'
    else:
        raise NotImplementedError
    dist_file = f'./data/{dataset_name}/num_atoms_per_sg.csv'
    return np.loadtxt(dist_file, delimiter=',')
    

train_dist = {
    'perov' : [0, 0, 0, 0, 0, 1],
    'carbon' : [0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3250697750779839,
                0.0,
                0.27795107535708424,
                0.0,
                0.15383352487276308,
                0.0,
                0.11246100804465604,
                0.0,
                0.04958134953209654,
                0.0,
                0.038745690362830404,
                0.0,
                0.019044491873255624,
                0.0,
                0.010178952552946971,
                0.0,
                0.007059596125430964,
                0.0,
                0.006074536200952225],
    'mp' : [0.0,
            0.0021742334905660377,
            0.021079009433962265,
            0.019826061320754717,
            0.15271226415094338,
            0.047132959905660375,
            0.08464770047169812,
            0.021079009433962265,
            0.07808814858490566,
            0.03434551886792453,
            0.0972877358490566,
            0.013303360849056603,
            0.09669811320754718,
            0.02155807783018868,
            0.06522700471698113,
            0.014372051886792452,
            0.06703272405660378,
            0.00972877358490566,
            0.053176591981132074,
            0.010576356132075472,
            0.08995430424528301]
}

def diffusion(loader, model, step_lr, use_sg=False):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    spacegroups = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr = step_lr)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())
        if use_sg:
            spacegroups.append(outputs['spacegroup'].detach().cpu())


    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)
    if len(spacegroups) > 0:
        spacegroups = torch.cat(spacegroups, dim=0)
    else:
        spacegroups = None

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, spacegroups
    )

class SampleDataset(Dataset):

    def __init__(self, dataset, total_num, use_num_atoms_per_sg = False):
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.sg_distribution = SpaceGroupDist[dataset]
        if use_num_atoms_per_sg:
            num_atoms_per_sg = get_num_atoms_per_sg(dataset)
            num_atoms_per_sg = num_atoms_per_sg / num_atoms_per_sg.sum()
            choices = np.random.choice(num_atoms_per_sg.shape[0]*num_atoms_per_sg.shape[1], total_num, p = num_atoms_per_sg.reshape(-1))
            self.num_atoms = choices % num_atoms_per_sg.shape[1]
            self.sg = choices // num_atoms_per_sg.shape[1]
        else:
            self.sg = np.random.choice(len(self.sg_distribution), total_num, p = self.sg_distribution)
            self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.is_carbon = dataset == 'carbon'

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):

        num_atom = self.num_atoms[index]
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
            spacegroup=torch.LongTensor([self.sg[index]]),
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path, load_data=False)
    use_sg = cfg.model.use_spacegroup

    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')

    test_set = SampleDataset(args.dataset, args.batch_size * args.num_batches_to_samples, args.num_atoms_per_sg)
    test_loader = DataLoader(test_set, batch_size = args.batch_size)

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, spacegroups) = diffusion(test_loader, model, args.step_lr, use_sg)

    if args.label == '':
        gen_out_name = 'eval_gen.pt'
    else:
        gen_out_name = f'eval_gen_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'spacegroups': spacegroups,
    }, model_path / gen_out_name)
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--num_atoms_per_sg', default=False, type=bool)
    args = parser.parse_args()


    main(args)
