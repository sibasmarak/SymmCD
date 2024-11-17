import time
import argparse
import torch
import csv
import os
from collections import Counter, defaultdict
from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map

EPS = 1e-4*np.random.randn(3)
POINT = np.array([0.5, 0.5, 0.5]) + EPS

import sys
sys.path.append('.')
from scripts.eval_utils import load_model, lattices_to_params_shape, get_crystals_list


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

def diffusion(loader, model, step_lr):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    spacegroups = []
    site_symmetries = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr = step_lr)
        del traj
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())
        spacegroups.append(outputs['spacegroup'].detach().cpu())
        site_symmetries.append(outputs['site_symm'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    spacegroups = torch.cat(spacegroups, dim=0)
    site_symmetries = torch.cat(site_symmetries, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, spacegroups, site_symmetries
    )

class SampleDataset(Dataset):

    def __init__(self, dataset, total_num, train_ori_path=None, sg_info_path=None, restrict_spacegroups=None,):
        # pass the training data in the `train_ori_path` 
        # to get the distribution of space groups and atom numbers in the training dataset
        # restrict_spacegroups: np array of spacegroups to sample from (using renormalized full probabilities)
        
        super().__init__()
        self.total_num = total_num
        self.distribution = train_dist[dataset]
        self.num_atoms = np.random.choice(len(self.distribution), total_num, p = self.distribution)
        self.is_carbon = dataset == 'carbon'

        self.sg_num_atoms, self.sg_dist, self.sg_number_binary_mapper = self.get_sg_statistics(train_ori_path, sg_info_path)

        if restrict_spacegroups is not None:
            print("Sampling ONLY from spacegroups " + str(restrict_spacegroups))
            new_sg_dist = np.zeros_like(self.sg_dist)
            new_sg_dist[restrict_spacegroups - 1] = self.sg_dist[restrict_spacegroups - 1]
            new_sg_dist = new_sg_dist / new_sg_dist.sum()
            self.sg_dist = new_sg_dist


    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        spacegroup = np.random.choice(230, p = self.sg_dist) + 1
        num_atom = np.random.choice(list(self.sg_num_atoms[spacegroup].keys()), p = list(self.sg_num_atoms[spacegroup].values()))
        
        data = Data(
            num_atoms=torch.LongTensor([num_atom]),
            num_nodes=num_atom,
            spacegroup=spacegroup, # number
            sg_condition=self.sg_number_binary_mapper[spacegroup], # float tensor
        )
        if self.is_carbon:
            data.atom_types = torch.LongTensor([6] * num_atom)
        return data

    def get_sg_statistics(self, train_path=None, sg_info_path=None):
        '''
        Get per-spacegroup
        '''
        if sg_info_path and os.path.exists(sg_info_path):
            print(f'Loading spacegroup statistics from {sg_info_path}')
            return torch.load(sg_info_path)
        dataset = torch.load(train_path)
        dataset_len = len(dataset)
        
        sg_counter = defaultdict(lambda : 0)
        sg_num_atoms = defaultdict(lambda : defaultdict(lambda : 0))
        sg_number_binary_mapper = {}
        for i in range(dataset_len):
            data_dict = dataset[i]
            (frac_coords, atom_types, lengths, angles, ks, edge_indices,
            to_jimages, num_atoms) = data_dict['graph_arrays']
            spacegroup = data_dict['spacegroup']
            # masking on the basis of identifiers of orbits in a crystal
            identifiers = data_dict['identifier']
            
            mask = np.zeros_like(identifiers)

            # Process each unique identifier
            for identifier in np.unique(identifiers):
                # Find indices where this identifier occurs
                indices = np.where(identifiers == identifier)[0]
                # Get index closest to random point in center
                min_index = ((frac_coords - POINT)**2).sum(1)[indices].argmin().item()
                mask[indices[min_index]] = 1

                
            frac_coords = frac_coords[mask.astype(bool)]
            atom_types = atom_types[mask.astype(bool)]
            num_atoms = len(frac_coords)

            
            sg_counter[spacegroup] += 1
            sg_number_binary_mapper[spacegroup] = data_dict['sg_binary']
            sg_num_atoms[spacegroup][num_atoms] += 1
            
            
        # spacegroup distribution
        sg_dist = []
        for i in range(1, 231): sg_dist.append(sg_counter[i])
        sg_dist = np.array(sg_dist)
        sg_dist = sg_dist / dataset_len
        sg_number_binary_mapper = sg_number_binary_mapper
        
        # for each space group, atom number distribution
        for sg in sg_num_atoms:
            total = sum(sg_num_atoms[sg].values())
            for num_atoms in sg_num_atoms[sg]: sg_num_atoms[sg][num_atoms] /= total
        if sg_info_path:
            sg_num_atoms_hashable = {k: {kk: vv for kk, vv in v.items()} for k, v in sg_num_atoms.items()}
            torch.save((sg_num_atoms_hashable, sg_dist, sg_number_binary_mapper), sg_info_path)
        return  sg_num_atoms, sg_dist, sg_number_binary_mapper

def save_cif(model_path, crys_array_list, label):
    if label == '':
        cif_path = model_path / 'crystal.csv'
    else:
        cif_path = model_path / f'crystal_{label}.csv'
    with open(cif_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        for i in tqdm(range(len(crys_array_list))):
            crys_dict = crys_array_list[i]
            if len(crys_dict['atom_types'].shape) == 2:
                atom_types = crys_dict['atom_types'].argmax(-1) + 1
            else:
                atom_types = crys_dict['atom_types']
            crystal = Structure(
                lattice=Lattice.from_parameters(
                    *(crys_dict['lengths'].tolist() + crys_dict['angles'].tolist())),
                species=atom_types,
                coords=crys_dict['frac_coords'],
                coords_are_cartesian=False)
        csvwriter.writerow([crystal.to(fmt='cif')])

def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path, load_data=False)

    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')
    restrict_spacegroups = np.array(args.restrict_spacegroups) if args.restrict_spacegroups is not None else None
    test_set = SampleDataset(args.dataset, 
                             args.batch_size * args.num_batches_to_samples, 
                             train_ori_path=cfg.data.datamodule.datasets.train.save_path,
                             sg_info_path=cfg.data.datamodule.datasets.train.sg_info_path,
                             restrict_spacegroups=restrict_spacegroups)
    test_loader = DataLoader(test_set, batch_size = args.batch_size)

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, spacegroups, site_symmetries) = diffusion(test_loader, model, args.step_lr)

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
        'site_symmetries': site_symmetries,
    }, model_path / gen_out_name)

    if args.save_cif is not None:
        crys_array_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms, spacegroups=spacegroups, site_symmetries=site_symmetries)
        save_cif(model_path, crys_array_list, args.label)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float, help='step size for Langevin dynamics')
    parser.add_argument('--num_batches_to_samples', default=10, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--restrict_spacegroups', nargs='+', type=int, help='list of spacegroups to sample from')
    parser.add_argument('--save_cif', help='option to save cif files', default=None)

    args = parser.parse_args()


    main(args)
