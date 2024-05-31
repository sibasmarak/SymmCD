import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os, random
from torch_geometric.data import Data
import pickle
import numpy as np

import symd

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)
EPS = 1e-4*np.random.randn(3)
POINT = np.array([0.5, 0.5, 0.5]) + EPS


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode, 
                 use_space_group: ValueNode, use_pos_index: ValueNode, number_representatives: ValueNode=0, 
                 use_random_representatives:ValueNode=False, lim=0, **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        self.number_representatives = number_representatives
        self.use_random_representatives = use_random_representatives

        self.preprocess(save_path, preprocess_workers, prop, lim)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop, lim):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop],
            use_space_group=self.use_space_group,
            tol=self.tolerance,
            num_repr=self.number_representatives,
            use_random_repr=self.use_random_representatives,
            lim=lim)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def get_asym_unit_position(self, positions, group):
        in_unit = symd.asymm_constraints(group.asymm_unit)
        mask_asym = [in_unit(*position) for position in positions]
        return np.array(mask_asym)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, ks, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html

        # masking on the basis of identifiers of orbits in a crystal
        identifiers = data_dict['identifier']

        #### Random atom from each orbit mask
        #### --------------------------------------
        # find a single representative for each identifier which can then mask
        if 'mask' not in data_dict:
            mask = np.zeros_like(identifiers)

            # # Process each unique identifier
            for identifier in np.unique(identifiers):
                # Find indices where this identifier occurs
                indices = np.where(identifiers == identifier)[0]
                # Get index closest to random point in center
                min_index = ((frac_coords - POINT)**2).sum(1)[indices].argmin().item()
                mask[indices[min_index]] = 1
                self.cached_data[index]['mask'] = mask
        else:
            mask = data_dict['mask']
       
        #### Asymmetric unit mask
        #### ---------------------
        
        # mask = self.get_asym_unit_position(frac_coords, symd.load_group(data_dict['spacegroup'], dim=3))
        
        # # randomly keep one of the selected (mask value is 1) atoms
        # unique_identifiers = np.unique(identifiers)
        # for identifier in unique_identifiers:
        #     if sum(mask[identifiers == identifier]) != 1:
        #         try:
        #             true_indices = np.where(mask[identifiers == identifier])[0]
        #             new_mask = np.zeros_like(mask[identifiers == identifier], dtype=bool)
        #             new_mask[random.choice(true_indices)] = True
        #             mask[identifiers == identifier] = new_mask
        #         except:
        #             new_mask = np.zeros_like(mask[identifiers == identifier], dtype=bool)
        #             new_mask[random.choice(np.arange(len(mask[identifiers == identifier])))] = True
        #             mask[identifiers == identifier] = new_mask
                    
        # assert sum(mask) == len(unique_identifiers), "Length of asymmetric unit must match with unique identifiers."
        
        frac_coords = frac_coords[mask.astype(bool)]
        atom_types = atom_types[mask.astype(bool)]
        num_atoms = len(frac_coords)

        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            ks=torch.Tensor(ks).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])
            data.number_representatives = torch.LongTensor([num_atoms]) # torch.LongTensor([data_dict['number_representatives']])

            data.sg_condition = torch.Tensor(data_dict['sg_binary'])
            data.site_symm = torch.Tensor(data_dict['site_symm_binary'].float())[mask.astype(bool)]
            
            data.dummy_repr_ind = torch.Tensor([data_dict['dummy_repr_ind']]).reshape(-1, 1)
            
            # compute position loss coefficient (basically, the multiplicity of each orbit)
            identifiers_torch = torch.tensor(data_dict['identifier'])
            changes = torch.where(torch.diff(identifiers_torch) != 0)[0] + 1
            changes = torch.cat((torch.tensor([0]), changes, torch.tensor([len(identifiers_torch)])))
            consecutive_counts_torch = torch.diff(changes)
            data.x_loss_coeff = consecutive_counts_torch.reshape(-1, 1)
            
            # data.wyckoff_labels = torch.LongTensor(wyckoff_labels_to_category(data_dict['labels']))[mask.astype(bool)]

        assert len(data.site_symm) == len(data.frac_coords) == len(data.atom_types) # == len(data.x_loss_coeff), breakpoint()
        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, ks, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            ks=torch.Tensor(ks).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from diffcsp.common.data_utils import get_scaler_from_data_list
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
