import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode, 
                 use_space_group: ValueNode, use_pos_index: ValueNode, number_representatives: ValueNode, 
                 use_random_representatives:ValueNode, **kwargs):
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

        self.preprocess(save_path, preprocess_workers, prop)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop):
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
            use_random_repr=self.use_random_representatives)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, ks, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # # processing to add dummy representatives after graph creation (ineffective and not necessary)
        # current_representatives = len(frac_coords) - 1
        # dummy_atom_type = atom_types[-1]
        # site_symm_binary = data_dict['site_symm_binary']
        # trivial_site_symm = np.zeros(22)
        # trivial_site_symm[0] = 1 # inversion
        # trivial_site_symm[2] = 1 # rotation
        # trivial_site_symm[7] = 1 # mirror plane
        # trivial_site_symm[9] = 1 # screw translation
        # trivial_site_symm[15] = 1 # glide plane
        # trivial_site_symm = np.hstack([trivial_site_symm, trivial_site_symm, trivial_site_symm])

        # if current_representatives < self.number_representatives:
        #     print("Adding dummy representatives after crystal graph creation")
        #     # insert dummy representatives (if it was not added during preprocessing dataset in data_utils.py)
        #     new_node_id = num_atoms
        #     for _ in range(self.number_representatives - current_representatives):
        #         if self.use_random_representatives:
        #             # sample a random coordinate from a uniform distribution
        #             frac_coords = np.vstack([frac_coords, np.random.uniform(size=3)]) # doesn't matter
        #             atom_types = np.hstack([atom_types, dummy_atom_type])
        #             site_symm_binary = np.vstack([site_symm_binary, trivial_site_symm]) # doesn't matter
        #             num_atoms += 1
                    
        #             # connect this node to all other nodes
        #             # add edges in [new_node_id, i] format
        #             new_edges_for_new_node = np.vstack([np.arange(num_atoms-1), np.ones(num_atoms-1)*new_node_id]).T
        #             # add to_jimages in [0, 0, 0] format
        #             to_jimages = np.vstack([to_jimages, np.zeros((num_atoms-1, 3))])
        #             edge_indices = np.vstack([edge_indices, new_edges_for_new_node])
                    
        #         else:
        #             random_index = np.random.randint(0, current_representatives)
        #             frac_coords = np.vstack([frac_coords, frac_coords[random_index]])
        #             atom_types = np.hstack([atom_types, dummy_atom_type])
        #             # add to data dict site symmetry
        #             site_symm_binary = np.vstack([site_symm_binary, data_dict['site_symm_binary'][random_index]])
        #             num_atoms += 1
                    
        #             # add the corresponding edge_indices
        #             edges_connected = edge_indices[np.any(edge_indices == random_index, axis=1)]
        #             new_edges_for_new_node = np.where(edges_connected == random_index, new_node_id, edges_connected)
                    
        #             # add jimages corresponding to the new edges
        #             to_jimages = np.vstack([to_jimages, to_jimages[np.any(edge_indices == random_index, axis=1)]])
        #             edge_indices = np.vstack([edge_indices, new_edges_for_new_node])

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
            y=prop.view(1, -1),
        )

        if self.use_space_group:
            data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
            data.ops = torch.Tensor(data_dict['wyckoff_ops'])
            data.anchor_index = torch.LongTensor(data_dict['anchors'])
            data.number_repsentatives = torch.LongTensor([data_dict['number_representatives']])

            data.sg_condition = torch.Tensor(data_dict['sg_binary'])
            data.site_symm = torch.Tensor(data_dict['site_symm_binary'])
            
            data.dummy_origin_ind = torch.Tensor([data_dict['dummy_origin_ind']])
            data.dummy_repr_ind = torch.Tensor([data_dict['dummy_repr_ind']])

        assert len(data.site_symm) == len(data.frac_coords) == len(data.site_symm), breakpoint()
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
