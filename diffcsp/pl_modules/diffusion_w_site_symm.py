import math, copy
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import defaultdict
from typing import Any, Dict, List

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from pyxtal.symmetry import search_cloest_wp, Group

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc, lattice_ks_to_matrix_torch, get_site_symmetry_binary_repr,
    sg_to_ks_mask, mask_ks, N_SPACEGROUPS)

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
from diffcsp.pl_modules.model import build_mlp

MAX_ATOMIC_NUM=101
CLUSTERED_SITES = json.load(open('/home/mila/s/siba-smarak.panigrahi/DiffCSP/cluster_sites.json', 'r'))

def find_num_atoms(dummy_ind, total_num_atoms):
    # num_atoms states how many atoms are there in each crystal (num_repr + dummy origin)
    actual_num_atoms = []
    atoms = 0
    for num in total_num_atoms:
        # find number of 0 in dummy_ind from atoms to atoms+num
        actual_num_atoms.append(torch.sum(dummy_ind[atoms:atoms+num] == 0).item())
        atoms += num
        
    return torch.tensor(actual_num_atoms)

def split_argmax_sitesymm(site_symm:torch.Tensor) -> np.ndarray:
    # site_symm : num_repr x 27
    site_symm = torch.sigmoid(site_symm).reshape(-1, 3, 9)
    # Initialize the result tensor with zeros
    result = torch.zeros_like(site_symm, dtype=torch.int64)

    # Compute argmax indices for each part
    argmax1 = site_symm[..., :2].argmax(dim=-1)
    argmax2 = site_symm[..., 2:7].argmax(dim=-1) + 2  # offset by 2
    argmax3 = site_symm[..., 7:].argmax(dim=-1) + 7  # offset by 7

    # Expanding dimensions to use for advanced indexing
    batch_range = torch.arange(site_symm.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1, site_symm.size(1), -1)
    row_range = torch.arange(site_symm.size(1)).unsqueeze(0).unsqueeze(-1).expand(site_symm.size(0), -1, -1)

    # Update the result tensor using advanced indexing
    result[batch_range, row_range, argmax1.unsqueeze(-1)] = 1
    result[batch_range, row_range, argmax2.unsqueeze(-1)] = 1
    result[batch_range, row_range, argmax3.unsqueeze(-1)] = 1

    return result.cpu().detach().numpy().reshape(-1, 27)


def modify_frac_coords_one(frac_coords, site_symm, atom_types, spacegroup):
    spacegroup = spacegroup.item()
    
    
    # perform split-argmax to obtain binary representation
    site_symm_argmax = split_argmax_sitesymm(site_symm)
    
    # mapping from binary representation to hm-notation
    wp_to_binary = dict()
    for wp in Group(spacegroup).Wyckoff_positions:
        wp.get_site_symmetry()
        wp_to_binary[wp] = get_site_symmetry_binary_repr(CLUSTERED_SITES[wp.site_symm], label=wp.get_label()).numpy()
     
     
    # iterate over frac coords and corresponding site-symm
    new_frac_coords, new_atom_types, new_site_symm = [], [], []
    for (sym, frac_coord, atm_type) in zip(site_symm_argmax, frac_coords, atom_types):
        frac_coord = frac_coord.cpu().detach().numpy()
        
        # find all wps whose hm-notation align with sym
        closes = []   
        for wp, binary in wp_to_binary.items():
            if np.sum(np.equal(binary, sym)) == sym.shape[-1]:
                for orbit_index in range(1):
                    close = search_cloest_wp(Group(spacegroup), wp, wp.ops[orbit_index], frac_coord)
                    closes.append((close, wp, orbit_index, np.linalg.norm(np.minimum((close - frac_coord)%1., (frac_coord - close)%1.))))
        try:
            # pick the nearest wp to project
            closest = sorted(closes, key=lambda x: x[-1])[0]
            wyckoff = closest[1]
            repr_index = closest[2]
            
            # use wp operations on frac_coord
            frac_coord = closest[0]
            for index in range(len(wyckoff)): 
                # new_frac_coords.append(wyckoff[index].operate(frac_coord)%1.)
                new_frac_coords.append(wyckoff[(index + repr_index) % len(wyckoff)].operate(frac_coord)%1.)
                new_atom_types.append(atm_type.cpu().detach().numpy())
                new_site_symm.append(sym)
        except:
            print('Weird things happen, and I did not predict correctly')
            new_frac_coords.append(frac_coord)
            new_atom_types.append(atm_type.cpu().detach().numpy())
            new_site_symm.append(sym)
        
    new_frac_coords = np.stack(new_frac_coords)
    new_atom_types = np.stack(new_atom_types)
    new_site_symm = np.stack(new_site_symm)
    return new_frac_coords, len(new_frac_coords), new_atom_types, new_site_symm

def modify_frac_coords(traj:Dict, spacegroups:List[int], num_repr:List[int]) -> Dict:
    device = traj['frac_coords'].device
    total_atoms = 0
    updated_frac_coords = []
    updated_num_atoms = []
    updated_atom_types = []
    updated_site_symm = []
    
    for index in range(len(num_repr)):
        if num_repr[index] > 0:
            # if something is predicted, otherwise it is an empty crystal which we are post-processing
            # this might happen if we predict a crystal with only dummy representative atoms
            new_frac_coords, new_num_atoms, new_atom_types, new_site_sym = modify_frac_coords_one(
                    traj['frac_coords'][total_atoms:total_atoms+num_repr[index]], # num_repr x 3
                    traj['site_symm'][total_atoms:total_atoms+num_repr[index]], # num_repr x 27
                    traj['atom_types'][total_atoms:total_atoms+num_repr[index]], # num_repr x 100
                    spacegroups[index], 
                )
        
            updated_frac_coords.append(new_frac_coords)
            updated_num_atoms.append(new_num_atoms)
            updated_atom_types.append(new_atom_types)
            updated_site_symm.append(new_site_sym)
        
        total_atoms += num_repr[index]
    
    traj['frac_coords'] = torch.cat([torch.from_numpy(x) for x in updated_frac_coords]).to(device)
    traj['atom_types'] = torch.cat([torch.from_numpy(x) for x in updated_atom_types]).to(device)
    traj['num_atoms'] = torch.tensor(updated_num_atoms).to(device)
    traj['site_symm'] = torch.cat([torch.from_numpy(x) for x in updated_site_symm]).to(device)
    
    return traj

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler_config}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # NOTE: set pred_site_symm_type to True to generate site symmetries also (set it to False to behave as DiffCSP)
        # pred_type is set to True to generate atom types
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, pred_type = True, pred_site_symm_type = True, smooth = True, max_atoms=MAX_ATOMIC_NUM)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.spacegroup_embedding = build_mlp(in_dim=73, hidden_dim=128, fc_num_layers=2, out_dim=self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.use_ks = self.hparams.use_ks



    def forward(self, batch):

        batch_size = batch.num_graphs
        dummy_repr_ind = batch.dummy_repr_ind
        
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times) + self.spacegroup_embedding(batch.sg_condition.reshape(-1, 73))

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]


        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)
        if c0.ndim == 2:
            c0_lattice = c0[:, self.scheduler.LATTICE]
            c1_lattice = c1[:, self.scheduler.LATTICE]
            c0_atom = c0[:, self.scheduler.ATOM]
            c1_atom = c1[:, self.scheduler.ATOM]
            c0_site_symm = c0[:, self.scheduler.SITE_SYMM]
            c1_site_symm = c1[:, self.scheduler.SITE_SYMM]
        else:
            c0_lattice = c0_atom = c0_site_symm = c0
            c1_lattice = c1_atom = c1_site_symm = c1

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]
        
        ks = batch.ks
        if self.use_ks:
            lattices = lattice_ks_to_matrix_torch(batch.ks)
            ks_mask, ks_add = sg_to_ks_mask(batch.spacegroup)
        else:
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        frac_coords = batch.frac_coords

        rand_x = torch.randn_like(frac_coords)
        rand_ks = torch.randn_like(ks)
        rand_l = torch.randn_like(lattices)

        if self.use_ks:
            input_ks = c0_lattice[:, None] * ks + c1_lattice[:, None] * rand_ks
            input_ks = mask_ks(input_ks, ks_mask, ks_add)
            input_lattice = lattice_ks_to_matrix_torch(input_ks)
        else:
            input_lattice = c0_lattice[:, None, None] * lattices + c1_lattice[:, None, None] * rand_l
            
            
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
        gt_site_symm_binary = batch.site_symm

        rand_t = torch.randn_like(gt_atom_types_onehot)
        rand_symm = torch.randn_like(gt_site_symm_binary)

        atom_type_probs = (c0_atom.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1_atom.repeat_interleave(batch.num_atoms)[:, None] * rand_t)
        site_symm_probs = (c0_site_symm.repeat_interleave(batch.num_atoms)[:, None] * gt_site_symm_binary + c1_site_symm.repeat_interleave(batch.num_atoms)[:, None] * rand_symm)

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices
            input_ks = ks

        # pass noised site symmetries and behave similar to atom type probs
        lattice_feats = input_ks if self.use_ks else input_lattice
        pred_lattice, pred_x, pred_t, pred_symm = self.decoder(time_emb, atom_type_probs, input_frac_coords, 
                                                    lattice_feats, input_lattice, batch.num_atoms, 
                                                    batch.batch, site_symm_probs=site_symm_probs)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        
        loss_lattice = F.mse_loss(pred_lattice, ks_mask * rand_ks) if self.use_ks else F.mse_loss(pred_lattice, rand_l)

        # loss_coord = F.mse_loss(pred_x * (1 - dummy_repr_ind), tar_x * (1 - dummy_repr_ind))
        loss_coord = torch.mean(batch.x_loss_coeff * F.mse_loss(pred_x, tar_x, reduction='none'))
        
        loss_type = F.mse_loss(pred_t, rand_t)
        
        # loss_symm = F.mse_loss(pred_symm * (1 - dummy_repr_ind), rand_symm * (1 - dummy_repr_ind))
        loss_symm = F.mse_loss(pred_symm, rand_symm)
    

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord + 
            self.hparams.cost_type * loss_type +
            self.hparams.cost_symm * loss_symm
        )

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_type' : loss_type,
            'loss_symm' : loss_symm,
        }

    @torch.no_grad()
    def sample(self, batch, diff_ratio = 1.0, step_lr = 1e-5):


        batch_size = batch.num_graphs

        if self.use_ks:
            ks_mask, ks_add = sg_to_ks_mask(batch.spacegroup)
            k_T = torch.randn([batch_size, 6]).to(self.device)
            k_T = mask_ks(k_T, ks_mask, ks_add)
            l_T = lattice_ks_to_matrix_torch(k_T)
        else:
            l_T = torch.randn([batch_size, 3, 3]).to(self.device)
            k_T = torch.zeros([batch_size, 6]).to(self.device) # not used
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)
        symm_T = torch.randn([batch.num_nodes, 27]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            k_T = batch.ks
            l_T = lattice_ks_to_matrix_torch(k_T) if self.use_ks else lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        traj = {self.beta_scheduler.timesteps : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,
            'site_symm' : symm_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T,
            'ks' : k_T,
            'spacegroup': batch.spacegroup,
        }}

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times) + self.spacegroup_embedding(batch.sg_condition.reshape(-1, 73))
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            if c0.ndim == 2:
                c0_lattice = c0[:, self.scheduler.LATTICE]
                c1_lattice = c1[:, self.scheduler.LATTICE]
                c0_atom = c0[:, self.scheduler.ATOM]
                c1_atom = c1[:, self.scheduler.ATOM]
                c0_site_symm = c0[:, self.scheduler.SITE_SYMM]
                c1_site_symm = c1[:, self.scheduler.SITE_SYMM]
            else:
                c0_lattice = c0_atom = c0_site_symm = c0
                c1_lattice = c1_atom = c1_site_symm = c1

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']
            symm_t = traj[t]['site_symm']
            k_t = traj[t]['ks']


            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T
                k_t = k_T

            # Corrector
            if self.use_ks:
                rand_k = torch.randn_like(k_T) if t > 1 else torch.zeros_like(k_T)
            else:
                rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_symm = torch.randn_like(symm_T) if t > 1 else torch.zeros_like(symm_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            lattice_feats_t = k_t if self.use_ks else l_t
            _, pred_x, _, _ = self.decoder(time_emb, t_t, x_t, 
                                                  lattice_feats_t, l_t, batch.num_atoms, 
                                                  batch.batch, site_symm_probs=symm_t)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t
            k_t_minus_05 = k_t

            t_t_minus_05 = t_t

            symm_t_minus_05 = symm_t


            # Predictor
            if self.use_ks:
                rand_k = torch.randn_like(k_T) if t > 1 else torch.zeros_like(k_T)
            else:
                rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)

            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_symm = torch.randn_like(symm_T) if t > 1 else torch.zeros_like(symm_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   
            lattice_feats_t_minus_05 = k_t_minus_05 if self.use_ks else l_t_minus_05

            pred_l, pred_x, pred_t, pred_symm = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, 
                                                        lattice_feats_t_minus_05, l_t_minus_05, batch.num_atoms, 
                                                        batch.batch, site_symm_probs=symm_t_minus_05)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            if self.use_ks:
                k_t_minus_1 = c0_lattice * (k_t_minus_05 - c1_lattice * pred_l) + sigmas * rand_k if not self.keep_lattice else k_t
                k_t_minus_1 = mask_ks(k_t_minus_1, ks_mask, ks_add)
                l_t_minus_1 = lattice_ks_to_matrix_torch(k_t_minus_1) if not self.keep_lattice else l_t
            else:
                l_t_minus_1 = c0_lattice * (l_t_minus_05 - c1_lattice * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
                k_t_minus_1 = k_t

            t_t_minus_1 = c0_atom * (t_t_minus_05 - c1_atom * pred_t) + sigmas * rand_t

            symm_t_minus_1 = c0_site_symm * (symm_t_minus_05 - c1_site_symm * pred_symm) + sigmas * rand_symm

            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'site_symm' : symm_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1,
                'ks' : k_t_minus_1,
                'spacegroup' : batch.spacegroup,
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
            'site_symm' : torch.stack([traj[i]['site_symm'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_ks': torch.stack([traj[i]['ks'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
            'all_spacegroup': torch.stack([traj[i]['spacegroup'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        }

        
        # drop all dummy elements (atom types = MAX_ATOMIC_NUM)
        # NOTE: add breakpoint() here if you want to check (traj[0]['atom_types'].sum(dim=1) can give you sum of atom types values)
        dummy_ind = (traj[0]['atom_types'].argmax(dim=-1) + 1 == MAX_ATOMIC_NUM).long()
        traj[0]['frac_coords'] = traj[0]['frac_coords'][(1 - dummy_ind).bool()]
        traj[0]['atom_types'] = traj[0]['atom_types'][(1 - dummy_ind).bool()]
        traj[0]['site_symm'] = traj[0]['site_symm'][(1 - dummy_ind).bool()]
        
        # find for each crystal how many non-dummy atoms are there
        traj[0]['num_atoms'] = find_num_atoms(dummy_ind, batch.num_atoms).to(self.device)
        
        # remove lattices and ks for empty crystals corresponding to num_atoms = 0
        empty_crystals = (traj[0]['num_atoms'] == 0).long()
        traj[0]['ks'] = traj[0]['ks'][(1 - empty_crystals).bool()]
        traj[0]['lattices'] = traj[0]['lattices'][(1 - empty_crystals).bool()]
        print(f"Number of empty crystals generated: {empty_crystals.sum().item()}/{batch_size}")
        
        # use predicted site symmetry to create copies of atoms
        # frac coords, atom types and num atoms removed for empty crystals in modify_frac_coords()
        traj[0] = modify_frac_coords(traj[0], batch.spacegroup, traj[0]['num_atoms'])
        
        # sanity checks for size of tensors
        assert traj[0]['frac_coords'].size(0) == traj[0]['atom_types'].size(0) == traj[0]['num_atoms'].sum()
        assert traj[0]['ks'].size(0) == traj[0]['lattices'].size(0) == traj[0]['num_atoms'].size(0)

        return traj[0], traj_stack



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss_symm = output_dict['loss_symm']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'type_loss': loss_type,
            'symm_loss': loss_symm,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss_symm = output_dict['loss_symm']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
            f'{prefix}_symm_loss': loss_symm,
        }

        return log_dict, loss

    