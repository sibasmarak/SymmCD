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
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
from tqdm import tqdm

from pyxtal.symmetry import search_cloest_wp, Group

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch, wyckoff_category_to_labels,
    frac_to_cart_coords, min_distance_sqr_pbc, lattice_ks_to_matrix_torch, get_site_symmetry_binary_repr,
    sg_to_ks_mask, mask_ks, N_SPACEGROUPS)

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
from diffcsp.pl_modules.model import build_mlp

MAX_ATOMIC_NUM=100
NUM_WYCKOFF = 186

CLUSTERED_SITES = json.load(open('/home/mila/d/daniel.levy/scratch/MatSci/intel-mat-diffusion/cluster_sites.json', 'r'))

class DiscreteNoise(nn.Module):
    def __init__(self, atom_type_marginals, wyckoff_marginals_per_sg, beta_scheduler):
        super().__init__()
        self.beta_scheduler = beta_scheduler
        #self.site_symm_marginals = site_symm_marginals
        #self.ss_marginals_list = self.ss_to_sections(site_symm_marginals)
        self.ss_marginals_per_sg = wyckoff_marginals_per_sg
        
        self.atom_type_marginals = atom_type_marginals
        self.P_ss = nn.Parameter(wyckoff_marginals_per_sg.unsqueeze(1).expand(-1, NUM_WYCKOFF, -1).clone(), requires_grad=False)
        self.P_a = nn.Parameter(atom_type_marginals.unsqueeze(0).expand(MAX_ATOMIC_NUM, -1).clone(), requires_grad=False)

    #def ss_to_sections(self, ss):
    #    ss_list = [ss[..., self.ss_section_idx[i]: self.ss_section_idx[i+1]] for i in range(len(self.ss_section_idx)-1)]
    #    return ss_list
    
    def multiply_block_diagonal(self, Qs, d):
        '''
        Multiply each matrix Qi in Qs with the corresponding block of d
        Qs: list of D matrices Qi, each Qi is of shape (ni, ni)
        d:  vector of length sum(ni)
        returns: vector of length sum(ni)
        '''
        outs = [] 
        idx = 0
        for Qi in Qs:
            ni = Qi.shape[-1]
            outs.append(d[..., idx:idx+ni] @ Qi)
            idx += ni
        return torch.cat(outs, -1)

    def q_t(self, P, t):
        alpha = self.beta_scheduler.alphas[t]
        num_classes = P.shape[-1]
        return alpha.view(-1,1,1)*torch.eye(num_classes, device=P.device)+(1-alpha.view(-1,1,1))*P
    
    def q_t_atom(self, t):
        return self.q_t(self.P_a, t)

    def q_t_ss(self, t, sgs):
        return self.q_t(self.P_ss[sgs], t)

    def q_t_bar(self, P, t):
        alpha_bar = self.beta_scheduler.alphas_cumprod[t]
        num_classes = P.shape[-1]
        return alpha_bar.view(-1,1,1)*torch.eye(num_classes, device=P.device)+ (1 - alpha_bar.view(-1,1,1))*P
    
    def q_t_bar_atom(self, t):
        return self.q_t_bar(self.P_a, t)

    def q_t_bar_ss(self, t, sgs):
        return self.q_t_bar(self.P_ss[sgs], t)

    def sigma_sqr_ratio(self, s_int, t_int):
        return self.beta_scheduler.alphas_cumprod[t_int] / self.beta_scheduler.alphas_cumprod[s_int]
    
    def apply_atom_noise(self, atom_type, t):
        Q_t_bar = self.q_t_bar_atom(t)
        prob_atom_types = atom_type @  Q_t_bar
        return prob_atom_types
    
    def apply_site_symm_noise(self, site_symm, t, sgs):
        Q_t_bar = self.q_t_bar_ss(t, sgs)
        prob_site_symms = site_symm @  Q_t_bar
        return prob_site_symms

    def sample_atom_types(self, atom_probs):
        return F.one_hot(torch.multinomial(atom_probs, 1).reshape(-1), MAX_ATOMIC_NUM).float()
    
    def sample_site_symms(self, site_symm_probs):
        return F.one_hot(torch.multinomial(site_symm_probs, 1).reshape(-1), MAX_ATOMIC_NUM).float()
    
    def sample_limit_dist(self, node_mask, sgs):
        """ Sample from the limit distribution of the diffusion process"""
        bs, n_max = node_mask.shape
        a_limit = self.atom_type_marginals.expand(bs, n_max, -1)
        U_a = a_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_a = F.one_hot(U_a, num_classes=a_limit.shape[-1]).float()
        U_a = U_a * node_mask.unsqueeze(-1)

        ss_limit = self.ss_marginals_per_sg[sgs].unsqueeze(1).expand(-1, n_max, -1)
        U_ss = ss_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_ss = F.one_hot(U_ss, num_classes=ss_limit.shape[-1]).float()
        U_ss = U_ss * node_mask.unsqueeze(-1)

        return U_a, U_ss

    def sample_discrete_features(self, prob_a, prob_ss, node_mask):
        ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
            :param prob_a: bs, n, dx_out        node features
            :param prob_ss: bs, n, dx_out        node features
        '''
        bs, n = node_mask.shape
        # The masked rows should define probability distributions as well
        prob_a[~node_mask] = 1 / prob_a.shape[-1]
        prob_ss[~node_mask] = 1 / prob_ss.shape[-1]

        # Flatten the probability tensor to sample with multinomial
        prob_a = prob_a.reshape(bs * n, -1)       # (bs * n, dx_out)
        # Sample a
        atom_t = prob_a.multinomial(1)                                  # (bs * n, 1)
        atom_t = atom_t.reshape(bs, n)     # (bs, n)
        atom_t = F.one_hot(atom_t, num_classes=prob_a.shape[-1]).float()
        # Sample ss
        prob_ss = prob_ss.reshape(bs * n, -1)       # (bs * n, dx_out)
        # Sample a
        site_symm_t = prob_ss.multinomial(1)                                  # (bs * n, 1)
        site_symm_t = site_symm_t.reshape(bs, n)     # (bs, n)
        site_symm_t = F.one_hot(site_symm_t, num_classes=prob_ss.shape[-1]).float()

        return atom_t, site_symm_t


    def p_s_and_t_given_0(self, z_t, Qt, Qsb, Qtb):
        """ M: X, E or charges
            Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
            X_t: bs, n, dt
            Qt: bs, d_t-1, dt
            Qsb: bs, d0, d_t-1
            Qtb: bs, d0, dt.
        """
        # TODO
        Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
        left_term = z_t @ Qt_T                      # bs, N, d_t-1
        left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

        right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
        numerator = left_term * right_term          # bs, N, d0, d_t-1

        X_t_transposed = z_t.transpose(-1, -2)      # bs, dt, N

        prod = Qtb @ X_t_transposed                 # bs, d0, N
        prod = prod.transpose(-1, -2)               # bs, N, d0
        denominator = prod.unsqueeze(-1)            # bs, N, d0, 
        denominator[denominator == 0] = 1e-6

        out = numerator / denominator
        return out

    def p_s_and_t_given_0_a(self, z_t_a, t, s):
        Qtb_a = self.q_t_bar_atom(t)
        Qsb_a = self.q_t_bar_atom(s)
        Qt_a = self.q_t_atom(t)
        return self.p_s_and_t_given_0(z_t_a,Qt_a, Qsb_a, Qtb_a)

    def p_s_and_t_given_0_ss(self, z_t_ss, t, s, sgs):
        Qtb_ss = self.q_t_bar_ss(t, sgs)
        Qsb_ss = self.q_t_bar_ss(s, sgs)
        Qt_ss = self.q_t_ss(t, sgs)
        return self.p_s_and_t_given_0(z_t_ss, Qt_ss, Qsb_ss, Qtb_ss)

    def sample_zs_from_zt_and_pred(self, z_t_a, z_t_ss, pred_a, pred_ss, t, s, node_mask, sgs):
        """Samples from zs ~ p(zs | zt). Only used during sampling. """

        # Retrieve transitions matrix
        #Qtb_a = self.q_t_bar_atom(t)
        #Qtb_ss = self.q_t_bar_ss(t)
        #Qsb_a = self.q_t_bar_atom(s)
        #Qsb_ss = self.q_t_bar_ss(s)
        #Qt_a = self.q_t_atom(t)
        #Qt_ss = self.q_t_ss(t)

        # Normalize predictions for the categorical features
        pred_a = F.softmax(pred_a, dim=-1)               # bs, n, d0
        pred_ss = F.softmax(pred_ss, dim=-1)              # bs, n, d0

        p_s_and_t_given_0_atom_types = self.p_s_and_t_given_0_a(z_t_a, t, s)
        p_s_and_t_given_0_site_symms = self.p_s_and_t_given_0_ss(z_t_ss, t, s, sgs)


        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_a = pred_a.unsqueeze(-1) * p_s_and_t_given_0_atom_types         # bs, n, d0, d_t-1
        unnormalized_prob_a = weighted_a.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_a[torch.sum(unnormalized_prob_a, dim=-1) == 0] = 1e-5
        prob_a = unnormalized_prob_a / torch.sum(unnormalized_prob_a, dim=-1, keepdim=True)  # bs, n, d_t-1

        weighted_ss = pred_ss.unsqueeze(-1) * p_s_and_t_given_0_site_symms         # bs, n, d0, d_t-1
        unnormalized_prob_ss = weighted_ss.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_ss[torch.sum(unnormalized_prob_ss, dim=-1) == 0] = 1e-5
        prob_ss = unnormalized_prob_ss / torch.sum(unnormalized_prob_ss, dim=-1, keepdim=True)  # bs, n, d_t-1


        assert ((prob_a.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_ss.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_a_s, sampled_ss_s = self.sample_discrete_features(prob_a, prob_ss, node_mask)
        return sampled_a_s, sampled_ss_s

    def discrete_loss(self, sample_a, sample_ss, pred_a, pred_ss):
        '''
        Cross entropy loss for atom_types as well as each site_symm component
        '''
        loss_a = F.cross_entropy(pred_a, sample_a.argmax(dim=-1))
        loss_ss = F.cross_entropy(pred_ss, sample_ss.argmax(dim=-1))
        return loss_a, loss_ss

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
    # site_symm : num_repr x 66
    return np.array(np.abs(1 - site_symm.cpu().detach().numpy()) < 0.1, dtype=float)


def modify_frac_coords_one(frac_coords, site_symm, atom_types, spacegroup):
    """
    perform replication for one crystal
    takes frac coords of representatives, corresponding predicted wyckoff labels, atom types along with spacegroup
    applies each wyckoff position on frac coords to obtain the entire orbit
    """

    spacegroup = Group(spacegroup.item())
    
    # convert the site_symm from one-hot to categorical
    int_wyckoff_labels = torch.where(site_symm != 0, site_symm, torch.tensor(float('-inf'), device=frac_coords.device)).argmax(dim=1)

    # get the string labels for wyckoff positions
    pred_wp_labels = wyckoff_category_to_labels([int_label.item() for int_label in int_wyckoff_labels])
    actual_spg_labels = [w.get_label() for w in spacegroup.Wyckoff_positions]
    if not set(pred_wp_labels).issubset(set(actual_spg_labels)):
        # check if the predicted set of wyckoff position belongs to the spacegroup
        hydra.utils.log.warning("Doesn't satisfy the spacegroup symmetry")
        return None, 0, None, None
    
    new_frac_coords, new_atom_types, new_site_symm = [], [], []
    # iterate over frac coords and corresponding site-symm
    for (pred_wp_label, sym, frac_coord, atm_type) in zip(pred_wp_labels, site_symm, frac_coords, atom_types):
        
        # get the wyckoff position based on the predicted site symmetry
        wp = spacegroup.get_wyckoff_position(pred_wp_label)
        
        # use wp operations on frac_coord
        frac_coord = frac_coord.cpu().detach().numpy()
        frac_coord = search_cloest_wp(spacegroup, wp, wp.ops[0], frac_coord)%1.
        for index in range(len(wp)):
            new_frac_coords.append(wp[index].operate(frac_coord)%1.)
            new_atom_types.append(atm_type.cpu().detach().numpy())
            new_site_symm.append(sym.cpu().detach().numpy())
            
    new_frac_coords = np.stack(new_frac_coords)
    new_atom_types = np.stack(new_atom_types)
    new_site_symm = np.stack(new_site_symm)
    return new_frac_coords, len(new_frac_coords), new_atom_types, new_site_symm
# def modify_frac_coords_one(frac_coords, site_symm, atom_types, spacegroup):
#     spacegroup = spacegroup.item()
    
    
#     # perform split-argmax to obtain binary representation
#     site_symm_argmax = split_argmax_sitesymm(site_symm)
    
#     # mapping from binary representation to hm-notation
#     wp_to_binary = dict()
#     for wp in Group(spacegroup).Wyckoff_positions:
#         wp.get_site_symmetry()
#         wp_to_binary[wp] = get_site_symmetry_binary_repr(CLUSTERED_SITES[wp.site_symm], label=wp.get_label()).numpy()
     
     
#     # iterate over frac coords and corresponding site-symm
#     new_frac_coords, new_atom_types = [], []
#     for (sym, frac_coord, atm_type) in zip(site_symm_argmax, frac_coords, atom_types):
#         frac_coord = frac_coord.cpu().detach().numpy()
        
#         # find all wps whose hm-notation align with sym
#         closes = []   
#         for wp, binary in wp_to_binary.items():
#             if np.sum(np.equal(binary, sym)) == sym.shape[-1]:
#                 close = search_cloest_wp(Group(spacegroup), wp, wp.ops[0], frac_coord)
#                 closes.append((close, wp, np.linalg.norm(np.minimum((close - frac_coord)%1., (frac_coord - close)%1.))))
#         try:
#             # pick the nearest wp to project
#             closest = sorted(closes, key=lambda x: x[-1])[0]
#             wyckoff = closest[1]
            
#             # use wp operations on frac_coord
#             frac_coord = closest[0]
#             for index in range(len(wyckoff)): 
#                 new_frac_coords.append(wyckoff[index].operate(frac_coord)%1.)
#                 new_atom_types.append(atm_type.cpu().detach().numpy())
#         except:
#             print('Weird things happen, and I did not predict correctly')
#             new_frac_coords.append(frac_coord)
#             new_atom_types.append(atm_type.cpu().detach().numpy())
        
#     new_frac_coords = np.stack(new_frac_coords)
#     new_atom_types = np.stack(new_atom_types)
#     return new_frac_coords, len(new_frac_coords), new_atom_types

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
                    traj['site_symm'][total_atoms:total_atoms+num_repr[index]], # num_repr x 66
                    traj['atom_types'][total_atoms:total_atoms+num_repr[index]], # num_repr x 100
                    spacegroups[index], 
                )

            if new_num_atoms:
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
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler).to(self.device)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler).to(self.device)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.spacegroup_embedding = build_mlp(in_dim=N_SPACEGROUPS, hidden_dim=128, fc_num_layers=2, out_dim=self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.use_ks = self.hparams.use_ks
        self.discrete_noise = self.init_discrete_noise()


    def init_discrete_noise(self):
        atom_type_marginals = torch.load('/home/mila/d/daniel.levy/scratch/MatSci/intel-mat-diffusion/data/mp_20/train_atom_types_marginals.pt')
        site_symm_marginals = torch.load('/home/mila/d/daniel.levy/scratch/MatSci/intel-mat-diffusion/data/mp_20/train_wyckoff_marginals_per_sg.pt')
        return DiscreteNoise(atom_type_marginals, site_symm_marginals, self.beta_scheduler)

    def forward(self, batch):

        batch_size = batch.num_graphs
        dummy_repr_ind = batch.dummy_repr_ind
        atom_types, node_mask = to_dense_batch(batch.atom_types, batch.batch, fill_value=0)
        site_symms, node_mask = to_dense_batch(batch.wyckoff_labels, batch.batch, fill_value=0)

        gt_spacegroup_onehot = F.one_hot(batch.spacegroup - 1, num_classes=N_SPACEGROUPS).float()
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times) + self.spacegroup_embedding(gt_spacegroup_onehot)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

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
            input_ks = c0[:, None] * ks + c1[:, None] * rand_ks
            input_ks = mask_ks(input_ks, ks_mask, ks_add)
            input_lattice = lattice_ks_to_matrix_torch(input_ks)
        else:
            input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
            
            
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        gt_atom_types_onehot = F.one_hot(atom_types, num_classes=MAX_ATOMIC_NUM).float()
        #gt_site_symm_binary = site_symms
        gt_site_symm_binary = F.one_hot(site_symms, num_classes=NUM_WYCKOFF).float()
        

        rand_t = torch.randn_like(gt_atom_types_onehot)
        rand_symm = torch.randn_like(gt_site_symm_binary)

        atom_type_probs = self.discrete_noise.apply_atom_noise(gt_atom_types_onehot, times)
        site_symm_probs = self.discrete_noise.apply_site_symm_noise(gt_site_symm_binary, times, batch.spacegroup)
        atom_types, site_symms = self.discrete_noise.sample_discrete_features(atom_type_probs, site_symm_probs, node_mask)
        #atom_types = self.discrete_noise.sample_atom_types(atom_type_probs)
        #site_symms = self.discrete_noise.sample_site_symms(site_symm_probs)
        #atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)
        #site_symm_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_site_symm_binary + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_symm)

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices
            input_ks = ks

        # pass noised site symmetries and behave similar to atom type probs
        lattice_feats = input_ks if self.use_ks else input_lattice
        pred_lattice, pred_x, pred_t, pred_symm = self.decoder(time_emb, atom_types[node_mask], input_frac_coords, 
                                                    lattice_feats, input_lattice, batch.num_atoms, 
                                                    batch.batch, site_symm_probs=site_symms[node_mask])

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        
        loss_lattice = F.mse_loss(pred_lattice, ks_mask * rand_ks) if self.use_ks else F.mse_loss(pred_lattice, rand_l)

        # loss_coord = F.mse_loss(pred_x * (1 - dummy_repr_ind), tar_x * (1 - dummy_repr_ind))
        loss_coord = F.mse_loss(pred_x, tar_x)
        
        #loss_type = F.cross_entropy(pred_t, batch.atom_types)
        
        # loss_symm = F.mse_loss(pred_symm * (1 - dummy_repr_ind), rand_symm * (1 - dummy_repr_ind))
        #loss_symm = F.mse_loss(pred_symm, rand_symm)

        loss_type, loss_symm = self.discrete_noise.discrete_loss(atom_types[node_mask], site_symms[node_mask], pred_t, pred_symm)

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

        ks_mask, ks_add = sg_to_ks_mask(batch.spacegroup)
        k_T = torch.randn([batch_size, 6]).to(self.device)
        k_T = mask_ks(k_T, ks_mask, ks_add)
        l_T = lattice_ks_to_matrix_torch(k_T)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)

        # TODO: there must be an easier way to do this
        _, node_mask = to_dense_batch(batch.batch, batch.batch, fill_value=0)
        t_T, symm_T = self.discrete_noise.sample_limit_dist(node_mask, batch.spacegroup)
        t_T = t_T[node_mask]
        symm_T = symm_T[node_mask]

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

            # get diffusion timestep embeddings, concatenated with spacegroup condition    
            gt_spacegroup_onehot = F.one_hot(batch.spacegroup - 1, num_classes=N_SPACEGROUPS).float()
            time_emb = self.time_embedding(times) + self.spacegroup_embedding(gt_spacegroup_onehot)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

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
            rand_k = torch.randn_like(k_T) if t > 1 else torch.zeros_like(k_T) 
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

            #rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            #rand_symm = torch.randn_like(symm_T) if t > 1 else torch.zeros_like(symm_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   
            lattice_feats_t_minus_05 = k_t_minus_05 if self.use_ks else l_t_minus_05

            pred_l, pred_x, pred_t, pred_symm = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, 
                                                        lattice_feats_t_minus_05, l_t_minus_05, batch.num_atoms, 
                                                        batch.batch, site_symm_probs=symm_t_minus_05)
            pred_t, _ = to_dense_batch(pred_t, batch.batch, fill_value=0)
            pred_symm, _ = to_dense_batch(pred_symm, batch.batch, fill_value=0)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            if self.use_ks:
                k_t_minus_1 = c0 * (k_t_minus_05 - c1 * pred_l) + sigmas * rand_k if not self.keep_lattice else k_t
                k_t_minus_1 = mask_ks(k_t_minus_1, ks_mask, ks_add)
                l_t_minus_1 = lattice_ks_to_matrix_torch(k_t_minus_1) if not self.keep_lattice else l_t
            else:
                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t
                k_t_minus_1 = k_t
            t_t_minus_05, _ = to_dense_batch(t_t_minus_05, batch.batch, fill_value=0)
            symm_t_minus_05, _ = to_dense_batch(symm_t_minus_05, batch.batch, fill_value=0)
            t_t_minus_1, symm_t_minus_1 = self.discrete_noise.sample_zs_from_zt_and_pred(t_t_minus_05, symm_t_minus_05, pred_t, pred_symm, times, times-1, node_mask, batch.spacegroup)
            #t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t
            #symm_t_minus_1 = c0 * (symm_t_minus_05 - c1 * pred_symm) + sigmas * rand_symm
            t_t_minus_1 = t_t_minus_1[node_mask]
            symm_t_minus_1 = symm_t_minus_1[node_mask]

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
        #assert traj[0]['frac_coords'].size(0) == traj[0]['atom_types'].size(0) == traj[0]['num_atoms'].sum()
        #assert traj[0]['ks'].size(0) == traj[0]['lattices'].size(0) == traj[0]['num_atoms'].size(0)

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
            on_step=True,
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

    