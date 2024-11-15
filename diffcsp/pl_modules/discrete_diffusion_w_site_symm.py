import math, copy
import json
import os
import numpy as np
from p_tqdm import p_map, t_map
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader


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
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc, lattice_ks_to_matrix_torch,
    sg_to_ks_mask, mask_ks, N_SPACEGROUPS)

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
from diffcsp.pl_modules.model import build_mlp
from scripts.generation import SampleDataset
from scripts.compute_metrics import Crystal, GenEval, get_gt_crys_ori
from scripts.eval_utils import lattices_to_params_shape,  get_crystals_list


MAX_ATOMIC_NUM=94
NUM_WYCKOFF = 186
NUM_SPACEGROUPS = 231
SITE_SYMM_AXES = 15
SITE_SYMM_PGS = 13
SITE_SYMM_DIM = SITE_SYMM_AXES * SITE_SYMM_PGS
SG_CONDITION_DIM = 397
SG_SYM = {spacegroup: Group(spacegroup) for spacegroup in range(1, 231)}
SG_TO_WP_TO_SITE_SYMM = dict()
for spacegroup in range(1, 231):
    SG_TO_WP_TO_SITE_SYMM[spacegroup] = dict()
    for wp in SG_SYM[spacegroup].Wyckoff_positions:
        wp.get_site_symmetry()
        SG_TO_WP_TO_SITE_SYMM[spacegroup][wp] = wp.get_site_symmetry_object().to_one_hot()

class DiscreteNoise(nn.Module):
    def __init__(self, atom_type_prior, site_symm_prior_per_sg, beta_scheduler, P_ss, P_a):
        super().__init__()
        self.beta_scheduler = beta_scheduler
        self.site_symm_prior_per_sg = site_symm_prior_per_sg
        self.atom_type_prior = atom_type_prior
        self.P_ss = P_ss 
        self.P_a = P_a 
        self.site_symm_pgs = SITE_SYMM_PGS
        self.site_symm_axes = SITE_SYMM_AXES
        self.max_atomic_num = MAX_ATOMIC_NUM

    def ss_to_sections(self, ss):
        return [ss[..., i*self.site_symm_pgs:(i+1)*self.site_symm_pgs] for i in range(self.site_symm_axes)]

    def reshape_ss(self, ss):
        return ss.reshape(-1, SITE_SYMM_AXES, self.site_symm_pgs) 

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
        return [self.q_t(self.P_ss[i][sgs], t) for i in range(len(self.P_ss))]

    def q_t_bar(self, P, t):
        alpha_bar = self.beta_scheduler.alphas_cumprod[t]
        num_classes = P.shape[-1]
        return alpha_bar.view(-1,1,1)*torch.eye(num_classes, device=P.device)+ (1 - alpha_bar.view(-1,1,1))*P
    
    def q_t_bar_atom(self, t):
        return self.q_t_bar(self.P_a, t)

    def q_t_bar_ss(self, t, sgs):
        return [self.q_t_bar(self.P_ss[i][sgs], t) for i in range(len(self.P_ss))]

    def sigma_sqr_ratio(self, s_int, t_int):
        return self.beta_scheduler.alphas_cumprod[t_int] / self.beta_scheduler.alphas_cumprod[s_int]
    
    def apply_atom_noise(self, atom_type, t):
        Q_t_bar = self.q_t_bar_atom(t)
        prob_atom_types = atom_type @  Q_t_bar
        return prob_atom_types
    
    def apply_site_symm_noise(self, site_symm, t, sgs):
        Q_t_bars = self.q_t_bar_ss(t, sgs)
        prob_site_symms = self.multiply_block_diagonal(Q_t_bars, site_symm)
        return prob_site_symms

    def sample_atom_types(self, atom_probs):
        return F.one_hot(torch.multinomial(atom_probs, 1).reshape(-1), self.max_atomic_num).float()

    def sample_site_symms(self, site_symms):
        outs = []
        idx = 0
        for ni in self.ss_lengths:
            outs.append(F.one_hot(torch.multinomial(site_symms[..., idx:idx+ni], 1).reshape(-1), ni).float())
            idx += ni
        return torch.cat(outs, 1)
    
    def sample_limit_dist(self, node_mask, sgs):
        """ Sample from the limit distribution of the diffusion process"""
        bs, n_max = node_mask.shape
        a_limit = self.atom_type_prior.expand(bs, n_max, -1)
        U_a = a_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_a = F.one_hot(U_a, num_classes=a_limit.shape[-1]).float()
        U_a = U_a * node_mask.unsqueeze(-1)

        U_ss_list = []
        for ss_priors_i_per_sg in self.site_symm_prior_per_sg:
            ss_priors_i = ss_priors_i_per_sg[sgs]
            ss_limit_i = ss_priors_i.unsqueeze(-2).expand(bs, n_max, -1)
            U_ss_i = ss_limit_i.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
            U_ss_i = F.one_hot(U_ss_i, num_classes=ss_limit_i.shape[-1]).float()
            U_ss_list.append(U_ss_i)
        U_ss = torch.cat(U_ss_list, dim=-1)

        return U_a, U_ss

    def sample_discrete_features(self, prob_a, prob_ss, node_mask):
        ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
            :param prob_a: bs, n, dx_out        node features
            :param prob_ss: bs, n, dx_out        node features
        '''
        bs, n = node_mask.shape
        # The masked rows should define probability distributions as well
        prob_a[~node_mask] = 1 / prob_a.shape[-1]
        prob_ss_list = self.ss_to_sections(prob_ss)
        #prob_ss_norm_list = []
        for i in range(SITE_SYMM_AXES):
            prob_ss_list[i][~node_mask] = 1 / prob_ss_list[i].shape[-1]
        # Flatten the probability tensor to sample with multinomial
        prob_a = prob_a.reshape(bs * n, -1)       # (bs * n, dx_out)
        # Sample a
        atom_t = prob_a.multinomial(1)                                  # (bs * n, 1)
        atom_t = atom_t.reshape(bs, n)     # (bs, n)
        atom_t = F.one_hot(atom_t, num_classes=prob_a.shape[-1]).float()
        # Sample ss
        site_symm_t_list = []
        for i in range(SITE_SYMM_AXES):
            prob_ss_i = prob_ss_list[i].reshape(bs * n, -1)       # (bs * n, dx_out)
            site_symm_t_i_cat = prob_ss_i.multinomial(1).reshape(bs, n)
            site_symm_t_i =  F.one_hot(site_symm_t_i_cat, num_classes=self.site_symm_pgs).float()
            site_symm_t_list.append(site_symm_t_i)
        site_symm_t = torch.cat(site_symm_t_list, -1)

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
        p_s_and_t_given_0_site_symms = []#torch.zeros((list(z_t_ss.shape[:-1]) +  [27, 0]), device=z_t_ss.device)
        for i in range(len(self.P_ss)):
            p_s_and_t_given_0_site_symms.append(self.p_s_and_t_given_0(z_t_ss[i], Qt_ss[i], Qsb_ss[i], Qtb_ss[i]))
        return p_s_and_t_given_0_site_symms

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
        z_t_ss_split = self.ss_to_sections(z_t_ss)
        p_s_and_t_given_0_atom_types = self.p_s_and_t_given_0_a(z_t_a, t, s)
        p_s_and_t_given_0_site_symms = self.p_s_and_t_given_0_ss(z_t_ss_split, t, s, sgs)


        # Dim of these two tensors: bs, N, d0, d_t-1
        #pred_a = F.softmax(pred_a, dim=-1)               # bs, n, d0
        weighted_a = pred_a.unsqueeze(-1) * p_s_and_t_given_0_atom_types         # bs, n, d0, d_t-1
        unnormalized_prob_a = weighted_a.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_a[torch.sum(unnormalized_prob_a, dim=-1) == 0] = 1e-5
        prob_a = unnormalized_prob_a / torch.sum(unnormalized_prob_a, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_ss_split = self.ss_to_sections(pred_ss)
        prob_ss_list = []
        for pred_ss_i, p_s_and_t_given_0_site_symms_i in zip(pred_ss_split, p_s_and_t_given_0_site_symms):
            #pred_ss_i = F.softmax(pred_ss_i, dim=-1)              # bs, n, d0
            weighted_ss = pred_ss_i.unsqueeze(-1) * p_s_and_t_given_0_site_symms_i         # bs, n, d0, d_t-1
            unnormalized_prob_ss = weighted_ss.sum(dim=2)                     # bs, n, d_t-1
            unnormalized_prob_ss[torch.sum(unnormalized_prob_ss, dim=-1) == 0] = 1e-5
            prob_ss = unnormalized_prob_ss / torch.sum(unnormalized_prob_ss, dim=-1, keepdim=True)  # bs, n, d_t-1
            prob_ss_list.append(prob_ss)
        prob_ss = torch.cat(prob_ss_list, -1)

        assert ((prob_a.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_ss.sum(dim=-1) - len(prob_ss_list)).abs() < 1e-4).all()

        sampled_a_s, sampled_ss_s = self.sample_discrete_features(prob_a, prob_ss, node_mask)
        return sampled_a_s, sampled_ss_s

    def discrete_loss(self, sample_a, sample_ss, pred_a, pred_ss):
        '''
        Cross entropy loss for atom_types as well as each site_symm component
        '''
        loss_a = F.nll_loss(torch.log(pred_a + 1e-20), sample_a)
        pred_ss_split = self.ss_to_sections(pred_ss)
        losses = []
        for i, pred_ss_i in enumerate(pred_ss_split):
            loss_ss_i = F.nll_loss(torch.log(pred_ss_i +  1e-20), sample_ss[..., i])
            losses.append(loss_ss_i)
        loss_ss = torch.stack(losses).mean()
        return loss_a, loss_ss

class DiscreteNoiseMarginal(DiscreteNoise):
    def __init__(self, atom_marginals_path, ss_marginals_path, beta_scheduler):
        atom_type_prior = torch.load(atom_marginals_path)
        site_symm_prior_per_sg = torch.load(ss_marginals_path)
        P_ss = nn.ParameterList([nn.Parameter(site_symm_prior_per_sg[i].unsqueeze(-2).expand(NUM_SPACEGROUPS, SITE_SYMM_PGS, SITE_SYMM_PGS).clone(), requires_grad=False) for i in range(SITE_SYMM_AXES)])
        P_a = nn.Parameter(atom_type_prior.unsqueeze(0).expand(MAX_ATOMIC_NUM, -1).clone(), requires_grad=False)
        super().__init__(atom_type_prior, site_symm_prior_per_sg, beta_scheduler, P_ss, P_a)


class DiscreteNoiseMasked(DiscreteNoise):
    def __init__(self, beta_scheduler):
        atom_type_prior = torch.zeros(MAX_ATOMIC_NUM + 1)
        atom_type_prior[-1] = 1
        site_symm_prior = [torch.zeros(SITE_SYMM_PGS + 1) for i in range(SITE_SYMM_AXES)]
        for i in range(SITE_SYMM_AXES):
            site_symm_prior[i][-1] = 1
        site_symm_prior_per_sg = [site_symm_prior_i.expand(NUM_SPACEGROUPS, -1) for site_symm_prior_i in site_symm_prior]
        P_ss = nn.ParameterList([nn.Parameter(site_symm_prior[i].unsqueeze(-2).expand(NUM_SPACEGROUPS, SITE_SYMM_PGS+1, SITE_SYMM_PGS+1).clone(), requires_grad=False) for i in range(SITE_SYMM_AXES)])
        P_a = nn.Parameter(atom_type_prior.unsqueeze(0).expand(MAX_ATOMIC_NUM+1, -1).clone(), requires_grad=False)
        super().__init__(atom_type_prior, site_symm_prior_per_sg, beta_scheduler, P_ss, P_a)
        self.max_atomic_num = MAX_ATOMIC_NUM + 1
        self.site_symm_pgs = SITE_SYMM_PGS + 1

    def sub_predictions(self, pred_a, pred_ss, atom_types, site_symms):
        # return pred_a, pred_ss
        # Modify atom and site symmetry predictions to account for masked tokens

        # Never predict masked tokens – zero them out
        mask_mask_atom = torch.zeros_like(pred_a) + 1
        mask_mask_atom[:, -1] = 0
        pred_a = pred_a * mask_mask_atom
        mask_mask_symm = torch.zeros_like(pred_ss) + 1
        self.reshape_ss(mask_mask_symm)[:,:,-1] = 0
        pred_ss = pred_ss * mask_mask_symm

        # If something is unmasked, keep it unmasked instead of predicting
        unmasked_atom = (atom_types[..., -1] == 0)[:,None].expand(-1, self.max_atomic_num)
        pred_a = torch.where(unmasked_atom, atom_types, pred_a)
        unmasked_symm = ((self.reshape_ss(site_symms)[:, :, -1] == 0)[:,:,None].expand(-1, -1, self.site_symm_pgs)).flatten(-2, -1)
        pred_ss = torch.where(unmasked_symm, site_symms, pred_ss)

        return pred_a, pred_ss

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
    spacegroup = spacegroup.item()
    site_symm_axis = site_symm.reshape(-1, SITE_SYMM_AXES, SITE_SYMM_PGS).detach().cpu()
    # Get site symmetry of each WP for the spacegroup
    wp_to_site_symm = SG_TO_WP_TO_SITE_SYMM[spacegroup]

    # iterate over frac coords and corresponding site-symm
    new_frac_coords, new_atom_types, new_site_symm = [], [], []
    min_ss_dists = []
    wp_projection_dists = []
    for (sym, frac_coord, atm_type) in zip(site_symm_axis, frac_coords, atom_types):
        frac_coord = frac_coord.cpu().detach().numpy()

        # Get all WPs that are closest in terms of site symmetry
        wp_to_ss_dist = {wp: torch.norm(sym.flatten() - ss.flatten()) for wp, ss in wp_to_site_symm.items()}
        min_ss_dist = min(wp_to_ss_dist.values())
        min_ss_dists.append(min_ss_dist.item())
        closest_ss_wps = [wp for wp, dist in wp_to_ss_dist.items() if dist==min_ss_dist]

        # For each WP find closest position in space
        closes = []
        for wp in closest_ss_wps:
            for orbit_index in range(len(wp.ops)):
                close = search_cloest_wp(SG_SYM[spacegroup], wp, wp.ops[orbit_index], frac_coord)%1.
                closes.append((close, wp, orbit_index, np.linalg.norm(np.minimum((close - frac_coord)%1., (frac_coord - close)%1.))))
        try:
            # pick the nearest wp to project
            closest = sorted(closes, key=lambda x: x[-1])[0]
            wyckoff = closest[1]
            repr_index = closest[2]
            wp_projection_dists.append(closest[3])
            # use wp operations on frac_coord
            frac_coord = closest[0]
            for index in range(len(wyckoff)):
                # new_frac_coords.append(wyckoff[index].operate(frac_coord)%1.)
                new_frac_coords.append(wyckoff[(index + repr_index) % len(wyckoff)].operate(frac_coord)%1.)
                new_atom_types.append(atm_type.cpu().detach().numpy())
                new_site_symm.append(sym)
        except:
            # print('Weird things happen, and I did not predict correctly')
            new_frac_coords.append(frac_coord)
            new_atom_types.append(atm_type.cpu().detach().numpy())
            new_site_symm.append(sym.cpu().detach().numpy())
            
    new_frac_coords = np.stack(new_frac_coords)
    new_atom_types = np.stack(new_atom_types)
    new_site_symm = np.stack(new_site_symm)
    return new_frac_coords, len(new_frac_coords), new_atom_types, new_site_symm, min_ss_dists, wp_projection_dists


def modify_frac_coords(traj:Dict, spacegroups:List[int], num_repr:List[int]) -> Dict:
    device = traj['frac_coords'].device
    total_atoms = 0
    updated_frac_coords = []
    updated_num_atoms = []
    updated_atom_types = []
    updated_site_symm = []
    min_ss_dists, wp_projection_dists = [], []
    
    for index in range(len(num_repr)):
        if num_repr[index] > 0:
            # if something is predicted, otherwise it is an empty crystal which we are post-processing
            # this might happen if we predict a crystal with only dummy representative atoms
            new_frac_coords, new_num_atoms, new_atom_types, new_site_sym,  min_ss_dist, wp_projection_dist = modify_frac_coords_one(
                    traj['frac_coords'][total_atoms:total_atoms+num_repr[index]], # num_repr x 3
                    traj['site_symm'][total_atoms:total_atoms+num_repr[index]], # num_repr x 195
                    traj['atom_types'][total_atoms:total_atoms+num_repr[index]], # num_repr x 100
                    spacegroups[index], 
                )

            if new_num_atoms:
                updated_frac_coords.append(new_frac_coords)
                updated_num_atoms.append(new_num_atoms)
                updated_atom_types.append(new_atom_types)
                updated_site_symm.append(new_site_sym)
                min_ss_dists.append(min_ss_dist)
                wp_projection_dists.append(wp_projection_dist)
        
        total_atoms += num_repr[index]
    
    traj['frac_coords'] = torch.cat([torch.from_numpy(x) for x in updated_frac_coords]).to(device)
    traj['atom_types'] = torch.cat([torch.from_numpy(x) for x in updated_atom_types]).to(device)
    traj['num_atoms'] = torch.tensor(updated_num_atoms).to(device)
    traj['site_symm'] = torch.cat([torch.from_numpy(x) for x in updated_site_symm]).to(device)
    traj['min_ss_dists'] = min_ss_dists
    traj['wp_projection_dists'] = wp_projection_dists

    
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
        mask_token = 1 if self.hparams.prior == 'masked' else 0 
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, time_dim=self.hparams.time_dim + self.hparams.latent_dim, latent_dim = self.hparams.latent_dim, pred_type = True, pred_site_symm_type = True, smooth = True, max_atoms=MAX_ATOMIC_NUM+mask_token, mask_token=mask_token)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler).to(self.device)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler).to(self.device)
        self.time_dim = self.hparams.time_dim
        self.latent_dim = self.hparams.latent_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.spacegroup_embedding = build_mlp(in_dim=SG_CONDITION_DIM, hidden_dim=128, fc_num_layers=2, out_dim=self.latent_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.use_ks = self.hparams.use_ks
        self.discrete_noise = self.init_discrete_noise(self.hparams.prior)

    def on_train_start(self):
        log_dict = {
            'comp_valid': 0,
            'struct_valid': 0,
            'valid': 0
        }
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def init_discrete_noise(self, prior='marginal'):
        if prior == 'marginal':
            return DiscreteNoiseMarginal(self.hparams.data.datamodule.atom_marginals_path,
                                         self.hparams.data.datamodule.ss_marginals_path, self.beta_scheduler)
        elif prior == 'masked':
            return DiscreteNoiseMasked(self.beta_scheduler)

    def forward(self, batch):

        batch_size = batch.num_graphs
        dummy_repr_ind = batch.dummy_repr_ind
        atom_types, node_mask = to_dense_batch(batch.atom_types - 1, batch.batch, fill_value=0)
        if self.hparams.prior == 'masked':
            site_symm = torch.cat([batch.site_symm, torch.zeros_like(batch.site_symm)[..., :1]], dim=-1)
        else:
            site_symm = batch.site_symm
        site_symms, node_mask = to_dense_batch(site_symm.flatten(-2, -1), batch.batch, fill_value=0)
        gt_spacegroup_onehot = F.one_hot(batch.spacegroup - 1, num_classes=N_SPACEGROUPS).float()
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        spacegroup_emb = self.spacegroup_embedding(batch.sg_condition.reshape(-1, SG_CONDITION_DIM))
        time_emb = torch.cat([self.time_embedding(times), spacegroup_emb], dim=-1)
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

        gt_atom_types_onehot = F.one_hot(atom_types, num_classes=self.discrete_noise.max_atomic_num).float()
        gt_site_symm_binary = site_symms

        atom_type_noised_probs = self.discrete_noise.apply_atom_noise(gt_atom_types_onehot, times)
        site_symm_noised_probs = self.discrete_noise.apply_site_symm_noise(gt_site_symm_binary, times, batch.spacegroup)
        atom_types_noised, site_symms_noised = self.discrete_noise.sample_discrete_features(atom_type_noised_probs, site_symm_noised_probs, node_mask)

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices
            input_ks = ks

        # pass noised site symmetries and behave similar to atom type probs
        lattice_feats = input_ks if self.use_ks else input_lattice
        symm_t = site_symms_noised[node_mask]
        atom_types_t = atom_types_noised[node_mask]
        pred_lattice, pred_x, pred_t_logit, pred_symm_logit = self.decoder(time_emb, atom_types_t, input_frac_coords, 
                                                    lattice_feats, input_lattice, batch.num_atoms, 
                                                    batch.batch, site_symm_probs=symm_t)
        
        pred_t = F.softmax(pred_t_logit, -1)
        pred_symm = F.softmax(self.discrete_noise.reshape_ss(pred_symm_logit), -1).flatten(-2, -1)
        if self.hparams.prior == 'masked':
            pred_t, pred_symm = self.discrete_noise.sub_predictions(pred_t, pred_symm, atom_types_t, symm_t)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        
        loss_lattice = F.mse_loss(pred_lattice, ks_mask * rand_ks) if self.use_ks else F.mse_loss(pred_lattice, rand_l)

        loss_coord = F.mse_loss(pred_x, tar_x)
        
        loss_type, loss_symm = self.discrete_noise.discrete_loss(batch.atom_types - 1, batch.site_symm.argmax(-1), pred_t, pred_symm)

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
            #gt_spacegroup_onehot = F.one_hot(batch.spacegroup - 1, num_classes=N_SPACEGROUPS).float()
            #time_emb = torch.cat([self.time_embedding(times), self.spacegroup_embedding(gt_spacegroup_onehot)], dim=-1)
            spacegroup_emb = self.spacegroup_embedding(batch.sg_condition.reshape(-1, SG_CONDITION_DIM))
            time_emb = torch.cat([self.time_embedding(times), spacegroup_emb], dim=-1)

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

            pred_l, pred_x, pred_t_logit, pred_symm_logit = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, 
                                                        lattice_feats_t_minus_05, l_t_minus_05, batch.num_atoms, 
                                                        batch.batch, site_symm_probs=symm_t_minus_05)

            # Convert logits to probabilities
            pred_t = F.softmax(pred_t_logit, -1)
            pred_symm = F.softmax(self.discrete_noise.reshape_ss(pred_symm_logit), -1).flatten(-2, -1)
            if self.hparams.prior == 'masked':
                pred_t, pred_symm = self.discrete_noise.sub_predictions(pred_t, pred_symm, t_t_minus_05, symm_t_minus_05)

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
        #dummy_ind = (traj[0]['atom_types'].argmax(dim=-1) + 1 == self.discrete_noise.max_atomic_num).long()
        dummy_ind = (traj[0]['atom_types'].argmax(dim=-1) == self.discrete_noise.max_atomic_num).long()
        traj[0]['frac_coords'] = traj[0]['frac_coords'][(1 - dummy_ind).bool()]
        traj[0]['atom_types'] = traj[0]['atom_types'][(1 - dummy_ind).bool()]
        traj[0]['site_symm'] = traj[0]['site_symm'][(1 - dummy_ind).bool()]
        if self.hparams.prior == 'masked':
            # Get rid of masking dimension
            traj[0]['site_symm'] = traj[0]['site_symm'].reshape(-1, SITE_SYMM_AXES, self.discrete_noise.site_symm_pgs)[..., :SITE_SYMM_PGS].flatten(-2, -1)
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

        if (self.current_epoch + 1) % self.hparams.data.eval_every_epoch == 0 and batch_idx == 0:
            # run a simpler evaluation
            self.simple_gen_evaluation()

        return loss

    def simple_gen_evaluation(self):
        
        eval_model_name_dataset = {
            "mp20": "mp", # encompasses mp20, mpsa52
            "perovskite": "perov",
            "carbon": "carbon",
        }
        test_set = SampleDataset(
                            eval_model_name_dataset[self.hparams.data.eval_model_name], 
                            self.hparams.data.eval_generate_samples, 
                            self.hparams.data.datamodule.datasets.train.save_path,
                            self.hparams.data.datamodule.datasets.train.sg_info_path,)
        
        test_loader = DataLoader(test_set, batch_size = 50)
        frac_coords = []
        num_atoms = []
        atom_types = []
        lattices = []
        spacegroups = []
        site_symmetries = []
        for idx, batch in enumerate(test_loader):

            if torch.cuda.is_available():
                batch.cuda()
            outputs, traj = self.sample(batch, step_lr = 1e-5)
            del traj
            frac_coords.append(outputs['frac_coords'].detach().cpu())
            num_atoms.append(outputs['num_atoms'].detach().cpu())
            atom_types.append(outputs['atom_types'].detach().cpu())
            lattices.append(outputs['lattices'].detach().cpu())
            spacegroups.append(outputs['spacegroup'].detach().cpu())
            site_symmetries.append(outputs['site_symm'].detach().cpu())
            del outputs

        frac_coords = torch.cat(frac_coords, dim=0)
        num_atoms = torch.cat(num_atoms, dim=0)
        atom_types = torch.cat(atom_types, dim=0)
        lattices = torch.cat(lattices, dim=0)
        spacegroups = torch.cat(spacegroups, dim=0)
        site_symmetries = torch.cat(site_symmetries, dim=0)
        lengths, angles = lattices_to_params_shape(lattices)
        
        # generated crystals
        kwargs = {"spacegroups": spacegroups, "site_symmetries": site_symmetries}
        pred_crys_array_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms, **kwargs)
        gen_crys = p_map(lambda x: Crystal(x), pred_crys_array_list)
        print(f"INFO: Done generating {self.hparams.data.eval_generate_samples} crystals (Epoch: {self.current_epoch + 1})")
        
        # ground truth crystals
        if os.path.exists(self.hparams.data.datamodule.datasets.val[0].gt_crys_path):
            gt_crys = torch.load(self.hparams.data.datamodule.datasets.val[0].gt_crys_path)
        else:
            csv = pd.read_csv(self.hparams.data.datamodule.datasets.val[0].path)
            gt_crys = t_map(get_gt_crys_ori, csv['cif'])
            torch.save(gt_crys, self.hparams.data.datamodule.datasets.val[0].gt_crys_path)
            
        print(f"INFO: Done reading ground truth crystals (Epoch: {self.current_epoch + 1})")
        
        gen_evaluator = GenEval(gen_crys, gt_crys, n_samples=0, eval_model_name=self.hparams.data.eval_model_name,
                                gt_prop_eval_path=self.hparams.data.datamodule.datasets.val[0].gt_prop_eval_path)
        gen_metrics = gen_evaluator.get_metrics()
        print(gen_metrics)
        
        self.log_dict(gen_metrics)

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

    