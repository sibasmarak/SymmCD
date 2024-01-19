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
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset
from eval_utils import load_model, lattices_to_params_shape, get_crystals_list
from diffcsp.common.constants import SpaceGroupDist
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc, lattice_ks_to_matrix_torch, sg_to_ks_mask, mask_ks, N_SPACEGROUPS)
MAX_ATOMIC_NUM=100

from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from pyxtal.symmetry import Group
import chemparse
import numpy as np
from p_tqdm import p_map
from torch.autograd import grad

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
    
def apply_symmop_torch(affine_matrix, frac_coords):
    ones_vec = torch.ones(frac_coords.shape[0], 1, device=frac_coords.device)
    affine_points = torch.cat([frac_coords, ones_vec], 1)
    if affine_matrix.dim() == 2:
        affine_matrix = affine_matrix.unsqueeze(0)
    mult  = torch.inner(affine_points, affine_matrix)
    mult = mult.transpose(0, 1)[...,:-1]
    return mult %1 


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
def dist_periodic(x, y):
    dx = x - y
    return torch.round(dx) - dx


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y, species):
        # The Sinkhorn algorithm takes as input three variables :
        C, dist_matrix_sqr = self._cost_matrix(x, y, species)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=x.device).fill_(1.0 / x_points).squeeze(0)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=y.device).fill_(1.0 / y_points).squeeze(0)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C, dist_matrix_sqr

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, species, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        dist_matrix_sqr = torch.sum((torch.abs(dist_periodic(x_col, y_lin))) ** p, -1)
        C = dist_matrix_sqr* torch.abs(1/(species @ species.T) + 1e-5)
        return C, dist_matrix_sqr

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

@torch.no_grad()
def sample_sym(self, batch, diff_ratio = 1.0, step_lr = 1e-5, symmops=None, num_transformations=48, sym_loss_alpha=0.1, silent=False):
    sinkhorn = SinkhornDistance(eps = 0.1, max_iter=100)
    batch_size = batch.num_graphs
    if self.use_spacegroup and self.use_ks:
            ks_mask, ks_add = sg_to_ks_mask(batch.spacegroup)

    if self.use_ks:
        k_T = torch.randn([batch_size, 6]).to(self.device)
        if self.use_spacegroup:
            k_T = mask_ks(k_T, ks_mask, ks_add)
        l_T = lattice_ks_to_matrix_torch(k_T)
    else:
        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        k_T = torch.zeros([batch_size, 6]).to(self.device) # dummy 
    x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
    t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)


    if self.keep_coords:
        x_T = batch.frac_coords

    if self.keep_lattice:
        k_T = batch.ks
        l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)     

    traj = {self.beta_scheduler.timesteps : {
        'num_atoms' : batch.num_atoms,
        'atom_types' : t_T,
        'frac_coords' : x_T % 1.,
        'lattices' : l_T,
        'ks': k_T
    }}

    for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1), disable=silent):

        times = torch.full((batch_size, ), t, device = self.device)

        time_emb = self.time_embedding(times)
        
        alphas = self.beta_scheduler.alphas[t]
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

        sigmas = self.beta_scheduler.sigmas[t]
        sigma_x = self.sigma_scheduler.sigmas[t]
        sigma_norm = self.sigma_scheduler.sigmas_norm[t]

        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

        x_t = traj[t]['frac_coords']
        l_t = traj[t]['lattices']
        k_t = traj[t]['ks']
        t_t = traj[t]['atom_types']

        if self.keep_coords:
            x_t = x_T

        if self.keep_lattice:
            k_t = k_T
            l_t = l_T

        # Corrector
        # For whatever reason, lattice parameters are not updated in the original code.
        if self.use_ks:
            rand_k = torch.randn_like(k_T) if t > 1 else torch.zeros_like(k_T)
        else:
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
        rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

        step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
        std_x = torch.sqrt(2 * step_size)

        lattice_feats_t = k_t if self.use_ks else l_t
        pred_lattice, pred_x, pred_t = self.decoder(time_emb, t_t, x_t, lattice_feats_t, l_t, batch.num_atoms, batch.batch, batch.spacegroup)

        pred_x = pred_x * torch.sqrt(sigma_norm)

        x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
        k_t_minus_05 = k_t
        l_t_minus_05 = l_t

        t_t_minus_05 = t_t


        # Predictor
        if self.use_ks:
            rand_k = torch.randn_like(k_T) if t > 1 else torch.zeros_like(k_T)
        else:
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
        rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

        adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
        step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
        std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   
        lattice_feats_t_minus_05 = k_t_minus_05 if self.use_ks else l_t_minus_05

        pred_lattice, pred_x, pred_t = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, lattice_feats_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, batch.spacegroup)

        pred_x = pred_x * torch.sqrt(sigma_norm)

        x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t
        x_t_minus_1 = x_t_minus_1 % 1

        if self.use_ks:
            k_t_minus_1 = c0 * (k_t_minus_05 - c1 * pred_lattice) + sigmas * rand_k if not self.keep_lattice else k_t
            if self.use_spacegroup and not self.keep_lattice:
                k_t_minus_1 = mask_ks(k_t_minus_1, ks_mask, ks_add)
            l_t_minus_1 = lattice_ks_to_matrix_torch(k_t_minus_1) if not self.keep_lattice else l_t
        else:
            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_lattice) + sigmas * rand_l if not self.keep_lattice else l_t
            k_t_minus_1 = k_t

        t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

        if symmops is not None:
            with torch.set_grad_enabled(True):
                #split_x = torch.split(x_T, tuple(batch.num_atoms.tolist()))
                x = x_t_minus_1.clone()
                x.requires_grad = True
                loss = torch.zeros(1, requires_grad=True, device=x.device)
                loss.requires_grad = True
                if len(symmops) > num_transformations:
                    symmop_is = np.random.choice(len(symmops), num_transformations, replace=False)
                else:
                    symmop_is = range(len(symmops))
                sampled_symmops = symmops[symmop_is]
                x_sym = apply_symmop_torch(sampled_symmops, x)
                cost, P, C, dist_matrix_sqr = sinkhorn(x, x_sym, t_t_minus_1)
                loss_i = (P * dist_matrix_sqr).sum()
                loss = loss + (1/2)*loss_i/len(symmop_is)

                sym_gradient = grad(loss, x, allow_unused=False)
                x_t_minus_1 = (x - sym_loss_alpha*sym_gradient[0]).detach() % 1

        traj[t - 1] = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_t_minus_1,
            'frac_coords' : x_t_minus_1 % 1.,
            'lattices' : l_t_minus_1,
            'ks': k_t_minus_1,
        }
        if self.use_spacegroup:
            traj[t - 1]['spacegroup'] = batch.spacegroup

    traj_stack = {
        'num_atoms' : batch.num_atoms,
        'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
        'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        'all_ks': torch.stack([traj[i]['ks'] for i in range(self.beta_scheduler.timesteps, -1, -1)])
    }

    return traj[0], traj_stack



def diffusion(loader, model, step_lr, use_sg=False, num_transformations=48, sym_loss_alpha=0.1):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    spacegroups = []
    for idx, batch in tqdm(enumerate(loader)):

        if torch.cuda.is_available():
            batch.cuda()
        spacegroup = batch.spacegroup.item()
        print(spacegroup)
        all_symmops = [symmop.affine_matrix for symmop in (SpaceGroup(sg_symbol_from_int_number(spacegroup)).symmetry_ops)]
        symmops_torch = torch.tensor(all_symmops, device=batch.spacegroup.device, dtype=torch.float)
    

        outputs, traj = sample_sym(model, batch, step_lr = step_lr, symmops=symmops_torch,
                                   num_transformations=num_transformations,
                                    sym_loss_alpha=sym_loss_alpha,
                                    silent=True)
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

    def __init__(self, dataset, total_num, use_num_atoms_per_sg=False):
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
    num_transformations = args.num_transformations
    sym_loss_alpha = args.sym_loss_alpha
    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')

    if args.load_dataset is not None:
        test_set = torch.load(args.load_dataset)
    else:
        test_set = SampleDataset(args.dataset, args.batch_size * args.num_batches_to_samples, args.num_atoms_per_sg)
    if args.save_dataset is not None:
        torch.save(test_set, args.save_dataset)

    test_loader = DataLoader(test_set, batch_size = args.batch_size)

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, spacegroups) = diffusion(test_loader, model, args.step_lr, use_sg, num_transformations, sym_loss_alpha)

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
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--label', default='')
    parser.add_argument('--num_atoms_per_sg', default=False, type=bool)
    parser.add_argument('--num_transformations', default=48, type=int)
    parser.add_argument('--sym_loss_alpha', default=0.1, type=float)
    parser.add_argument('--load_dataset', default=None, type=str)
    parser.add_argument('--save_dataset', default=None, type=str)
    args = parser.parse_args()


    main(args)
