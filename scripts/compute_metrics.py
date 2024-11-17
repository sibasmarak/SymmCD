from collections import Counter
import argparse
import os
import json
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
import pandas as pd
import torch

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.composition.composite import ElementProperty

import sys
sys.path.append('.')
sys.path.append('..')
import os
myDir = os.getcwd()
sys.path.append(myDir)

from symmcd.common.data_utils import build_crystal, get_symmetry_info
from pyxtal import pyxtal

import pickle

warnings.simplefilter("ignore")
from scripts.eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

Crystal_Tol = 0.1
CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CrystalNNFP_full = SiteStatsFingerprint(
    CrystalNNFingerprint.from_preset('ops'),
    stats=('mean', 'maximum', 'minimum', 'std_dev'))
CrystalNNFP_plus = SiteStatsFingerprint(
    CrystalNNFingerprint.from_preset('ops'),
    stats=('mean', 'maximum'))

CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


def club_consecutive_elements(arr):
    # Initialize variables
    output = []
    start_index = 0

    # Iterate over the array
    for i in range(1, len(arr)):
        if arr[i] != arr[start_index]:
            # Append the element, its start index, and count to the output list
            output.append((arr[start_index], start_index, i - start_index))
            start_index = i

    # Handle the last group
    output.append((arr[start_index], start_index, len(arr) - start_index))

    return output

class Crystal(object):

    def __init__(self, crys_array_dict, filter=False, full_fingerprint=False):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.full_fingerprint = full_fingerprint
        if 'spacegroups' in crys_array_dict:
            self.spacegroup = crys_array_dict['spacegroups']
        else:
            self.spacegroup = None
        
        # check for NaN values 
        if np.isnan(self.lengths).any() or np.isinf(self.lengths).any():
            self.lengths = np.array([1, 1, 1]) * 100
            crys_array_dict['lengths'] = self.lengths
            
            self.constructed = False
            self.invalid_reason = 'nan_value'
        
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            # this implies the distribution over atom_types is passed instead of the atom_types
            # for perov that would mean a numpy array of (5, 100) instead of (5,)
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)
        
        # NOTE: post-processing hack to see what will happen if we remove the atoms that are too close (and have the same types)
        # find the distance matrix between atoms of same type -> if less than cutoff merge them
        if filter:
            self.distance_cutoff = 0.5 # set to 0.5 because structural validity cutoff is 0.5
            updated_frac_coords, updated_atom_types = [], []
            
            for atm_type, atm_index, counts in club_consecutive_elements(self.atom_types):
                atm_type_frac_coords = self.frac_coords[atm_index:atm_index+counts] # all frac coords for this atom type
                upd_frac_coords, upd_atom_types = [], [] # updated frac coords and atom types for this atom type
                
                for frac in atm_type_frac_coords:
                    add_flag = True

                    if len(upd_frac_coords) > 0:
                        distances = np.linalg.norm(
                                    np.minimum(
                                        (frac - np.array(upd_frac_coords))%1. * self.lengths, 
                                        (np.array(upd_frac_coords) - frac)%1. * self.lengths
                                        ), axis=-1)
                        
                        if np.min(distances) <= self.distance_cutoff:
                            add_flag = False
                            break

                    if add_flag:
                        upd_frac_coords.append(frac)
                        upd_atom_types.append(atm_type)
                        
                updated_frac_coords.extend(upd_frac_coords)
                updated_atom_types.extend(upd_atom_types)
                    
            self.frac_coords = np.array(updated_frac_coords)
            self.atom_types = np.array(updated_atom_types)
            self.dict['atom_types'] = self.atom_types
            self.dict['frac_coords'] = self.frac_coords
            self.dict['num_atoms'] = len(self.atom_types)
        
        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()
        self.get_symmetry()


    def get_structure(self):
        if (1 > self.atom_types).any() or (self.atom_types > 94).any():
            self.constructed = False
            self.invalid_reason = f"{self.atom_types=} are not with range"
        if len(self.frac_coords) > 30:
            self.constructed = False
            self.invalid_reason = 'too_many_atoms'
        if len(self.atom_types) == 0:
            self.constructed = False
            self.invalid_reason = 'empty'
        elif min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        elif np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'
        elif np.isinf(self.lengths).any() or np.isinf(self.angles).any() or  np.isinf(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'inf_value'
        elif (self.lengths > 1000).any():
            self.constructed = False
            self.invalid_reason = 'bad_value'
        elif (self.angles >= 180).any() or (self.angles <= 0).any():
            self.constructed = False
            self.invalid_reason = 'bad_value'
        elif (self.frac_coords > 1).any() or (self.frac_coords < 0).any():
            self.constructed = False
            self.invalid_reason = 'bad_value'
        elif self.lengths.min() < 1:
            self.constructed = False
            self.invalid_reason = 'bad_value'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = 'unrealistically_small_lattice'

            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        if len(elem_counter) == 0:
            self.elems = ()
            self.comps = ()
            return
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        if self.constructed:
            if len(self.elems) == 0:
                self.comp_valid = False
            else:
                self.comp_valid = smact_validity(self.elems, self.comps)
            self.struct_valid = structure_validity(self.structure)
        else:
            self.comp_valid = False
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        if len(self.atom_types) == 0:
            self.struct_fp = None
            self.comp_fp = None
            return
        if not self.constructed:
            self.struct_fp = None
            self.comp_fp = None
            return 
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        if self.full_fingerprint:
            try:
                struct_fp = CrystalNNFP_full.featurize(self.structure)
                self.struct_fp = np.array(struct_fp)
            except Exception:
                self.valid = False
                self.comp_fp = None
                self.struct_fp = None
        else:
            try:
                site_fps = [CrystalNNFP.featurize(
                    self.structure, i) for i in range(len(self.structure))]
            except Exception:
                # counts crystal as invalid if fingerprint cannot be constructed.
                self.valid = False
                self.comp_fp = None
                self.struct_fp = None
                return
            self.struct_fp = np.array(site_fps).mean(axis=0)

    def get_symmetry(self):
        if self.constructed:
            try:
                spga = SpacegroupAnalyzer(self.structure, symprec=Crystal_Tol)
                self.real_spacegroup = spga.get_space_group_number()
            except Exception:
                self.real_spacegroup = 1
            if self.real_spacegroup is None:
                # Default to setting to 1
                self.real_spacegroup = 1
        else:
            self.real_spacegroup = None
        self.spacegroup_match = self.real_spacegroup == self.spacegroup

class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None
        validity = [c1.valid and c2.valid for c1,c2 in zip(self.preds, self.gts)]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}     

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class RecEvalBatch(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))
            
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}    

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics



class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None, gt_prop_eval_path=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name
        self.gt_prop_eval_path = gt_prop_eval_path

        valid_crys = [c for c in pred_crys if c.valid]
        if n_samples == 0:
            # use all valid crystals instead
            self.valid_samples = valid_crys
        elif len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            if self.gt_prop_eval_path is not None and os.path.exists(self.gt_prop_eval_path):
                gt_props = torch.load(self.gt_prop_eval_path)
            else:
                gt_props = prop_model_eval(self.eval_model_name, [
                                           c.dict for c in self.gt_crys])
                torch.save(gt_props, self.gt_prop_eval_path)
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_spacegroup_wdist(self):
        pred_spacegroup = [c.real_spacegroup for c in self.valid_samples]
        gt_spacegroup = [c.spacegroup for c in self.gt_crys]
        wdist_spacegroup = wasserstein_distance(pred_spacegroup, gt_spacegroup)
        return {'wdist_spacegroup': wdist_spacegroup}

    def get_spacegroup_match(self):
        spacegroup_match = np.array([c.spacegroup_match for c in self.valid_samples]).mean()
        return {'spacegroup_match': spacegroup_match}


    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        if len(self.valid_samples) == 0:
            print("No valid crystals generated")
            return metrics
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_prop_wdist())
        metrics.update(self.get_num_elem_wdist())
        print(metrics)
        metrics.update(self.get_coverage())
        metrics.update(self.get_spacegroup_wdist())
        metrics.update(self.get_spacegroup_match())
        return metrics


def get_file_paths(root_path, task, label='', suffix='pt'):
    if label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        kwargs = dict()
        if 'spacegroups' in data and 'site_symmetries' in data:
            kwargs['spacegroups'] = data['spacegroups']
            kwargs['site_symmetries'] = data['site_symmetries']
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'],
            **kwargs)        
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    spacegroup = structure.get_space_group_info()[1]
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles),
        'spacegroups': spacegroup
    }
    return Crystal(crys_array_dict)

def get_gt_crys_ori_conventional(cif):
    crystal = build_crystal(cif)
    crystal, sym_info, dummy_repr_ind, dummy_origin_ind, identifier = get_symmetry_info(crystal, tol=0.01, num_repr=0)
    spacegroup = sym_info['spacegroup']
    lattice = crystal.lattice
    crys_array_dict = {
        'frac_coords':crystal.frac_coords,
        'atom_types':np.array([_.Z for _ in crystal.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles),
        'spacegroups': spacegroup
    }
    return Crystal(crys_array_dict) 

def main(args):
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    if 'gen' in args.tasks:

        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path, batch_idx = -2)
        if args.gt_crys_file != '':
            if os.path.exists(args.gt_crys_file):
                print("loading gt_crys")
                gt_crys = torch.load(args.gt_crys_file)    
            else:
                print("Reading gt_crys csv")
                csv = pd.read_csv(args.gt_file)
                gt_crys = p_map(get_gt_crys_ori_conventional, csv['cif'])
                torch.save(gt_crys, args.gt_crys_file)
        elif args.gt_file != '':
            print("Reading gt_crys csv")
            csv = pd.read_csv(args.gt_file)
            if args.conventional:
                gt_crys = p_map(get_gt_crys_ori_conventional, csv['cif'])
            else:
                gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            # always ground gt_file is provided
            # if not provided then only use the reconstruction path to load true crystals
            recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)
        if os.path.exists(args.root_path + f'/gen_crys_{args.label}.pt'):
            gen_crys = torch.load(args.root_path + f'/gen_crys_{args.label}.pt')
        else:
            gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
            torch.save(gen_crys, args.root_path + f'/gen_crys_{args.label}.pt')

        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name, n_samples=args.n_samples,
            gt_prop_eval_path=cfg.data.datamodule.datasets.test[0].gt_prop_eval_path)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)


    else:

        recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
        batch_idx = -1 if args.multi_eval else 0
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path, batch_idx = batch_idx)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

        if not args.multi_eval:
            pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
        else:
            pred_crys = []
            for i in range(len(crys_array_list)):
                print(f"Processing batch {i}")
                pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i]))   


        if 'csp' in args.tasks: 

            if args.multi_eval:
                rec_evaluator = RecEvalBatch(pred_crys, gt_crys)
            else:
                rec_evaluator = RecEval(pred_crys, gt_crys)

            recon_metrics = rec_evaluator.get_metrics()

            all_metrics.update(recon_metrics)

   

    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['csp', 'gen'])
    parser.add_argument('--gt_file',default='')
    parser.add_argument('--gt_crys_file',default='')
    parser.add_argument('--multi_eval',action='store_true')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--conventional', type=bool, default=False,
                        help='whether to use the conventional lattice instead of the primitive lattice')
    args = parser.parse_args()
    main(args)
