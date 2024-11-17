import torch
from pyxtal.symmetry import Group

cached_sg_to_site_symm, cached_sg_to_atoms = [], []

for spacegroup in range(1, 231):
    wp_to_site_symm = dict()
    wp_to_atoms = dict()
    for wp in Group(spacegroup).Wyckoff_positions:
        # get number of atoms in the Wyckoff position
        wp.get_site_symmetry()
        wp_to_atoms[wp] = wp.multiplicity
        wp_to_site_symm[wp] = wp.get_site_symmetry_object().to_one_hot()
    
    cached_sg_to_site_symm.append(torch.stack([torch.from_numpy(x.flatten()) for x in wp_to_site_symm.values()]))
    cached_sg_to_atoms.append(torch.tensor(list(wp_to_atoms.values())))
    
torch.save(cached_sg_to_site_symm, '/home/mila/s/siba-smarak.panigrahi/DiffCSP/diffcsp/pl_modules/cached_sg_to_site_symm.pt')
torch.save(cached_sg_to_atoms, '/home/mila/s/siba-smarak.panigrahi/DiffCSP/diffcsp/pl_modules/cached_sg_to_atoms.pt')