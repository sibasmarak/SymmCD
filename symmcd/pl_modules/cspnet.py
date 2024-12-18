import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from einops import rearrange, repeat

from symmcd.common.data_utils import lattice_params_to_matrix_torch, get_pbc_distances, radius_graph_pbc, frac_to_cart_coords, repeat_blocks

from symmcd.pl_modules.model import build_mlp

from torch_geometric.nn.conv.transformer_conv import TransformerConv

MAX_ATOMIC_NUM=94
N_AXES = 15
N_SS = 13
SITE_SYMM_DIM = N_AXES * N_SS

class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies = 10, n_space = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class CSPLayer(nn.Module):
    """ Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ln=False,
        ip=True,
        use_ks=False
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        self.use_ks = use_ks
        self.lattice_dim = 9 if self.ip else 6
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.lattice_dim + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def edge_model(self, node_features, frac_coords, lattice_feats, edge_index, edge2graph, frac_diff = None):

        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_feats = lattice_feats @ lattice_feats.transpose(-1, -2)
        lattice_feats_flatten = lattice_feats.view(-1, self.lattice_dim)
        lattice_feats_flatten_edges = lattice_feats_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_feats_flatten_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(edge_features, edge_index[0], dim = 0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim = 1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattice_feats, edge_index, edge2graph, frac_diff = None):

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattice_feats, edge_index, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim=128, heads=8, ln=True, concat=True, dis_emb=None, act_fn=nn.SiLU()):
        super(TransformerLayer, self).__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.lattice_dim = 6
        self.ln = ln
        self.conv = TransformerConv(hidden_dim + self.lattice_dim + self.dis_dim, out_channels=hidden_dim, heads=heads, concat=concat, edge_dim=dis_emb.dim)
        self.act_fn = act_fn
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, frac_coords, lattice_feats, edge_index, node2graph, frac_diff = None):
        edge_features = self.dis_emb(frac_diff)
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        lattice_feats_flatten = lattice_feats.view(-1, self.lattice_dim)
        lattice_feats_flatten_nodes = lattice_feats_flatten[node2graph]
        node_features = torch.cat([node_features, lattice_feats_flatten_nodes, frac_coords], dim=1)
        node_output = self.act_fn(self.conv(node_features, edge_index, edge_features))
        return node_input + node_output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim=128, heads=8, ln=True, concat=True, dis_emb=None, act_fn=nn.SiLU()):
        super(TransformerLayer, self).__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.lattice_dim = 6
        self.ln = ln
        self.conv = TransformerConv(hidden_dim + self.lattice_dim + self.dis_dim, out_channels=hidden_dim, heads=heads, concat=concat, edge_dim=dis_emb.dim)
        self.act_fn = act_fn
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, frac_coords, lattice_feats, edge_index, node2graph, frac_diff = None):
        edge_features = self.dis_emb(frac_diff)
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        lattice_feats_flatten = lattice_feats.reshape(-1, self.lattice_dim)
        lattice_feats_flatten_nodes = lattice_feats_flatten[node2graph]
        node_features = torch.cat([node_features, lattice_feats_flatten_nodes, frac_coords], dim=1)
        node_output = self.act_fn(self.conv(node_features, edge_index, edge_features))
        return node_input + node_output

class CSPNet(nn.Module):

    def __init__(
        self,
        network='gnn',
        hidden_dim = 128,
        latent_dim = 256,
        time_dim = 256,
        num_layers = 4,
        max_atoms = 100,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 10,
        edge_style = 'fc',
        cutoff = 6.0,
        max_neighbors = 20,
        ln = False,
        ip = True,
        use_ks = False,
        use_gt_frac_coords=False,
        use_site_symm=False,
        smooth = False,
        pred_type = False,
        pred_site_symm_type = False,
        site_symm_matrix_embed=False,
        mask_token=False,
        heads=8
    ):
        super(CSPNet, self).__init__()

        self.ip = ip
        self.smooth = smooth
        self.use_ks = use_ks
        self.use_gt_frac_coords = use_gt_frac_coords
        self.use_site_symm = use_site_symm
        self.mask_token = 1 if mask_token else 0
        self.heads = heads
        assert self.use_ks != self.ip # Cannot use both inner product representation and k representation
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.site_symm_matrix_embed = site_symm_matrix_embed
        self.n_ss = N_SS + self.mask_token
        self.site_symm_dim = N_AXES*self.n_ss
        if self.site_symm_matrix_embed:
            self.ss_matrix_embed_dim = 4
            self.axis_embedding = nn.Linear(self.n_ss, self.ss_matrix_embed_dim)
            self.site_symm_embedding = nn.Linear(N_AXES*self.ss_matrix_embed_dim, latent_dim)
        else:
            self.site_symm_embedding = nn.Linear(self.site_symm_dim, latent_dim)

        lattice_dim = 9 if self.ip else 6
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies = num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        if self.use_gt_frac_coords and self.dis_emb:
            frac_coord_dim = self.dis_emb.dim
        else:
            frac_coord_dim = 3 if self.use_gt_frac_coords else 0
        self.atom_latent_emb = nn.Linear(hidden_dim + time_dim + (latent_dim if self.use_site_symm else 0) + frac_coord_dim, hidden_dim)
        for i in range(0, num_layers):
            if network == 'gnn':
                self.add_module(
                    "csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip, use_ks=use_ks)
                )  
            elif network == 'transformer':
                self.add_module(
                    "csp_layer_%d" % i, TransformerLayer(hidden_dim, heads=self.heads, concat=False, ln=ln, dis_emb=self.dis_emb, act_fn=self.act_fn)
                )   
        self.network = network       
        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias = False)
        self.lattice_out = nn.Linear(hidden_dim, lattice_dim, bias = False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.edge_style = edge_style
        self.pred_type = pred_type
        self.pred_site_symm_type = pred_site_symm_type
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, max_atoms)
        if self.pred_site_symm_type:
            self.site_symm_out = nn.Linear(hidden_dim, self.site_symm_dim)
            if self.site_symm_matrix_embed:
                self.ss_matrix_out_dim = 8
                self.axis_wise_out = nn.Linear(hidden_dim, N_AXES*self.ss_matrix_out_dim)
                self.symm_wise_out = nn.Linear(hidden_dim, self.n_ss*self.ss_matrix_out_dim)
            else:
                self.site_symm_out = nn.Linear(hidden_dim, self.site_symm_dim)

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_vector_new,
        )

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):

        if self.edge_style == 'fc':
            lis = [torch.ones(n,n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]])
        elif self.edge_style == 'knn':
            lattice_nodes = lattices[node2graph]
            cart_coords = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)
            
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords, None, None, num_atoms, self.cutoff, self.max_neighbors,
                device=num_atoms.device, lattices=lattices)

            j_index, i_index = edge_index
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(edge_index, to_jimages, num_bonds, distance_vectors)

            return edge_index_new, -edge_vector_new
            

    def forward(self, t, atom_types, frac_coords, lattice_feats, lattices, num_atoms, node2graph, site_symm_probs=None):

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        if self.use_site_symm:
            if self.site_symm_matrix_embed:
                axis_embeddings = self.axis_embedding(site_symm_probs.reshape(-1, N_AXES, self.n_ss))
                site_symm_embeddings = self.site_symm_embedding(axis_embeddings.reshape(-1, N_AXES*self.ss_matrix_embed_dim))
            else:
                site_symm_embeddings = self.site_symm_embedding(site_symm_probs)
            node_features = torch.cat([node_features, site_symm_embeddings], dim=1)
        if self.use_gt_frac_coords and self.dis_emb:
            node_features = torch.cat([node_features, self.dis_emb(frac_coords)], dim=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            if self.network == 'transformer':
                node_features = self._modules["csp_layer_%d" % i](node_features, frac_coords, lattice_feats, edges, node2graph, frac_diff = frac_diff)
            elif self.network == 'gnn':
                node_features = self._modules["csp_layer_%d" % i](node_features, frac_coords, lattice_feats, edges, edge2graph, frac_diff = frac_diff)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)

        graph_features = scatter(node_features, node2graph, dim = 0, reduce = 'mean')
        lattice_out = self.lattice_out(graph_features)
        if not self.use_ks:
            lattice_out = lattice_out.view(-1, 3, 3)
            if self.ip:
                lattice_out = torch.einsum('bij,bjk->bik', lattice_out, lattices)
        if self.pred_type and self.pred_site_symm_type:
            type_out = self.type_out(node_features)
            if self.site_symm_matrix_embed:
                axis_embs = self.axis_wise_out(node_features).reshape(-1, N_AXES, self.ss_matrix_out_dim)
                symm_embs = self.symm_wise_out(node_features).reshape(-1, self.n_ss, self.ss_matrix_out_dim)
                site_symm_out = (axis_embs @ symm_embs.swapaxes(-1, -2)).flatten()
            else:
                site_symm_out = self.site_symm_out(node_features)
            return lattice_out, coord_out, type_out, site_symm_out.reshape(-1, self.site_symm_dim)
        if self.pred_type and not self.pred_site_symm_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out
        if not self.pred_type and self.pred_site_symm_type:
            site_symm_out = self.site_symm_out(node_features)
            return lattice_out, coord_out, site_symm_out.reshape(-1, self.site_symm_dim)

        return lattice_out, coord_out

