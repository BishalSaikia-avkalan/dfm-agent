"""
DiffusionNet model architecture for DFM analysis.
Extracted from comparative_DiffusionNet.ipynb.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings

warnings.filterwarnings("ignore", ".*Sparse invariant checks are implicitly disabled.*")

K_EIGENVECS = 128


def spectral_diffuse(x, evals, evecs, mass, t):
    """Diffuse features using the spectral decomposition of the Laplace-Beltrami operator."""
    x_mass = x * mass.unsqueeze(-1)
    x_spec = evecs.T @ x_mass
    decay = torch.exp(-t.unsqueeze(0) * evals.unsqueeze(-1).clamp(min=0))
    return evecs @ (x_spec * decay)


def build_block_sparse(batch_tensor, grad_rows, grad_cols, grad_vals,
                       grad_n, grad_nnz, device):
    """Build a block-diagonal sparse matrix from per-graph COO components."""
    N_total = batch_tensor.shape[0]
    B = grad_n.shape[0]

    node_off = torch.zeros(B, dtype=torch.long, device=device)
    node_off[1:] = grad_n[:-1].to(device).cumsum(0)

    nnz_off = torch.zeros(B, dtype=torch.long, device=device)
    nnz_off[1:] = grad_nnz[:-1].to(device).cumsum(0)

    all_rows, all_cols = [], []
    for g in range(B):
        ns = nnz_off[g].item()
        ne = ns + grad_nnz[g].item()
        off = node_off[g].item()
        all_rows.append(grad_rows[ns:ne].to(device) + off)
        all_cols.append(grad_cols[ns:ne].to(device) + off)

    rows_bd = torch.cat(all_rows)
    cols_bd = torch.cat(all_cols)

    return torch.sparse_coo_tensor(
        torch.stack([rows_bd, cols_bd]),
        grad_vals.to(device=device, dtype=torch.float32),
        (N_total, N_total),
        device=device,
        dtype=torch.float32
    ).coalesce()


def global_mean_pool(x, batch):
    """
    Simple global mean pooling: average node features per graph.
    Replaces torch_geometric.nn.global_mean_pool to avoid heavy dependency.
    """
    batch_size = int(batch.max().item()) + 1
    result = []
    for i in range(batch_size):
        mask = batch == i
        result.append(x[mask].mean(dim=0))
    return torch.stack(result)


class DiffusionBlock(nn.Module):
    """A single DiffusionNet block: spectral diffusion + gradient features + MLP."""

    def __init__(self, C, dropout=0.3):
        super().__init__()
        self.t_params = nn.Parameter(torch.ones(C) * 0.5)
        self.grad_lin = nn.Linear(C, C)
        self.mlp = nn.Sequential(
            nn.Linear(2 * C, C), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(C, C),
        )
        self.norm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, evals_batch, evecs, mass, grad_X_sparse, batch):
        K = evecs.shape[1]
        t = torch.abs(self.t_params)
        diff_out = torch.zeros_like(x)

        for g in batch.unique():
            mask = batch == g
            x_g = x[mask]
            ev_g = evecs[mask]
            ma_g = mass[mask]
            ev_val = evals_batch[g * K: (g + 1) * K]
            diff_out[mask] = spectral_diffuse(x_g, ev_val, ev_g, ma_g, t)

        # Gradient features via sparse matrix multiply
        grad_feat = torch.sparse.mm(grad_X_sparse, x)
        grad_feat = self.grad_lin(grad_feat)

        fused = self.mlp(torch.cat([diff_out, grad_feat], dim=-1))
        return self.norm(x + self.dropout(fused))


class DFM_DiffusionNet(nn.Module):
    """
    DiffusionNet for DFM metric prediction.

    Outputs:
        regression_head: predicted log1p(volume) [B, 1]
        clf_head: logits for [area, contour_count, contour_length, overhang, pass_fail] [B, 5]
    """

    def __init__(self, in_dim=7, C=128, n_blocks=4,
                 graph_stat_dim=11, dropout=0.3, n_reg=1, n_clf=5):
        super().__init__()
        self.C = C
        self.K = K_EIGENVECS
        self.graph_stat_dim = graph_stat_dim
        self.embed = nn.Sequential(nn.Linear(in_dim, C), nn.ReLU())
        self.blocks = nn.ModuleList(
            [DiffusionBlock(C, dropout) for _ in range(n_blocks)])
        fused_dim = C + graph_stat_dim
        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_reg))
        self.clf_head = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_clf))

    def _build_grad_X(self, data):
        grad_rows = data.grad_rows
        grad_cols = data.grad_cols
        grad_vals = data.grad_vals
        grad_n = data.grad_n
        batch = data.batch
        device = data.x.device

        B = grad_n.shape[0]
        node_off = torch.zeros(B, dtype=torch.long)
        node_off[1:] = grad_n[:-1].cumsum(0)

        grad_nnz = torch.zeros(B, dtype=torch.long)
        for g in range(B):
            lo = node_off[g].item()
            hi = lo + grad_n[g].item()
            grad_nnz[g] = ((grad_rows >= lo) & (grad_rows < hi)).sum()

        return build_block_sparse(
            batch, grad_rows, grad_cols, grad_vals,
            grad_n, grad_nnz, device
        )

    def forward(self, data):
        x = data.x
        evals = data.evals
        evecs = data.evecs
        mass = data.mass
        graph_stats = data.graph_stats
        batch = data.batch
        
        grad_X = self._build_grad_X(data)

        x = self.embed(x)
        for block in self.blocks:
            x = block(x, evals, evecs, mass, grad_X, batch)

        x_pool = global_mean_pool(x, batch)
        graph_stats = graph_stats.view(-1, self.graph_stat_dim)
        x_fused = torch.cat([x_pool, graph_stats], dim=1)
        return self.regression_head(x_fused), self.clf_head(x_fused)

    def forward_gradcam(self, data):
        """
        Same as forward() but returns the pre-pooling node features
        with requires_grad=True so GradCAM can backprop through them.
        """
        x = data.x
        evals = data.evals
        evecs = data.evecs
        mass = data.mass
        graph_stats = data.graph_stats
        batch = data.batch
        
        grad_X = self._build_grad_X(data)

        x = self.embed(x)
        for block in self.blocks:
            x = block(x, evals, evecs, mass, grad_X, batch)

        # Detach then re-enable grad — this is the GradCAM target tensor
        node_acts = x  
        
        x_pool = global_mean_pool(node_acts, batch)
        x_fused = torch.cat([x_pool, graph_stats.view(-1, self.graph_stat_dim)], dim=1)
        return self.regression_head(x_fused), self.clf_head(x_fused), node_acts
