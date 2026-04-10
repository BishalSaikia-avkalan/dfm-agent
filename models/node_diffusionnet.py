import torch
import torch.nn as nn
from models.diffusionnet import DiffusionBlock, spectral_diffuse, build_block_sparse

class NodeLevel_DFM_DiffusionNet(nn.Module):
    """
    Node-level DiffusionNet.
    Backbone identical to DFM_DiffusionNet.
    Output: [N, 5] per-vertex constraint logits instead of [B, 5].
    """
    def __init__(self, in_dim=7, C=128, n_blocks=4,
                 graph_stat_dim=11, dropout=0.3, n_clf=5):
        super().__init__()
        self.C              = C
        self.graph_stat_dim = graph_stat_dim
        self.embed          = nn.Sequential(nn.Linear(in_dim, C), nn.ReLU())
        self.blocks         = nn.ModuleList(
            [DiffusionBlock(C, dropout) for _ in range(n_blocks)])

        fused_dim = C + graph_stat_dim

        self.node_clf_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_clf)
        )

    def _build_grad_X(self, data):
        grad_rows = data.grad_rows
        grad_cols = data.grad_cols
        grad_vals = data.grad_vals
        grad_n    = data.grad_n
        batch     = data.batch
        device    = data.x.device

        B         = grad_n.shape[0]
        node_off  = torch.zeros(B, dtype=torch.long, device=device)
        node_off[1:] = grad_n[:-1].to(device).cumsum(0)
        grad_nnz  = torch.zeros(B, dtype=torch.long, device=device)
        for g in range(B):
            lo = node_off[g].item()
            hi = lo + grad_n[g].item()
            grad_nnz[g] = ((grad_rows >= lo) & (grad_rows < hi)).sum()
            
        return build_block_sparse(
            batch, grad_rows, grad_cols, grad_vals,
            grad_n, grad_nnz, device
        )

    def forward(self, data):
        x           = data.x           
        evals       = data.evals       
        evecs       = data.evecs       
        mass        = data.mass        
        graph_stats = data.graph_stats
        batch       = data.batch       

        grad_X = self._build_grad_X(data)

        x = self.embed(x)
        for block in self.blocks:
            x = block(x, evals, evecs, mass, grad_X, batch)

        B = batch.max().item() + 1
        graph_stats = graph_stats.view(B, self.graph_stat_dim)
        expanded_gs = graph_stats[batch]

        x_fused = torch.cat([x, expanded_gs], dim=1)

        return self.node_clf_head(x_fused)
