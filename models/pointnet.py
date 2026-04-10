import torch
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, global_max_pool


def _fps_fallback(pos, batch, ratio):
    """Pure-PyTorch farthest point sampling fallback (no torch-cluster needed)."""
    device = pos.device
    unique_graphs = batch.unique()
    all_idx = []
    for g in unique_graphs:
        mask = batch == g
        g_pos = pos[mask]
        n = g_pos.shape[0]
        k = max(1, int(n * ratio))

        # Greedy FPS
        selected = torch.zeros(k, dtype=torch.long, device=device)
        distances = torch.full((n,), float('inf'), device=device)
        current = 0
        for i in range(k):
            selected[i] = current
            dist = (g_pos - g_pos[current]).pow(2).sum(dim=-1)
            distances = torch.minimum(distances, dist)
            current = distances.argmax().item()

        g_indices = mask.nonzero(as_tuple=True)[0]
        all_idx.append(g_indices[selected])

    return torch.cat(all_idx)


def _radius_fallback(pos_all, pos_query, r, batch_x, batch_y, max_num_neighbors):
    """Pure-PyTorch radius graph fallback (no torch-cluster needed)."""
    rows, cols = [], []
    unique_graphs = batch_y.unique()
    for g in unique_graphs:
        mask_x = batch_x == g
        mask_y = batch_y == g
        idx_x = mask_x.nonzero(as_tuple=True)[0]
        idx_y = mask_y.nonzero(as_tuple=True)[0]
        if len(idx_x) == 0 or len(idx_y) == 0:
            continue
        # Pairwise distances
        diff = pos_all[idx_x].unsqueeze(0) - pos_query[idx_y].unsqueeze(1)  # [Qg, Pg, 3]
        dist = diff.pow(2).sum(-1)  # [Qg, Pg]
        within = dist < r * r
        q_idx, p_idx = within.nonzero(as_tuple=True)
        if len(q_idx) > 0:
            rows.append(idx_y[q_idx])
            cols.append(idx_x[p_idx])
    if rows:
        return torch.cat(rows), torch.cat(cols)
    return torch.zeros(0, dtype=torch.long, device=pos_all.device), \
           torch.zeros(0, dtype=torch.long, device=pos_all.device)


class SAModule(nn.Module):
    """Set Abstraction: FPS + radius grouping + PointNetConv MLP.

    Uses PointNetConv to match the original checkpoint key structure
    (sa1.conv.local_nn.*), but replaces torch-cluster's fps/radius
    with pure-PyTorch fallbacks.
    """
    def __init__(self, ratio, r, nn_mlp, max_neighbors=32):
        super().__init__()
        self.ratio         = ratio
        self.r             = r
        self.max_neighbors = max_neighbors
        # PointNetConv stores nn_mlp as self.local_nn → checkpoint key = conv.local_nn.*
        self.conv          = PointNetConv(nn_mlp, add_self_loops=False)

    def forward(self, x, pos, batch):
        # Pure-PyTorch replacements for fps() and radius()
        idx        = _fps_fallback(pos, batch, self.ratio)
        row, col   = _radius_fallback(pos, pos[idx], self.r,
                                      batch_x=batch, batch_y=batch[idx],
                                      max_num_neighbors=self.max_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        x_out      = self.conv(x, (pos, pos[idx]), edge_index)
        return x_out, pos[idx], batch[idx]


class GlobalSAModule(nn.Module):
    """Global Set Abstraction: pool all points → one vector per graph."""
    def __init__(self, nn_mlp):
        super().__init__()
        self.nn = nn_mlp

    def forward(self, x, pos, batch):
        x_cat     = torch.cat([x, pos], dim=-1) if x is not None else pos
        x_out     = self.nn(x_cat)
        x_out     = global_max_pool(x_out, batch)
        pos_out   = pos.new_zeros((x_out.size(0), 3))
        batch_out = torch.arange(x_out.size(0), device=batch.device)
        return x_out, pos_out, batch_out


class DFM_PointNetPP(nn.Module):
    """
    Slim PointNet++ for Colab T4 (15 GB VRAM).
    """
    def __init__(self, in_dim=6, graph_stat_dim=11, dropout=0.3, n_reg=1, n_clf=5):
        super().__init__()

        # SA1 — fine local detail
        self.sa1 = SAModule(ratio=0.4, r=0.2, max_neighbors=32,
                            nn_mlp=MLP([in_dim + 3, 32, 64]))

        # SA2 — broader regions
        self.sa2 = SAModule(ratio=0.3, r=0.4, max_neighbors=32,
                            nn_mlp=MLP([64 + 3, 64, 128]))

        # SA3 — global summary
        self.sa3 = GlobalSAModule(nn_mlp=MLP([128 + 3, 256, 512]))

        self.graph_stat_dim = graph_stat_dim
        fused_dim           = 512 + graph_stat_dim   # 523

        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_reg)
        )
        self.clf_head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_clf)
        )

    def forward(self, data):
        pos, x, batch = data.pos, data.x, data.batch
        graph_stats   = data.graph_stats

        B           = batch.max().item() + 1
        graph_stats = graph_stats.view(B, self.graph_stat_dim)

        x1, pos1, b1 = self.sa1(x,  pos,  batch)
        x2, pos2, b2 = self.sa2(x1, pos1, b1)
        x3, _,    _  = self.sa3(x2, pos2, b2)   # [B, 512]

        x_fused = torch.cat([x3, graph_stats], dim=1)   # [B, 523]
        y_reg   = self.regression_head(x_fused)
        y_clf   = self.clf_head(x_fused)
        return y_reg, y_clf
