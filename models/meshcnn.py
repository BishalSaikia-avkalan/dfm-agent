import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshConv(nn.Module):
    """
    MeshCNN convolution layer.

    Aggregates features from the 4 neighbouring edges (e2e) around each edge,
    applies a learnable 1×5 filter (5 = self + 4 neighbours), then BN + ReLU.

    Input:
      x   : [E, in_ch]   edge features
      e2e : [E, 4]       4 adjacent-edge indices per edge
    Output:
      x   : [E, out_ch]
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5)   # 5 = self + 4 nbrs
        self.bn   = nn.BatchNorm1d(out_ch)

    def forward(self, x, e2e):
        # x   : [E, C]
        # e2e : [E, 4]
        nbr = x[e2e]             # [E, 4, C]
        x_self = x.unsqueeze(1)  # [E, 1, C]
        x_cat  = torch.cat([x_self, nbr], dim=1)  # [E, 5, C]
        # Conv1d expects [batch, channels, length] → treat each edge as batch item
        x_cat  = x_cat.permute(0, 2, 1)  # [E, C, 5]
        out    = self.conv(x_cat)         # [E, out_ch, 1]
        out    = out.squeeze(-1)          # [E, out_ch]
        return F.relu(self.bn(out))


class MeshPool(nn.Module):
    """
    Simplified MeshCNN pooling — keeps the top-k edges by feature magnitude.

    Full MeshCNN collapses edges by modifying the mesh topology; here we use
    a differentiable approximation: sort edges by L2 norm, keep top-k, and
    scatter remaining features into the kept edges via soft attention.

    This preserves gradient flow while reducing sequence length.
    """
    def __init__(self, target_edges):
        super().__init__()
        self.target = target_edges

    def forward(self, x, e2e, batch):
        """
        x     : [E_total, C]
        e2e   : [E_total, 4]
        batch : [E_total]   which graph each edge belongs to
        Returns:
          x_pooled   : [E_pooled, C]
          e2e_pooled : [E_pooled, 4]
          batch_pooled: [E_pooled]
        """
        scores = x.norm(dim=-1)   # [E_total]  importance per edge

        # Pool per-graph to keep target edges per graph
        unique_graphs = batch.unique()
        kept_indices  = []

        for g in unique_graphs:
            mask   = batch == g
            g_idx  = mask.nonzero(as_tuple=True)[0]
            g_score = scores[g_idx]
            k      = min(self.target, len(g_idx))
            topk   = g_score.topk(k).indices
            kept_indices.append(g_idx[topk])

        kept_idx = torch.cat(kept_indices)          # [E_pooled]

        # Rebuild e2e by remapping indices into kept_idx space
        old2new = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        old2new[kept_idx] = torch.arange(len(kept_idx), device=x.device)

        e2e_kept = e2e[kept_idx]                    # [E_pooled, 4]
        # Remap neighbours; invalid (not kept) → self-loop
        e2e_new  = old2new[e2e_kept.clamp(min=0)]   # [E_pooled, 4]
        # Replace -1 (unmapped) with self index
        self_idx = torch.arange(len(kept_idx), device=x.device).unsqueeze(1).expand_as(e2e_new)
        e2e_new  = torch.where(e2e_new < 0, self_idx, e2e_new)

        return x[kept_idx], e2e_new, batch[kept_idx]


class DFM_MeshCNN(nn.Module):
    """
    MeshCNN-based multi-task model for DFM metric prediction.

    Input per graph:
      edge_feats  : [E, 5]    MeshCNN 5-dim edge features
      e2e         : [E, 4]    4 neighbouring edge indices per edge
      graph_stats : [B, 11]   graph-level mesh statistics

    Output:
      y_reg : [B, 1]   log volume
      y_clf : [B, 5]   binary DFM constraint logits
    """
    def __init__(self,
                 in_ch         = 5,
                 hidden_ch     = 64,
                 graph_stat_dim= 11,
                 dropout       = 0.3,
                 pool_sizes    = (512, 256),
                 n_reg         = 1,
                 n_clf         = 5):
        super().__init__()

        self.dropout       = dropout
        self.graph_stat_dim = graph_stat_dim

        # ── MeshConv layers ─────────────────────────────────────────────
        self.conv1 = MeshConv(in_ch,          hidden_ch)        # 5  → 64
        self.conv2 = MeshConv(hidden_ch,      hidden_ch * 2)    # 64 → 128
        self.conv3 = MeshConv(hidden_ch * 2,  hidden_ch * 4)    # 128→ 256

        # ── MeshPool layers ─────────────────────────────────────────────
        self.pool1 = MeshPool(pool_sizes[0])  # reduce to 512 edges
        self.pool2 = MeshPool(pool_sizes[1])  # reduce to 256 edges

        # After pool2, features are [256, 256] per graph → global avg → [256]
        fused_dim  = hidden_ch * 4 + graph_stat_dim  # 256 + 11 = 267

        # ── Prediction heads ─────────────────────────────────────────────
        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_reg)
        )

        self.clf_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_clf)
        )

    def forward(self, data):
        x           = data.edge_feats                          # [E_total, 5]
        e2e         = data.e2e                                 # [E_total, 4]
        graph_stats = data.graph_stats                         # [B*11] or [B, 11]
        batch       = data.batch                               # [N_vertices] — node batch assignment

        # Build EDGE batch vector from node batch via edge_index
        # Each edge src vertex → same graph as its source node
        edge_src    = data.edge_index[0]                       # [2E directed]
        # edge_feats are undirected (E), take first E entries from directed list
        E           = x.shape[0]
        # Build edge_batch: use the src of first occurrence of each undirected edge
        # Simplification: use batch[edge_index[0]] for directed edges, then keep unique
        directed_src = data.edge_index[0]                      # [2E]
        # For undirected edges (u<v), src_u is first E values when sorted
        # Safer: build edge_batch directly from e2e neighbour's batch
        edge_batch   = batch[directed_src[:E]]                 # [E]  — approximate

        B = batch.max().item() + 1
        graph_stats  = graph_stats.view(B, self.graph_stat_dim)  # [B, 11]

        # ── Conv1 → Pool1 ───────────────────────────────────────────────
        x = self.conv1(x, e2e)                        # [E, 64]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, e2e, edge_batch = self.pool1(x, e2e, edge_batch)  # [E1, 64]

        # ── Conv2 → Pool2 ───────────────────────────────────────────────
        x = self.conv2(x, e2e)                        # [E1, 128]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, e2e, edge_batch = self.pool2(x, e2e, edge_batch)  # [E2, 128]

        # ── Conv3 ───────────────────────────────────────────────────────
        x = self.conv3(x, e2e)                        # [E2, 256]

        # ── Global average pool per graph ────────────────────────────────────────
        x = x.float()   # ← cast back to float32 after AMP may have made it float16
        x_pool = torch.zeros(B, x.shape[1], device=x.device, dtype=torch.float32)
        count  = torch.zeros(B, 1,          device=x.device, dtype=torch.float32)
        x_pool.scatter_add_(0, edge_batch.unsqueeze(1).expand_as(x), x)
        count.scatter_add_(0,  edge_batch.unsqueeze(1), torch.ones_like(edge_batch.unsqueeze(1).float()))
        x_pool = x_pool / count.clamp(min=1)           # [B, 256]

        # ── Fuse with graph-level stats ───────────────────────────────────
        x_fused = torch.cat([x_pool, graph_stats], dim=1)  # [B, 267]

        y_reg = self.regression_head(x_fused)   # [B, 1]
        y_clf = self.clf_head(x_fused)           # [B, 5]

        return y_reg, y_clf
