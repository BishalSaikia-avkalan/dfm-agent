import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class DFM_GAT(nn.Module):
    """
    Multi-task GAT for DFM metric prediction.

    Input:
      x           : [N, 9]   node features
      edge_index  : [2, 2E]  undirected edges
      graph_stats : [B, 11]  graph-level mesh statistics

    Output:
      y_reg : [B, 1]  log volume
      y_clf : [B, 5]  binary constraint logits
    """
    def __init__(self,
                 in_dim        = 9,
                 hidden_dim    = 64,    # per head
                 heads         = 2,     # attention heads
                 num_layers    = 3,
                 dropout       = 0.3,
                 graph_stat_dim= 11,
                 n_reg         = 1,
                 n_clf         = 5):
        super().__init__()

        self.dropout    = dropout
        self.num_layers = num_layers
        self.convs      = nn.ModuleList()
        self.bns        = nn.ModuleList()

        # Layer dims: in_dim → hidden*heads → hidden*heads → ... → hidden (last layer concat=False)
        for i in range(num_layers):
            in_ch  = in_dim if i == 0 else hidden_dim * heads
            # Last layer: concat=False → output is hidden_dim (not hidden_dim*heads)
            is_last = (i == num_layers - 1)
            out_ch  = hidden_dim if is_last else hidden_dim
            self.convs.append(
                GATConv(in_ch, out_ch, heads=heads,
                        concat=(not is_last), dropout=dropout)
            )
            out_feat = out_ch if is_last else out_ch * heads
            self.bns.append(nn.BatchNorm1d(out_feat))

        # After pooling: hidden_dim + graph_stat_dim
        fused_dim = hidden_dim + graph_stat_dim  # 64 + 11 = 75

        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_reg)
        )

        self.clf_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_clf)
        )

    def forward(self, data):
        # Use gat_x [N, 9] if available, fallback to x
        x = data.gat_x if hasattr(data, 'gat_x') else data.x
        edge_index, batch = data.edge_index, data.batch
        graph_stats = data.graph_stats.view(-1, 11)  # [B, 11] — reshape from flat batched tensor

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)      # GAT message passing
            x = bn(x)
            x = F.elu(x)                 # ELU works better than ReLU for GAT
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        x_pool = global_mean_pool(x, batch)  # [B, hidden_dim]

        # Inject graph-level mesh stats
        x_fused = torch.cat([x_pool, graph_stats], dim=1)  # [B, hidden_dim + 11]

        y_reg = self.regression_head(x_fused)   # [B, 1]
        y_clf = self.clf_head(x_fused)           # [B, 5]

        return y_reg, y_clf
