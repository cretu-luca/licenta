import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class KnotGAT(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_layers: int):
        super(KnotGAT, self).__init__()

        self.embed = nn.Embedding(2, hidden_dim)
        self.gats = nn.ModuleList(
            [GATConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze(-1))

        for gat in self.gats[:-1]:
            x = F.relu(gat(x, data.edge_index))
        x = self.gats[-1](x, data.edge_index)

        x = global_mean_pool(x, data.batch)
        return self.readout(x)