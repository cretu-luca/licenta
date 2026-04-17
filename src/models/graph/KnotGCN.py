import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, global_mean_pool


class KnotGCN(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_layers: int):
        super(KnotGCN, self).__init__()

        self.embed = nn.Embedding(2, hidden_dim)
        self.convs = ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = self.embed(data.x.squeeze(-1))
        
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, data.edge_index))
        x = self.convs[-1](x, data.edge_index)
        
        x = global_mean_pool(x, data.batch)

        return self.readout(x)