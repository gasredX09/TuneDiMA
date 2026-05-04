import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


def make_edge_mlp(edge_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(edge_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class LigandGNN(nn.Module):
    def __init__(self, in_dim=16, edge_dim=10, hidden_dim=128, num_layers=3, dropout=0.5, mol_desc_dim=17):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GINEConv(make_edge_mlp(edge_dim, hidden_dim)))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(make_edge_mlp(edge_dim, hidden_dim)))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim + mol_desc_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index, edge_attr)))
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        if hasattr(data, "mol_desc"):
            desc = data.mol_desc.to(x.dtype)
            x = torch.cat([x, desc], dim=1)
        return self.fc(x).squeeze(-1)
