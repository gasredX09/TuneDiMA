import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


def make_node_mlp(hidden_dim):
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class LigandPocketNet(nn.Module):
    def __init__(self, lig_in=16, poc_in=25, int_in=25, edge_dim=10, int_edge_dim=3, hidden_dim=128, num_layers=3, dropout=0.5, mol_desc_dim=17):
        super().__init__()
        self.lig_input_proj = nn.Linear(lig_in, hidden_dim)
        self.poc_input_proj = nn.Linear(poc_in, hidden_dim)
        self.int_input_proj = nn.Linear(int_in, hidden_dim)
        self.lig_convs = nn.ModuleList([GINEConv(make_node_mlp(hidden_dim), edge_dim=edge_dim)])
        self.lig_bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        for _ in range(num_layers - 1):
            self.lig_convs.append(GINEConv(make_node_mlp(hidden_dim), edge_dim=edge_dim))
            self.lig_bns.append(nn.BatchNorm1d(hidden_dim))

        self.poc_convs = nn.ModuleList([GINEConv(make_node_mlp(hidden_dim), edge_dim=1)])
        self.poc_bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        for _ in range(num_layers - 1):
            self.poc_convs.append(GINEConv(make_node_mlp(hidden_dim), edge_dim=1))
            self.poc_bns.append(nn.BatchNorm1d(hidden_dim))

        self.int_convs = nn.ModuleList([GINEConv(make_node_mlp(hidden_dim), edge_dim=int_edge_dim)])
        self.int_bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim)])
        for _ in range(num_layers - 1):
            self.int_convs.append(GINEConv(make_node_mlp(hidden_dim), edge_dim=int_edge_dim))
            self.int_bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.cross_bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(4 * hidden_dim + mol_desc_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, lig_data, poc_data, int_data=None):
        x_l, edge_l, edge_attr_l, batch_l = lig_data.x, lig_data.edge_index, lig_data.edge_attr, lig_data.batch
        x_l = self.lig_input_proj(x_l)
        for conv, bn in zip(self.lig_convs, self.lig_bns):
            x_l = F.relu(bn(conv(x_l, edge_l, edge_attr_l)))
            x_l = self.dropout(x_l)
        h_l = global_mean_pool(x_l, batch_l)

        x_p, edge_p, edge_attr_p, batch_p = poc_data.x, poc_data.edge_index, poc_data.edge_attr, poc_data.batch
        x_p = self.poc_input_proj(x_p)
        for conv, bn in zip(self.poc_convs, self.poc_bns):
            x_p = F.relu(bn(conv(x_p, edge_p, edge_attr_p)))
            x_p = self.dropout(x_p)
        h_p = global_mean_pool(x_p, batch_p)

        if int_data is not None:
            x_i, edge_i, edge_attr_i, batch_i = int_data.x, int_data.edge_index, int_data.edge_attr, int_data.batch
            x_i = self.int_input_proj(x_i)
            for conv, bn in zip(self.int_convs, self.int_bns):
                x_i = F.relu(bn(conv(x_i, edge_i, edge_attr_i)))
                x_i = self.dropout(x_i)
            h_i = global_mean_pool(x_i, batch_i)
        else:
            h_i = torch.zeros_like(h_l)

        interaction = F.relu(self.cross_bilinear(h_l, h_p))
        if hasattr(lig_data, "mol_desc"):
            desc = lig_data.mol_desc.to(h_l.dtype)
            batch_size = h_l.size(0)

            # PyG batching can flatten per-graph descriptors into 1D (e.g., B*17).
            # Recover [B, D] shape when possible.
            if desc.dim() == 1:
                if batch_size > 0 and desc.numel() % batch_size == 0:
                    desc = desc.view(batch_size, -1)
                else:
                    desc = desc.unsqueeze(0)

            if desc.dim() == 2 and desc.size(0) != batch_size and batch_size > 0 and desc.numel() % batch_size == 0:
                desc = desc.view(batch_size, -1)

            if desc.size(0) != h_l.size(0):
                if desc.size(0) == 1:
                    desc = desc.repeat(batch_size, 1)
                else:
                    raise ValueError(
                        f"mol_desc batch mismatch: got {tuple(desc.shape)}, expected batch {h_l.size(0)}"
                    )

            expected_desc_dim = self.fc1.in_features - 4 * h_l.size(1)
            if desc.size(1) != expected_desc_dim:
                raise ValueError(
                    f"mol_desc feature mismatch: got {desc.size(1)}, expected {expected_desc_dim}; shape={tuple(desc.shape)}"
                )

            h = torch.cat([h_l, h_p, h_i, interaction, desc], dim=1)
        else:
            h = torch.cat([h_l, h_p, h_i, interaction], dim=1)
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1)
