import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool


class MPNNNet(nn.Module):
    def __init__(self, input_dim, edge_dim=3, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.edge_dim = edge_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        nn1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * hidden_dim),
        )
        self.convs.append(NNConv(input_dim, hidden_dim, nn1, aggr="mean"))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        in_ch = hidden_dim
        for _ in range(num_layers - 2):
            nn_k = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_ch * hidden_dim),
            )
            self.convs.append(NNConv(in_ch, hidden_dim, nn_k, aggr="mean"))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        nn_last = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_ch * hidden_dim),
        )
        self.convs.append(NNConv(in_ch, hidden_dim, nn_last, aggr="mean"))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _edge_attr(self, edge_index, edge_attr, ref):
        if edge_attr is not None:
            return edge_attr
        return ref.new_zeros((edge_index.size(1), self.edge_dim))

    def forward(self, x, edge_index, batch, edge_attr=None):
        edge_attr = self._edge_attr(edge_index, edge_attr, x)
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x_res.size(-1) == x.size(-1):
                x = x + x_res
        x = global_mean_pool(x, batch)
        return self.mlp(x)

    def get_embedding(self, x, edge_index, batch, edge_attr=None):
        edge_attr = self._edge_attr(edge_index, edge_attr, x)
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            if x_res.size(-1) == x.size(-1):
                x = x + x_res
        return global_mean_pool(x, batch)
