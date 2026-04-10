import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import scatter


class GraphSAGENet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4, dropout=0.2, edge_dim=3):
        super().__init__()
        self.edge_dim = edge_dim
        self.edge_h = hidden_dim
        self.edge_linears = nn.ModuleList([nn.Linear(edge_dim, hidden_dim) for _ in range(num_layers)])
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim + hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim + hidden_dim, hidden_dim))
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

    def _incident_edge_ctx(self, edge_attr, edge_index, num_nodes, edge_lin):
        _, col = edge_index
        e = F.relu(edge_lin(edge_attr))
        return scatter(e, col, dim=0, dim_size=num_nodes, reduce="mean")

    def forward(self, x, edge_index, batch, edge_attr=None):
        edge_attr = self._edge_attr(edge_index, edge_attr, x)
        h = x
        for conv, bn, edge_lin in zip(self.convs, self.bns, self.edge_linears):
            ctx = self._incident_edge_ctx(edge_attr, edge_index, h.size(0), edge_lin)
            h_cat = torch.cat([h, ctx], dim=-1)
            h_res = h
            h = conv(h_cat, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if h_res.size(-1) == h.size(-1):
                h = h + h_res
        h = global_mean_pool(h, batch)
        return self.mlp(h)

    def get_embedding(self, x, edge_index, batch, edge_attr=None):
        edge_attr = self._edge_attr(edge_index, edge_attr, x)
        h = x
        for conv, bn, edge_lin in zip(self.convs, self.bns, self.edge_linears):
            ctx = self._incident_edge_ctx(edge_attr, edge_index, h.size(0), edge_lin)
            h_cat = torch.cat([h, ctx], dim=-1)
            h_res = h
            h = conv(h_cat, edge_index)
            h = bn(h)
            h = F.relu(h)
            if h_res.size(-1) == h.size(-1):
                h = h + h_res
        return global_mean_pool(h, batch)
