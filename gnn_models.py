import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool


class MofGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin_out = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def encode(self, x, edge_index, batch, edge_attr=None):
        """Return graph-level embedding (before final classification head)."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  # (batch_size, hidden_channels)

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # embedding

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.encode(x, edge_index, batch, edge_attr)
        out = self.lin_out(x)
        return out  # logits

