from dgllife.model import GAT_dgl
import torch


class GAT(torch.nn.Module):
    """
    GAT-based model.
    """

    def __init__(self, *args, **kwargs):
        super(GAT, self).__init__()
        self.gnn = GAT_dgl(*args, **kwargs)
        self.readout = torch.nn.Linear(self.gnn.hidden_feats[-1], 1)

    def forward(self, g, feats):
        node_preds = self.gnn(g, feats)
        node_preds = self.readout(node_preds)
        return node_preds.sum(dim=0)
