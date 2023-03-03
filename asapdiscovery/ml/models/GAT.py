from dgllife.model import GAT as GAT_dgl
from dgllife.model import WeightedSumAndMax

# from dgllife.model import GATPredictor
import torch


class GAT(torch.nn.Module):
    """
    GAT-based model.
    """

    def __init__(self, *args, **kwargs):
        # super(GAT, self).__init__(*args, **kwargs)
        super(GAT, self).__init__()
        self.gnn = GAT_dgl(*args, **kwargs)

        # Copied from GATPredictor class, figure out how many features the last
        #  layer of the GNN will have
        if self.gnn.agg_modes[-1] == "flatten":
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)

        # Use given hidden feats if supplied, otherwise use 1/2 gnn_out_feats
        if "predictor_hidden_feats" in kwargs:
            predictor_hidden_feats = kwargs["predictor_hidden_feats"]
        else:
            predictor_hidden_feats = gnn_out_feats // 2

        # 2 layer MLP with ReLU activation (borrowed from GATPredictor)
        self.predict = torch.nn.Sequential(
            torch.nn.Linear(2 * gnn_out_feats, predictor_hidden_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(predictor_hidden_feats, 1),
        )

    def forward(self, g, feats):
        device = next(self.parameters()).device
        g = g.to(device)
        feats = feats.to(device)
        ### doesn't work bc GATPredictor calls torch.functional.batch_norm,
        ###  which requires multiple channels (which we don't have)
        # return super(GAT, self).forward(g, feats)
        node_feats = self.gnn(g, feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)
