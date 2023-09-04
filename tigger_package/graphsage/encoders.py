import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, device='cpu', 
            feature_transform=False,
            dropout=0): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.device = device
        self.aggregator.device = device
        input_dim_fc1 = self.feat_dim if self.gcn else 2 * self.feat_dim
        self.fc1 = nn.Linear(input_dim_fc1 ,embed_dim).to(self.device)
        self.fc2 = nn.Linear(embed_dim, embed_dim).to(self.device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
                self.num_sample)
        #print(neigh_feats.shape)
        if not self.gcn:
            self_feats = self.features(torch.LongTensor(nodes).to(self.device))
            combined = torch.cat([self_feats, neigh_feats], dim=1).to(self.device)
        else:
            combined = neigh_feats
        
        combined = self.dropout1(combined)
        combined = F.leaky_relu(self.fc1(combined), negative_slope=0.03)
        combined = self.dropout2(combined)
        combined = torch.tanh(self.fc2(combined)) ### used for msg network
        return combined
