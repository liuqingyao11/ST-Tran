import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        #self.bn = nn.BatchNorm2d(nhid)
 
    def forward(self, x, adj):
        #print(adj.shape)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        #print('after gcn:\n',x)
        return x 
