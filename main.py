import os.path as osp
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP
import torch_geometric.transforms as T

from DrugData_load import MyDrugDataset

from numpy.random import seed
import numpy as np
import random as python_random

parser = argparse.ArgumentParser()

# parser.add_argument('--dataset', type=str, default='Cora')
# parser.add_argument('--epochs', type=int, default=300)
# parser.add_argument('--channels', type=int, default=128)
# parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--model', type=str, default='VGNAE')

# parser.add_argument('--model', type=str, default='GNAE')
# parser.add_argument('--dataset', type=str, default='drug_four_lowdim_0.98')
# parser.add_argument('--epochs', type=int, default=1000)
# parser.add_argument('--channels', type=int, default=64)
# parser.add_argument('--scaling_factor', type=float, default=5.0)
# parser.add_argument('--training_rate', type=float, default=0.8)

# parser.add_argument('--model', type=str, default='GNAE')
parser.add_argument('--dataset', type=str, default='drug_four_lowdim_0.98')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--scaling_factor', type=float, default=5.0)
parser.add_argument('--training_rate', type=float, default=0.8)

# # 设随机种子
# parser.add_argument('--seed', type=float, default=57)


# # 换的批归一化
# parser.add_argument('--scaling_factor', type=float, default=1.0)

args = parser.parse_args()

# def set_seed():
#     seed(args.seed)
#     np.random.seed(args.seed)
#     python_random.seed(args.seed)
#     torch.manual_seed(args.seed)


dataset = MyDrugDataset('data', args.dataset)

data = dataset[0]

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=2, alpha=0.5)

        # # 换的批归一化
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, not_prop=0):
        if args.model == 'GNAE':
            x = self.linear1(x)
            x = F.normalize(x, p=2, dim=1) * args.scaling_factor
            x = self.propagate(x, edge_index)

            # # 换的批归一化
            # x = self.linear1(x)
            # x = self.bn(x)
            # x = F.normalize(x, p=2, dim=1) * args.scaling_factor
            # x = self.propagate(x, edge_index)

            return x


        if args.model == 'VGNAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)
            x = self.linear2(x)
            x = F.normalize(x, p=2, dim=1) * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x, x_

        return x


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels = args.channels
train_rate = args.training_rate
val_ratio = (1-args.training_rate) / 3
test_ratio = (1-args.training_rate) / 3 * 2
data = train_test_split_edges(data.to(dev), val_ratio=val_ratio, test_ratio=test_ratio)

N = int(data.x.size()[0])

if args.model == 'GNAE':   
    model = GAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)
if args.model == 'VGNAE':
    model = VGAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)

data.train_mask = data.val_mask = data.test_mask = data.y = None
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.model in ['VGAE']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss

# def test(pos_edge_index, neg_edge_index, plot_his=0):
def test(pos_edge_index, neg_edge_index):
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)

    return model.test(z, pos_edge_index, neg_edge_index)

for epoch in range(1, args.epochs):
    loss = train()
    loss = float(loss)
    
    with torch.no_grad():
        test_pos, test_neg = data.test_pos_edge_index, data.test_neg_edge_index
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))
