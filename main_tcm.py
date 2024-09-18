import os.path as osp
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP

from DrugData_load import MyDrugDataset

from numpy.random import seed
import numpy as np
import random as python_random

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()

# tcm_pare
parser.add_argument('--model', type=str, default='GNAE')
parser.add_argument('--dataset', type=str, default='tcm_more')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--scaling_factor', type=float, default=3.6)
parser.add_argument('--training_rate', type=float, default=0.8)

parser.add_argument('--embedding_dimension', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
#
# # parser.add_argument('--model', type=str, default='VGNAE')
# parser.add_argument('--model', type=str, default='GNAE')
# parser.add_argument('--dataset', type=str, default='tcm_more')
# # parser.add_argument('--dataset', type=str, default='tcm_bert32')
# # parser.add_argument('--epochs', type=int, default=500)  # 固定了 vgnae
# # parser.add_argument('--scaling_factor', type=float, default=5.0)  # 固定了 # 0.6的traning 5.0 0.9434 0.6的也是3.6效果好
# # parser.add_argument('--scaling_factor', type=float, default=3.5) # vgnae
# # 0.8 1.0 AUC: 0.8774, AP: 0.8680 1.8 AUC: 0.9154, AP: 0.9021  3.6 9610 9538     5.0 0.9687, AP: 0.9572  6.0 AUC: 0.9657, AP: 0.9526
# # parser.add_argument('--channels', type=int, default=256)  # 固定了
# # parser.add_argument('--channels', type=int, default=128)  # 固定了
# parser.add_argument('--channels', type=int, default=64)
# # parser.add_argument('--embedding_dimension', type=int, default=64)  # 固定了
# parser.add_argument('--embedding_dimension', type=int, default=32)
# # parser.add_argument('--learning_rate', type=float, default=0.005)  # 固定了
# # parser.add_argument('--learning_rate', type=float, default=0.0005)  # 固定了
# # parser.add_argument('--learning_rate', type=float, default=0.01)  # 固定了 # vgnae
# # parser.add_argument('--training_rate', type=float, default=0.9)  # 固定了
#
# # # gnae 0.9305
# # parser.add_argument('--epochs', type=int, default=1000)
# # parser.add_argument('--learning_rate', type=float, default=0.0003)
# # parser.add_argument('--scaling_factor', type=float, default=3.5)
#
# # gnae 0.005:0.9349   0.01:0.9368
# parser.add_argument('--epochs', type=int, default=600)
# parser.add_argument('--learning_rate', type=float, default=0.01)  # 固定了
# # parser.add_argument('--scaling_factor', type=float, default=4.0) # 0.9380
# parser.add_argument('--scaling_factor', type=float, default=5.0) # 0.9393
# parser.add_argument('--training_rate', type=float, default=0.8)  # gnae AUC: 0.9217, AP: 0.9339

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, args.embedding_dimension)
        # self.linear2 = nn.Linear(in_channels, out_channels)
        # self.propagate = APPNP(K=2, alpha=0.5)
        # self.propagate = APPNP(K=2, alpha=0.25)
        self.propagate = APPNP(K=2, alpha=0.35) # vgnae gnae
        # self.propagate = APPNP(K=2, alpha=0.1)
        # self.act1 = nn.Sequential(nn.Tanh())
        # self.linear2 = nn.Linear(out_channels, 64)
        # 连续进行K次的特征汇聚。在每次汇聚时以α的概率保留住初始的特征信息，再以α的概率保留当前层汇聚的特征信息，将二者相加作为此层汇聚后的特征信息
        # self.propagate = APPNP(K=2, alpha=0.5)
        self.act1 = nn.Sequential(nn.Tanh()) # gnae 0.9422

        # self.linear2 = nn.Linear(out_channels, 128)
        # self.linear3 = nn.Linear(64, 32)
        # self.linear3 = nn.Linear(128, 64)

        # self.act1 = nn.Sequential(nn.ReLU()) # 0.9138
        # self.act1 = nn.Sequential(nn.Sigmoid())
        # self.act2 = nn.Sequential(nn.Tanh())


    def forward(self, x, edge_index, not_prop=0):
        if args.model == 'GNAE':

            x = self.linear1(x)
            x = self.act1(x)
            # x = self.linear2(x)

            # 使用torch.nn.functional.normalize进行L2归一化
            x = F.normalize(x, p=2, dim=1) * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x



        if args.model == 'VGNAE':
            x_ = self.linear1(x)
            x_ = self.propagate(x_, edge_index)
            x = self.linear2(x)
            x = F.normalize(x, p=2, dim=1) * args.scaling_factor
            x = self.propagate(x, edge_index)
            return x, x_



        return x

def train():

    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.model in ['VGAE']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss

def test(pos_edge_index, neg_edge_index):

    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)

    return model.test(z, pos_edge_index, neg_edge_index)

def draw_curve(name, the_values):

    plt.plot(the_values)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.title('Training {} Curve'.format(name))
    plt.savefig('{}/tcm/{}_curve.png'.format(args.model, name))
    plt.cla()

def save_output(n, para, loss_value, auc_value, ap_value):
    output_list = []
    pd.DataFrame(para).to_csv('{}/output/lowdim_0.98_{}.csv'.format(args.model, n), mode='a', index=False, encoding='utf-8', header=False)
    for i in range(args.epochs-1):
        output_list.append([loss_value[i], auc_value[i], ap_value[i]])
    data = pd.DataFrame(output_list, columns=['LOSS', 'AUC', 'AP'])
    data.to_csv('{}/output/lowdim_0.98_{}.csv'.format(args.model, n), mode='a', index=False, encoding='utf-8', header=True)

if __name__ == '__main__':
    # for j in range(1, 2):
        loss_value = []
        auc_value = []
        ap_value = []
        parameter_list = []
        args = parser.parse_args()
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = MyDrugDataset('data', args.dataset)
        data = dataset[0]

        channels = args.channels
        train_rate = args.training_rate

        val_ratio = (1 - args.training_rate) / 3
        test_ratio = (1 - args.training_rate) / 3 * 2

        data = train_test_split_edges(data.to(dev), val_ratio=val_ratio, test_ratio=test_ratio)

        test_pos = data.test_pos_edge_index

        pos_a_index = test_pos[0].tolist()
        pos_b_index = test_pos[1].tolist()

        index_ab = pd.DataFrame({'index_a': pos_a_index, 'index_b': pos_b_index})
        index_ab.to_csv('tcm_top/tcm_test_index.csv', index=None, encoding='utf8')

        N = int(data.x.size()[0])

        if args.model == 'GNAE':
            model = GAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)
        if args.model == 'VGNAE':
            model = VGAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)

        data.train_mask = data.val_mask = data.test_mask = data.y = None
        x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

        for epoch in range(1, args.epochs):

            loss = train()
            loss = float(loss)
            loss_value.append(loss)

            with torch.no_grad():
                test_pos, test_neg = data.test_pos_edge_index, data.test_neg_edge_index
                pos_pred_value, auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)


                print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f}, AP: {:.4f}, Learning Rate: {:.9f}'.format(epoch, loss, auc, ap, optimizer.param_groups[0]['lr']))

                if epoch >= 495:
                    parameter_list.append(pos_pred_value)
                # auc_value.append(auc)
                # ap_value.append(ap)



        # draw_curve('Loss', loss_value)
        # draw_curve('AUC', auc_value)
        # draw_curve('AP', ap_value)
        # save_output(j, parameter_list, loss_value, auc_value, ap_value)
        score_data = pd.DataFrame(parameter_list)
        score_data.to_csv('tcm_top/tcm_top.csv', mode='a', index=False, encoding='utf-8', header=True)
