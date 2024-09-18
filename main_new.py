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

# parser.add_argument('--model', type=str, default='VGNAE')
parser.add_argument('--model', type=str, default='GNAE')
parser.add_argument('--dataset', type=str, default='drug_four_lowdim_0.98')
# parser.add_argument('--dataset', type=str, default='tcm_0.98')

parser.add_argument('--epochs', type=int, default=500)  # 固定了
parser.add_argument('--scaling_factor', type=float, default=5.0)  # 固定了 # 0.6的traning 5.0 0.9434 0.6的也是3.6效果好
# 0.8 1.0 AUC: 0.8774, AP: 0.8680 1.8 AUC: 0.9154, AP: 0.9021  3.6 9610 9538     5.0 0.9687, AP: 0.9572  6.0 AUC: 0.9657, AP: 0.9526

# parser.add_argument('--channels', type=int, default=256)  # 固定了
parser.add_argument('--channels', type=int, default=128)  # 固定了
parser.add_argument('--embedding_dimension', type=int, default=64)  # 固定了

parser.add_argument('--learning_rate', type=float, default=0.005)  # 固定了

parser.add_argument('--training_rate', type=float, default=0.9)  # 固定了

#

# parser.add_argument('--model', type=str, default='GNAE')
# parser.add_argument('--dataset', type=str, default='drug_four_lowdim_0.98')

# parser.add_argument('--epochs', type=int, default=500) #  300,350,400,450,500
# parser.add_argument('--scaling_factor', type=float, default=5.0) # 1.0,1.8,3.6,5.0,6.0

# parser.add_argument('--channels', type=int, default=256)  # 128,256,512
# parser.add_argument('--embedding_dimision', type=int, default=64)  # 16, 32, 64, 128

# parser.add_argument('--learning_rate', type=float, default=0.05) # 0.001, 0.005, 0.01, 0.05
# parser.add_argument('--training_rate', type=float, default=0.9) # 0.4, 0.6, 0.7, 0.8, 0.9




# # 设随机种子
# parser.add_argument('--seed', type=float, default=57)

# # 换的批归一化
# parser.add_argument('--scaling_factor', type=float, default=1.0)

# def set_seed():
#     seed(args.seed)
#     np.random.seed(args.seed)
#     python_random.seed(args.seed)
#     torch.manual_seed(args.seed)



class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, args.embedding_dimension)
        self.propagate = APPNP(K=2, alpha=0.5)
        self.act1 = nn.Sequential(nn.Tanh())
        # self.linear2 = nn.Linear(out_channels, 64)
        # 连续进行K次的特征汇聚。在每次汇聚时以α的概率保留住初始的特征信息，再以α的概率保留当前层汇聚的特征信息，将二者相加作为此层汇聚后的特征信息
        # self.propagate = APPNP(K=2, alpha=0.5)
        # self.act1 = nn.Sequential(nn.Tanh())

        # self.linear2 = nn.Linear(out_channels, 128)
        # self.linear3 = nn.Linear(64, 32)
        # self.linear3 = nn.Linear(128, 64)

        # self.act1 = nn.Sequential(nn.ReLU())
        # self.act1 = nn.Sequential(nn.Sigmoid())
        # self.act2 = nn.Sequential(nn.Tanh())


    def forward(self, x, edge_index, not_prop=0):
        if args.model == 'GNAE':


            x = self.linear1(x)
            x = self.act1(x)
            x = self.linear2(x)

            # 使用torch.nn.functional.normalize进行L2归一化
            # x = F.normalize(x, p=2, dim=1) * args.scaling_factor
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

def test(pos_edge_index, neg_edge_index):
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)

    return model.test(z, pos_edge_index, neg_edge_index)

def draw_curve(name, the_values):

    plt.plot(the_values)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.title('Training {} Curve'.format(name))
    plt.savefig('{}/learningrate_plt_newencoder/{}_curve.png'.format(args.model, name))
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

        # the_edge = data.test_pos_edge_index.cpu().numpy()
        # the_edge = the_edge.T
        # the_edge = pd.DataFrame(the_edge)
        # the_edge.to_csv('{}/find_top_new/known_edge.csv'.format(args.model), mode='a', index=False, encoding='utf-8',
        #                 header=True)

        # the_train = data.train_pos_edge_index.cpu().numpy()
        # the_train = the_train.T
        # the_train = pd.DataFrame(the_train)

        N = int(data.x.size()[0])

        if args.model == 'GNAE':
            model = GAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)
            # model = GAE(EncoderGCN(data.x.size()[1], 64)).to(dev)
        if args.model == 'VGNAE':
            model = VGAE(Encoder(data.x.size()[1], channels, data.train_pos_edge_index)).to(dev)

        data.train_mask = data.val_mask = data.test_mask = data.y = None
        x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # # 定义学习率调度器
        # scheduler = StepLR(optimizer, step_size=700, gamma=0.1)
        # parameter_list = ['model:'+args.model, 'epochs:'+str(args.epochs), 'channels:'+str(args.channels), 'scaling_factor:'+str(args.scaling_factor), 'learning_rate:'+str(args.learning_rate)]
        for epoch in range(1, args.epochs):

            # if epoch > 300 and epoch % 150 == 0:
            # if epoch == 450:
            #     # new_lr = optimizer.param_groups[0]['lr'] * 0.1  # 降低学习率的比例，这里示例为0.1
            #     # new_lr = 0.001
            #     new_lr = 0.0005
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = new_lr
            # if epoch == 500:
            #     # new_lr = optimizer.param_groups[0]['lr'] * 0.1  # 降低学习率的比例，这里示例为0.1
            #     new_lr = 0.00005
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = new_lr


            loss = train()
            loss = float(loss)
            loss_value.append(loss)
            # scheduler.step()

            # print("Epoch {:03d} - Learning Rate: {:.9f}".format(epoch, optimizer.param_groups[0]['lr']))
            with torch.no_grad():
                test_pos, test_neg = data.test_pos_edge_index, data.test_neg_edge_index
                pos_pred_value, auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)

                # if epoch >= 495:
                #
                #     pred_edge = pd.DataFrame(pos_pred_value)
                #     pred_edge.to_csv('{}/find_top_new/pred_edge_{}.csv'.format(args.model, epoch), mode='a', index=False, encoding='utf-8', header=True)

                print('Epoch: {:03d}, LOSS: {:.4f}, AUC: {:.4f}, AP: {:.4f}, Learning Rate: {:.9f}'.format(epoch, loss, auc, ap, optimizer.param_groups[0]['lr']))

                auc_value.append(auc)
                ap_value.append(ap)

        draw_curve('Loss', loss_value)
        draw_curve('AUC', auc_value)
        draw_curve('AP', ap_value)
        # save_output(j, parameter_list, loss_value, auc_value, ap_value)

