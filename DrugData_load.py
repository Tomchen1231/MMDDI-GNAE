import torch
import os.path as osp
from torch_geometric.data import InMemoryDataset
from typing import Callable, Optional
import torch_geometric.transforms as T
from torch_geometric.utils import coalesce
import numpy as np
from torch_geometric.data import Data

# def get_transform(options):
#     """Splits data to train, validation and test, and moves them to the device"""
#     transform = T.Compose([
#         T.NormalizeFeatures(),
#         T.ToDevice(options["device"]),
#         T.RandomLinkSplit(num_val=0.05,
#                           num_test=0.15,
#                           is_undirected=True,
#                           split_labels=True,
#                           add_negative_train_samples=False),
#     ])
#
#     return transform

class MyDrugDataset(InMemoryDataset):
    def __init__(self, root: str, name: str, geom_gcn_preprocess: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    # 返回数据集源文件名，告诉原始的数据集存放在哪个文件夹下面，如果数据集已经存放进去了，那么就会直接从raw文件夹中读取。
        print(root, name)
    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        if self.geom_gcn_preprocess:
            return (['two_low_dim.txt', 'inter.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'


    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    def download(self):
        pass

    # 生成数据集所用的方法，程序第一次运行才执行并生成processed文件夹的处理过后数据的文件，否则必须删除已经生成的processed文件夹中的所有文件才会重新执行此函数
    def process(self):
        # Read data into huge `Data` list.
        # Read data into huge `Data` list.

        if self.geom_gcn_preprocess:
            print(self.raw_paths[0])
            # 节点 标签
            data_node = np.genfromtxt(self.raw_paths[0], dtype=np.dtype(str))
            # features = data_node[:, 1:]
            features = data_node[:, :]
            x = [[float(v) for v in n] for n in features]
            x = torch.tensor(x, dtype=torch.float)
            print(len(x))
            y_matrix = np.ones((len(x), 1), dtype=int)
            print(y_matrix.shape)
            y = [[float(q) for q in z] for z in y_matrix]
            y = torch.tensor(y, dtype=torch.long)

            # 边
            data_edge = np.genfromtxt(self.raw_paths[1], dtype=np.dtype(str))
            data_list = [[int(one_edge) for one_edge in pair] for pair in data_edge]
            edge_index = torch.tensor(data_list, dtype=torch.long).t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))

            # mask
            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                f = np.load(filepath)
                train_masks += [torch.from_numpy(f['train_mask'])]
                val_masks += [torch.from_numpy(f['val_mask'])]
                test_masks += [torch.from_numpy(f['test_mask'])]
            train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = torch.from_numpy(data['features']).to(torch.float)
            edge_index = torch.from_numpy(data['edges']).to(torch.long)
            edge_index = edge_index.t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=x.size(0))
            y = torch.from_numpy(data['target']).to(torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

