import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

import utils


class MyDataSets(Dataset):
    def __init__(self, root: str, dataset_name: str, num_views: int = 3, num_neibs: int = 8, fea_start: int = -1, fea_end: int = -1):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name  # 将数据集名称赋值给实例变量

        od_data = np.loadtxt(root + dataset_name + '.csv',
                             delimiter=',')  # 从指定路径的 csv 文件中加载数据，使用逗号作为分隔符，返回一个 numpy 数组 od_data。

        xs = None  # 始化 xs 变量
        if fea_start == -1 and fea_end == -1:
            xs = od_data[:, 0:-1]  # 将 od_data 中除最后一列外的所有列赋值给 xs
        else:
            xs = od_data[:, fea_start: fea_end]  # 将 od_data 中从特征起始位置到特征结束位置的所有列赋值给 xs。
        self.X_train = xs.astype(np.float32)  # 将 xs 转换为 float32 类型，并赋值给实例变量 X_train。
        X_train_global=od_data[:, 0:-1]
        X_train_global=X_train_global.astype(np.float32)
        self.X_train_global=torch.tensor(X_train_global, dtype=torch.float32)

        labels = od_data[:, -1]  # 将最后一列赋值给 labels
        labels = labels.astype(np.float32)
        self.labels = labels

        self.size = labels.shape[0]
        self.data = torch.tensor(self.X_train, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.float32)

        # a neighbor list for each view
        self.num_views = num_views  # 视图数量
        self.num_neibs = num_neibs  # 每个数据样本的邻居数量
        nbrs = NearestNeighbors(n_neighbors=num_neibs + 1, algorithm='ball_tree').fit(self.X_train)
        nbrs1 = NearestNeighbors(n_neighbors=num_neibs + 1, algorithm='ball_tree').fit(X_train_global)
        _, indices = nbrs.kneighbors(self.X_train)
        _, indices1 = nbrs1.kneighbors(X_train_global)
        self.neibs_local =indices
        self.neibs_global = indices1
        # self.weights_global = []


    def __getitem__(self, index):
        # 获取数据和标签
        sample, target = self.data[index], (self.targets[index])
        id_local_neighbor = self.neibs_local[index][1:self.num_neibs+1]
        neighbor_local = self.data[id_local_neighbor]
        id_global_neighbor = self.neibs_global[index][1:self.num_neibs + 1]
        neighbor_global = self.data[id_global_neighbor]
        return sample, target,  index, neighbor_local, id_local_neighbor, neighbor_global, id_global_neighbor ##, weight_global

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.data)