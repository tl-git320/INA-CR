import random

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

import utils


class MyDataSets(Dataset):
    def __init__(self, root: str, dataset_name: str, num_views: int = 3, num_neibs: int = 8, fea_start: int = -1, fea_end: int = -1):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name

        od_data = np.loadtxt(root + dataset_name + '.csv',
                             delimiter=',')

        xs = None
        if fea_start == -1 and fea_end == -1:
            xs = od_data[:, 0:-1]
        else:
            xs = od_data[:, fea_start: fea_end]
        self.X_train = xs.astype(np.float32)
        X_train_global=od_data[:, 0:-1]
        X_train_global=X_train_global.astype(np.float32)
        self.X_train_global=torch.tensor(X_train_global, dtype=torch.float32)

        labels = od_data[:, -1]
        labels = labels.astype(np.float32)
        self.labels = labels

        self.size = labels.shape[0]
        self.data = torch.tensor(self.X_train, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.float32)

        # a neighbor list for each view
        self.num_views = num_views
        self.num_neibs = num_neibs
        nbrs = NearestNeighbors(n_neighbors=num_neibs + 1, algorithm='ball_tree').fit(self.X_train)
        nbrs1 = NearestNeighbors(n_neighbors=num_neibs + 1, algorithm='ball_tree').fit(X_train_global)
        _, indices = nbrs.kneighbors(self.X_train)
        _, indices1 = nbrs1.kneighbors(X_train_global)
        self.neibs_local =indices
        self.neibs_global = indices1
        # self.weights_global = []


    def __getitem__(self, index):
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