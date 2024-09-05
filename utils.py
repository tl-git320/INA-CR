import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

DATA_DIM = 3*64*64
OUTLIER_RATE = 0.15
data_root_path = './datasets/'


def get_view_fea(NUM_VIEWS):
    '''
    该函数的作用是将总体特征维度DATA_DIM平分到NUM_VIEWS个视图中,
    其中前面几个视图具有平均的特征维度avg_num,最后几个视图具有avg_num+1的特征维度,
    :return: s, e 返回每个视图对应的特征维度起始索引s和结束索引e
    '''
    avg_num = int(DATA_DIM / NUM_VIEWS)
    # left_num = DATA_DIM % NUM_VIEWS
    fea_num = [avg_num for a in range(NUM_VIEWS)]
    # for i in range(left_num):
    #     fea_num[NUM_VIEWS - 1 - i] += 1
    start = 0
    end = 0
    s = [0 for a in range(NUM_VIEWS)]
    e = [0 for a in range(NUM_VIEWS)]
    for v in range(NUM_VIEWS):
        end = start + fea_num[v]
        s[v] = start
        e[v] = end
        start = end
    return s, e


def compute_auc(labels, scores):
    return roc_auc_score(labels, scores)


def get_percentile(scores, threshold):
    per = np.percentile(scores, 100 - int(100 * threshold))
    return per


def compute_f1_score(labels, scores):
    per = get_percentile(scores, OUTLIER_RATE)
    y_pred = (scores >= per)
    # print(np.sum(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(labels.astype(int),
                                                               y_pred.astype(int),
                                                               average='binary')
    return precision, recall, f1

def computeMinMaxScalerTensor(x):
    min_x = torch.min(x)
    max_x = torch.max(x)
    x = (x-min_x)/(max_x-min_x)
    return x

# 递归创建目录
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

