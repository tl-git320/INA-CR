import logging
import os
import time

import numpy as np
import torch

import utils
from MyDataSets import MyDataSets
from MyTrainer import MyTrainer
from network import GeneratorDecode, GeneratorEncode, Prediction

torch.manual_seed(1)  # reproducible

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run(data_path, dataset_name, batch_size, learning_rate, k_neibs, module_weight, albation_set, num_rounds=20,
        num_epochs=5, num_views=2):
    print('#########  LocalAE conducted on ' + dataset_name)

    logging.basicConfig(level=logging.INFO, filename='running_log')
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(torch.cuda.current_device())
    num_neibs = k_neibs
    pre_train = False
    pre_epochs = 100

    s, e = utils.get_view_fea(num_views)

    en_nets = [None for a in range(num_views)]
    de_nets = [None for a in range(num_views)]
    prediction_nets = [None for a in range(num_views)]
    datasets = [None for a in range(num_views)]
    view_encoded = [None for a in range(num_views)]
    recon_error = [None for a in range(num_views)]
    prediction_losss = [None for a in range(num_views)]
    nei_sum = [None for a in range(num_views)]
    ins_sum = [None for a in range(num_views)]
    labels = None

    for i in range(num_views):
        datasets[i] = MyDataSets(root=data_path, dataset_name=dataset_name, num_neibs=num_neibs, fea_start=s[i],
                                 fea_end=e[i])
    if labels is None:
        labels = datasets[0].get_labels()

    for v in range(num_views):  # 每个视图有一个神经网络
        en_nets[v] = GeneratorEncode(e[v] - s[v])
        de_nets[v] = GeneratorDecode(e[v] - s[v])
        prediction_nets[v] = Prediction(e[v] - s[v])

    Trainer = MyTrainer(optimizer_name='amsgrad',
                        lr=learning_rate,
                        n_epochs=num_epochs,
                        lr_milestones=tuple(()),
                        batch_size=batch_size,
                        weight_decay=1e-6,
                        device=device,
                        n_jobs_dataloader=0)

    for eround in range(num_rounds):
        for id_view in range(num_views):
            en_nets, de_nets[id_view], view_encoded[id_view], recon_error[id_view], nei_sum[
                id_view], ins_sum[id_view], prediction_losss[id_view] = \
                Trainer.train(eround, id_view, datasets[id_view], en_nets, de_nets[id_view],
                              prediction_nets[id_view], pre_train,
                              pre_epochs, module_weight, albation_set, datasets)
        print('round %d: ' % eround)

    recon_scores, nei_sum, ins_sum, prediction_losss = cal_scores(view_encoded, recon_error, nei_sum,
                                                                  ins_sum, prediction_losss)
    total_scores = albation_set[0] * recon_scores + albation_set[1] * nei_sum + albation_set[3] * ins_sum + \
                   albation_set[2] * prediction_losss
    max_auc = utils.compute_auc(labels, total_scores)
    total_fs = utils.compute_f1_score(labels, total_scores)
    print('total performance: %.4f  %.4f  %.4f  %.4f' % (
        max_auc, total_fs[0], total_fs[1], total_fs[2]))
    return max_auc, total_fs[2]


def cal_scores(view_encoded, recon_error, dis_global, mse_sum, pre_loss):
    num_views = len(view_encoded)
    num_obj = view_encoded[0].shape[0]

    recon_scores = np.zeros(num_obj)
    dis_globals = np.zeros(num_obj)
    mse_sums = np.zeros(num_obj)
    pre_losss = np.zeros(num_obj)
    for i in range(num_views):
        if (np.max(recon_error[i]) - np.min(recon_error[i])) == 0:
            s1 = recon_error[i]
        else:
            s1 = (recon_error[i] - np.min(recon_error[i])) / (np.max(recon_error[i]) - np.min(recon_error[i]))
        if (np.max(dis_global[i]) - np.min(dis_global[i])) == 0:
            s2 = dis_global[i]
        else:
            s2 = (dis_global[i] - np.min(dis_global[i])) / (np.max(dis_global[i]) - np.min(dis_global[i]))
        if (np.max(mse_sum[i]) - np.min(mse_sum[i])) == 0:
            s3 = mse_sum[i]
        else:
            s3 = (mse_sum[i] - np.min(mse_sum[i])) / (np.max(mse_sum[i]) - np.min(mse_sum[i]))
        if (np.max(pre_loss[i]) - np.min(pre_loss[i])) == 0:
            s5 = pre_loss[i]
        else:
            s5 = (pre_loss[i] - np.min(pre_loss[i])) / (np.max(pre_loss[i]) - np.min(pre_loss[i]))
        recon_scores += s1
        dis_globals += s2
        mse_sums += s3
        pre_losss += s5

    return recon_scores, dis_globals, mse_sums, pre_losss  #


if __name__ == '__main__':
    file_path = 'performance.txt'
    file_path1 = 'auc.txt'
    file_path2 = 'f1.txt'
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，则删除文件
        os.remove(file_path)
    if os.path.exists(file_path1):
        # 如果文件存在，则删除文件
        os.remove(file_path1)
    if os.path.exists(file_path2):
        # 如果文件存在，则删除文件
        os.remove(file_path2)

    start_time = time.time()
    for k in ['tiny-imagenet-1000-3-0.08-0.02-0.05', 'tiny-imagenet-1000-3-0.05-0.02-0.08']:
            start_epoch_time = time.time()
            total_auc = []
            total_f1 = []
            for i in range(3):
                albation = [(1, 0.25, 1, 0.25)]
                for albation_set in albation:
                    auc_once, f1_score = run(utils.data_root_path, k, 100, 0.0001,
                                             6, 1, albation_set, 1, 200,
                                             num_views=3)
                    total_auc.append(auc_once)
                    total_f1.append(f1_score)
            total_auc = np.array(total_auc)
            total_f1 = np.array(total_f1)
            print('=============================================================================')
            print(k)
            print(np.mean(total_auc), np.mean(total_f1))
            with open(file_path, 'a') as file:
                file.write('=============================================================================\n')
                file.write(str(k) + '\n')
                file.write(str(np.mean(total_auc)) + '\n')
                file.write(str(np.mean(total_f1)) + '\n')
            all_epoch_time = time.time() - start_epoch_time
            print(all_epoch_time)
    all_time = time.time() - start_time
