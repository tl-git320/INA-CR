import logging
import time
import torch
import torch.optim as optim
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils


class MyTrainer:

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    def train(self, id_round, id_view, dataset, en_nets, de_view_net, prediction_net, pre_train, pre_epochs,
              module_weight,
              albation_set, datasets):
        logger = logging.getLogger()
        # Set device for network
        en_nets1 = en_nets
        en_nets = []
        for en_view_net1 in en_nets1:
            en_nets.append(en_view_net1.to(self.device))
        en_view_net = en_nets[id_view]
        de_view_net = de_view_net.to(self.device)
        prediction_net = prediction_net.to(self.device)
        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.n_jobs_dataloader, drop_last=False)  #

        # Set optimizer (Adam optimizer for now)
        de_all_parameters = de_view_net.parameters()
        prediction_net_parameters = prediction_net.parameters()

        de_optimizer = optim.Adam(de_all_parameters, lr=self.lr, weight_decay=self.weight_decay,
                                  amsgrad=self.optimizer_name == 'amsgrad')
        prediction_net_optimizer = optim.Adam(prediction_net_parameters, lr=self.lr, weight_decay=self.weight_decay,
                                              amsgrad=self.optimizer_name == 'amsgrad')
        en_optimizer_list = []
        for en_view_net1 in en_nets:
            en_all_parameters1 = en_view_net1.parameters()
            en_optimizer1 = optim.Adam(en_all_parameters1, lr=self.lr, weight_decay=self.weight_decay,
                                       amsgrad=self.optimizer_name == 'amsgrad')
            en_optimizer_list.append(en_optimizer1)
        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(de_optimizer, milestones=self.lr_milestones,
                                                   gamma=0.1)

        # Training
        start_time = time.time()
        de_view_net.train()
        prediction_net.train()
        for en_view_net1 in en_nets:
            en_view_net1.train()
        num_epochs = self.n_epochs
        if pre_train and id_round == 0:
            num_epochs = pre_epochs

        for epoch in range(num_epochs):
            # time.sleep(1)

            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            loss_epoch = 0.0  # 本轮loss
            n_batches = 0  # natch轮数计数器
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, index, neighbor_local_list, id_local_neighbor_list, neighbor_global_list, id_global_neighbor = data
                inputs_tensor = torch.zeros(len(datasets), len(inputs), len(inputs[0]), device=self.device)
                for ide, dataset1 in enumerate(datasets):
                    for ind, inde in enumerate(index):
                        inputs_tensor[ide][ind], _, _, _, _, _, _ = dataset1.__getitem__(inde)
                inputs = inputs.to(self.device)
                neighbor_global_list = neighbor_global_list.to(self.device)

                for en_optimizer1 in en_optimizer_list:
                    en_optimizer1.zero_grad()
                de_optimizer.zero_grad()
                prediction_net_optimizer.zero_grad()

                encoded = en_view_net(inputs)
                encoded_neighbor_global_list = en_view_net(neighbor_global_list)
                encoded_neighbor_global_list = torch.mean(encoded_neighbor_global_list, dim=1)
                outputs = de_view_net(encoded)
                # 重构
                recon_error = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim()))) / len(
                    inputs)

                # 跨视图推理
                mse_tensor1 = torch.zeros(len(en_nets), len(inputs), len(encoded[0]), device=self.device)
                for i, en_view_net1 in enumerate(en_nets):
                    lantent1 = en_view_net1(inputs_tensor[i])
                    mse_tensor1[i] = lantent1
                mse_tensor2 = mse_tensor1.transpose(0, 1)
                mse_tensor3 = torch.zeros(len(inputs), len(encoded[0]), device=self.device)
                for i, mse_tensor33 in enumerate(mse_tensor2):
                    xxxxxxx = torch.mean(mse_tensor33, dim=0)
                    mse_tensor3[i] = xxxxxxx
                prediction_loss = torch.sum((mse_tensor3 - prediction_net(encoded)) ** 2,
                                            dim=tuple(range(1, encoded.dim()))) / len(encoded[0])
                # 实例对比学习
                ins_sum = torch.zeros(len(inputs), device=self.device)
                for j, mse_tensor11 in enumerate(mse_tensor1):
                    aaa = torch.exp(F.cosine_similarity(encoded, mse_tensor11, dim=1))
                    bbb = len(mse_tensor1[id_view]) * (
                        torch.exp(F.cosine_similarity(encoded, torch.mean(mse_tensor1[j], dim=0), dim=1)))
                    ins_sum += -torch.log(aaa / bbb)
                ins_sum /= len(en_nets) * len(encoded[0])

                # 邻居对比学习
                nei_sum = torch.zeros(len(inputs), device=self.device)
                for j, mse_tensor11 in enumerate(mse_tensor1):
                    aaa = torch.exp(F.cosine_similarity(encoded_neighbor_global_list, mse_tensor11, dim=1))
                    bbb = len(mse_tensor1[id_view]) * (torch.exp(
                        F.cosine_similarity(encoded_neighbor_global_list, torch.mean(mse_tensor1[j], dim=0),
                                            dim=1))) + torch.exp(
                        F.cosine_similarity(encoded_neighbor_global_list, torch.mean(encoded, dim=0), dim=1))
                    nei_sum += -torch.log(aaa / bbb)
                nei_sum /= len(en_nets) * len(encoded[0])

                scores = albation_set[0] * recon_error + albation_set[1] * nei_sum + albation_set[
                        2] * prediction_loss + albation_set[3] * ins_sum

                loss = torch.mean(scores)
                loss.backward()
                for en_optimizer1 in en_optimizer_list:
                    en_optimizer1.step()
                de_optimizer.step()
                prediction_net_optimizer.step()
                scheduler.step()
                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print(' View {} Round {} Epoch {}/{}\t Time: {:.3f}\t  GLoss: {:.8f} '.format(id_view + 1,
                                                                                          id_round,
                                                                                          epoch + 1,
                                                                                          num_epochs,
                                                                                          epoch_train_time,
                                                                                          loss_epoch / n_batches))

        view_encoded, recon_error, nei_sum, ins_sum, prediction_loss = test(self, id_view, dataset, en_nets,
                                                                                        de_view_net, prediction_net,
                                                                                        datasets)

        train_time = time.time() - start_time
        logger.info('View {} Training time: {:.3f}'.format(id_view + 1, train_time))
        # logger.info('Finished training.')

        return en_nets, de_view_net, view_encoded, recon_error, nei_sum, ins_sum, prediction_loss


def test(self, id_view, dataset, en_nets, de_view_net, prediction_net, datasets):  # 重建误差
    logger = logging.getLogger()

    # Set device for network
    en_nets1 = en_nets
    en_nets = []

    for en_view_net1 in en_nets1:
        en_nets.append(en_view_net1.to(self.device))
    en_view_net = en_nets[id_view]
    de_view_net = de_view_net.to(self.device)
    prediction_net = prediction_net.to(self.device)
    # Get test data loader
    test_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                             num_workers=self.n_jobs_dataloader, drop_last=False)

    # Testing
    logger.info('View {} Testing lg_ae...'.format(id_view + 1))
    loss_epoch = 0.0

    n_batches = 0
    start_time = time.time()
    idx_label_score = []
    for en_view_net1 in en_nets:
        en_view_net1.eval()
    de_view_net.eval()
    prediction_net.eval()
    encoded_data = None

    with torch.no_grad():
        for data in test_loader:

            inputs, _, idx, _, _, neighbor_global_list, id_global_neighbor = data
            inputs = inputs.to(self.device)
            neighbor_global_list = neighbor_global_list.to(self.device)
            inputs_tensor = torch.zeros(len(datasets), len(inputs), len(inputs[0]), device=self.device)
            for ide, dataset1 in enumerate(datasets):
                for ind, inde in enumerate(idx):
                    inputs_tensor[ide][ind], _, _, _, _, _, _ = dataset1.__getitem__(inde)

            encoded = en_view_net(inputs)
            encoded_neighbor_global_list = en_view_net(neighbor_global_list)
            encoded_neighbor_global_list = torch.mean(encoded_neighbor_global_list, dim=1)
            outputs = de_view_net(encoded)
            # outputs = de_view_net(encoded)

            # 重构
            recon_error = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))

            # 跨视图推理
            mse_tensor1 = torch.zeros(len(en_nets), len(inputs), len(encoded[0]), device=self.device)
            for i, en_view_net1 in enumerate(en_nets):
                lantent1 = en_view_net1(inputs_tensor[i])
                mse_tensor1[i] = lantent1
            mse_tensor2 = mse_tensor1.transpose(0, 1)
            mse_tensor3 = torch.zeros(len(inputs), len(encoded[0]), device=self.device)
            for i, mse_tensor33 in enumerate(mse_tensor2):
                xxxxxxx = torch.mean(mse_tensor33, dim=0)
                mse_tensor3[i] = xxxxxxx
            prediction_loss = torch.sum((mse_tensor3 - prediction_net(encoded)) ** 2,
                                        dim=tuple(range(1, encoded.dim()))) / len(encoded[0])
            # #实例对比学习
            ins_sum = torch.zeros(len(inputs), device=self.device)
            for j, mse_tensor11 in enumerate(mse_tensor1):
                aaa = torch.exp(F.cosine_similarity(encoded, mse_tensor11, dim=1))
                bbb = len(mse_tensor1[id_view]) * (torch.exp(
                    F.cosine_similarity(encoded, torch.mean(mse_tensor1[j], dim=0), dim=1)))
                ins_sum += -torch.log(aaa / bbb)
            ins_sum /= len(en_nets) * len(encoded[0])

            # 邻居对比学习
            nei_sum = torch.zeros(len(inputs), device=self.device)
            for j, mse_tensor11 in enumerate(mse_tensor1):
                bbb = torch.zeros(len(inputs), device=self.device)
                aaa = torch.exp(F.cosine_similarity(encoded_neighbor_global_list, mse_tensor11, dim=1))
                bbb = len(mse_tensor1[id_view]) * (torch.exp(
                    F.cosine_similarity(encoded_neighbor_global_list, torch.mean(mse_tensor1[j], dim=0),
                                        dim=1))) + torch.exp(
                    F.cosine_similarity(encoded_neighbor_global_list, torch.mean(encoded, dim=0), dim=1))
                nei_sum += -torch.log(aaa / bbb)
            nei_sum /= len(en_nets) * len(encoded[0])

            if n_batches == 0:
                encoded_data = encoded.cpu().data.numpy()
            else:
                encoded_data = np.concatenate((encoded_data, encoded.cpu().numpy()), axis=0)

            # Save triple of (idx, label, score) in a list
            idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                        recon_error.cpu().data.numpy().tolist(),
                                        nei_sum.cpu().data.numpy().tolist(),
                                        ins_sum.cpu().data.numpy().tolist(),
                                        prediction_loss.cpu().data.numpy().tolist()))
            n_batches += 1

    _, recon_error, nei_sum, ins_sum, prediction_loss = zip(*idx_label_score)
    recon_error = np.array(recon_error)
    nei_sum = np.array(nei_sum)
    ins_sum = np.array(ins_sum)
    prediction_loss = np.array(prediction_loss)

    return encoded_data, recon_error, nei_sum, ins_sum, prediction_loss
