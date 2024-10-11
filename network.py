import torch.nn as nn
import utils


class GeneratorEncode(nn.Module):

    def __init__(self, data_dim):
        super().__init__()
        self.rep_dim = 32
        self.data_dim = data_dim
        if utils.DATA_DIM == 12288:
            self.rep_dim = 512
            en_layers_num = [self.data_dim, 2048, 1024, self.rep_dim]  # s
        self.encoder = self.encode(en_layers_num)

    def encode(self, layers_num):
        if len(layers_num) > 2:
            encoded_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Tanh())
            for i in range(1, len(layers_num) - 2):
                encoded_output = nn.Sequential(encoded_output, nn.Linear(layers_num[i], layers_num[i + 1]), nn.Tanh())
            encoded_output = nn.Sequential(encoded_output, nn.Linear(layers_num[len(layers_num) - 2],
                                                                     layers_num[len(layers_num) - 1]))
        else:
            encoded_output = nn.Sequential(nn.Linear(layers_num[0], self.rep_dim), nn.Tanh())
        return encoded_output

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class GeneratorDecode(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.data_dim = data_dim
        if utils.DATA_DIM == 12288:
            self.rep_dim = 512
            en_layers_num = [self.data_dim, 2048, 1024, self.rep_dim]  # s
        de_layers_num = list(reversed(en_layers_num))
        self.decoder = self.decode(de_layers_num)

    def decode(self, layers_num):
        if len(layers_num) > 2:
            decode_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Tanh())
            for i in range(1, len(layers_num) - 2):
                decode_output = nn.Sequential(decode_output, nn.Linear(layers_num[i], layers_num[i + 1]), nn.Tanh())
            decode_output = nn.Sequential(decode_output, nn.Linear(layers_num[len(layers_num) - 2],
                                                                   layers_num[len(layers_num) - 1]), nn.Sigmoid())
        else:
            decode_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Sigmoid())
        return decode_output

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Prediction(nn.Module):

    def __init__(self, data_dim):
        super().__init__()
        self.rep_dim = 32
        self.data_dim = data_dim
        if utils.DATA_DIM == 12288:
            self.rep_dim = 512
            en_layers_num = [self.rep_dim, self.rep_dim, self.rep_dim]  # s
        self.encoder = self.encode(en_layers_num)

    def encode(self, layers_num):
        if len(layers_num) > 2:
            encoded_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Tanh())
            for i in range(1, len(layers_num) - 2):
                encoded_output = nn.Sequential(encoded_output, nn.Linear(layers_num[i], layers_num[i + 1]), nn.Tanh())
            encoded_output = nn.Sequential(encoded_output, nn.Linear(layers_num[len(layers_num) - 2],
                                                                     layers_num[len(layers_num) - 1]))
        else:
            encoded_output = nn.Sequential(nn.Linear(layers_num[0], self.rep_dim), nn.Tanh())
        return encoded_output

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
