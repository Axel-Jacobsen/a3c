import torch

from torch import nn


class ConvNet(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential()

