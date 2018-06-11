import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Block configs.
class BlockConfig(object):
    def __init__(self):
        pass
    def get_type(self):
        raise NotImplementedError


class ConvBlockConfig(BlockConfig):
    def __init__(self):
        self.in_count = 4
        self.out_counts = [16, 32, 32]
        self.strides = [2, 2, 2]
        self.use_bn = True
        self.linear_out = 100
    def get_type(self):
        return "conv"


class FeedForwardBlockConfig(BlockConfig):
    def __init__(self):
        self.input_shape = 4
        self.layer_sizes = [100, 100]
        self.nonlinearity = F.relu
    def get_type(self):
        return "feedforward"


# Block descriptions.


class FeedForwardBlock(nn.Module):
    def __init__(self, config: FeedForwardBlockConfig):
        super(FeedForwardBlock, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ConvBlock(nn.Module):

    def __init__(self, config: ConvBlockConfig):
        super(ConvBlock, self).__init__()
        raise NotImplementedError
        self.conv1 = nn.Conv2d(in_count, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head= nn.Linear(448, 2)

    def forward(self, x):
        raise NotImplementedError
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


# Network description


def NetworkConfig(object):
    def __init__(self):
        self.part_sizes = []
        self.reshape_queries = []
        self.block_configs = []
        self.join_layer_size = None
        self.last_layers = []
        self.head_size = []


def Network(nn.Module):
    def __init__(self, config: NetworkConfig):
        super(Network, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# Optimizers


def make_optimizer(optimizer_config, network):
    if optimizer_config.optimizer_type == "adam":
        return torch.optim.Adam(network.parameters())
    else:
        raise ValueError("Unknown optimizer_type.")


# Nonlinearities


def make_nonlinearity(nonlinearity_name):
    if nonlinearity_name == "relu":
        return F.relu
    else:
        raise ValueError("Unknown nonlinearity.")

