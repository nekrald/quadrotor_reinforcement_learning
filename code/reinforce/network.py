import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def make_network_config(network_json_config):
    raise NotImplementedError


def make_network(network_config):
    raise NotImplementedError


# Network description
def NetworkConfig(object):

    def __init__(self,
            head_config,
            tail_config_array,
            unpack_description=None,
            network_type=None):
        self.head_config = head_config
        self.tail_config_array = tail_config_array
        self.unpack_description = unpack_description
        self.network_type = network_type


# Network itself.


class Network(nn.Module):

    def __init__(self, network_config):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def _make_tails(self, x):
        raise NotImplementedError

    def _make_head(self, tails):
        raise NotImplementedError


# Optimizers


class OptimizerConfig(object):

    def __init__(self, optimizer_name="adam", lr="0.001"):
        self.optimizer_name = optimizer_name
        self.lr = lr


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

