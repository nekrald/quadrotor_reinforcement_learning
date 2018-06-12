import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Block configs.


class AbstractBlockConfig(object):

    def __init__(self):
        pass

    def get_type(self):
        raise NotImplementedError("Abstract class!")

    def get_last_fc_size(self):
        raise NotImplementedError("Abstract class!")


class ConvBlockConfig(AbstractBlockConfig):

    def __init__(self, in_count=4,
            out_counts=[16, 32, 32],
            strides=[2, 2, 2],
            kernel_sizes = [5, 5, 5],
            use_bn=True,
            linear_out=100,
            nonlinearity=F.relu,
            nonl_out=F.relu):

        self.in_count = in_count
        self.out_counts = out_counts
        self.strides = strides
        self.use_bn = use_bn
        self.linear_out = linear_out
        self.kernel_sizes = kernel_sizes
        self.nonlinearity = nonlinearity
        self.nonl_out = nonl_out

    def get_type(self):
        return "conv"

    def get_last_fc_size(self):
        return self.linear_out


class FeedForwardBlockConfig(BlockConfig):

    def __init__(self, in_size=None,
            hidden_sizes=[100, 100],
            nonlinearity=F.relu):
        self.in_size = in_size
        self.hiden_sizes = [100, 100]
        self.nonlinearity = F.relu

    def get_type(self):
        return "feedforward"

    def get_last_fc_size(self):
        return self.hiden_sizes[-1]


# Block descriptions.


class FeedForwardBlock(nn.Module):
    def __init__(self, config: FeedForwardBlockConfig):
        super(FeedForwardBlock, self).__init__()
        self.config = config
        self.hidden_sizes = config.hidden_sizes
        in_size = config.in_size
        self.fcs = []
        self.nls = []
        for hidden in self.config.hidden_sizes:
            if in_size is None:
                self.fcs.append(None)
            else:
                self.fcs.append(nn.Linear(in_size, hidden))
            self.nls.append(config.nonlinearity)
            in_size = hidden

    def forward(self, x):
        last_element = x
        for fc, nl in zip(self, fcs, self.nls):
            if fc is None:
                hidden = self.hidden_sizes[0]
                in_sz = x.view(x.size(0), -1).size(1)
                assert fcs[0] is None
                fcs[0] = nn.Linear(in_sz, hidden)
                fc = fcs[0]
            last_element = nl(fc(last_element))
        return last_element


class ConvBlock(nn.Module):

    def __init__(self, config: ConvBlockConfig):

        super(ConvBlock, self).__init__()
        self.config = config
        self.convs = []
        self.bns = []
        self.nls = []
        self.last_nl = config.nonl_out

        last_count = config.in_count
        for count, strd, ksz in zip(
                config.out_counts,
                config.strides,
                config.kernel_sizes):
            self.convs.append(nn.Conv2d(
                last_count, count, kernel_size=ksz, stride=strd))
            bn = None
            if config.use_bn:
                bn = nn.BatchNorm2d(count)
            self.bns.append(bn)
            self.nls.append(config.nonlinearity)
            last_count = count
        self.linear_size = config.linear_out
        self.fc_out = None

    def forward(self, x):
        last_element = x
        for conv, bn, nonl in zip(self.convs, self.bns, self.nls):
            last_element = conv(last_element)
            if bn is not None:
                last_element = bn(last_element)
            last_element = nonl(last_element)
        if self.fc_out is None:
            last_size = last_element.view(
                    last_element.size(0), -1).size(1)
            self.fc_out = nn.Linear(
                    last_size, self.linear_size)
        last_element = self.fc_out(last_element)
        if self.last_nl is not None:
            last_element = self.last_nl(last_element)
        return last_element


# Network description


def NetworkConfig(object):

    def __init__(self, part_sizes=28224,
            reshape_queries=[1, 4, 84, 84],
            block_configs=ConvBlockConfig(),
            last_fc_sizes=[],
            last_nls=[],
            head_size=8):
        self.part_sizes = part_sizes
        self.reshape_queries = reshape_queries
        self.block_configs = block_configs

        self.last_fc_sizes = last_fc_sizes
        self.last_nls = last_nls

        self.head_size = head_size


def Network(nn.Module):

    def __init__(self, config: NetworkConfig):
        super(Network, self).__init__()
        self.config = config
        self.shapes = config.reshape_queries
        assert len(config.parts_sizes) == len(config.reshape_queries)
        assert len(config.part_sizes) == len(config.block_configs)
        assert len(part_sizes) == 1, \
                "by now only one part is supported"
        assert config.block_configs[0].get_type == "conv", \
                "by now only conv block to be processed"
        self.blocks = [ConvBlock(config.block_configs[0])]
        out_linear_shape = config.block_configs[0].get_last_fc_size()
        self.fc_intermediate = []
        self.nl_intermediate = []
        last_sz = self.out_linear_shape
        for sz_fc, nl in zip(self.last_fc_sizes, self.last_nls):
            self.fc_intermediate.append(nn.Linear(last_sz, sz_fc))
            self.nl_intermediate.append(nl)
            last_sz = sz_fc
        self.out_linear = nn.Linear(last_sz, head_size)
        return self.out_linear

    def forward(self, x):
        assert len(self.shapes) == 1, "only one shape is not supported"
        shape = self.shapes[0]
        block = self.blocks[0]
        result = block.forward(x.view(*shape))
        for fc, nl in zip(self.fc_intermediate, self.nl_intermediate):
            result = nl(fc(result))
        result = self.out_linear(result)
        return result


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

