import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def make_tail_config(tail_json_config):
    # 1. Create one of block configs described below.
    # 2. Then create a block based on the config.
    raise NotImplementedError

def make_tail_module(tail_config):
    raise NotImplementedError


# Block configs (for input blocks).
class ITailBlockConfig(object):

    def __init__(self, tail_type=None):
        self.tail_type = tail_type
        raise NotImplementedError("Abstract class!")

    def get_type(self):
        raise NotImplementedError("Abstract class!")

    def get_last_fc_size(self):
        raise NotImplementedError("Abstract class!")


class TailConvBlockConfig(ITailBlockConfig):

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


class TailFeedForwardBlockConfig(ITailBlockConfig):

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


class TailNarrowingFCBlockConfig(ITailBlockConfig):

    def __init__(self):
        raise NotImplementedError


# Block descriptions (for input blocks).


class ITailBlock(nn.Module):

    def __init__(self, config: ITailBlockConfig):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class TailNarrowingFCBlock(ITailBlock):

    def __init__(self, config: TailNarrowingFCBlockConfig):
        raise NotImplementedError


class TailFeedForwardBlock(ITailBlock):

    def __init__(self, config: TailFeedForwardBlockConfig):
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


class TailConvBlock(ITailBlock):

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


