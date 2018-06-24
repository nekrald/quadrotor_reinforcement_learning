import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def make_head_config(head_json_config):
    raise NotImplementedError


def make_head_module(head_config):
    raise NotImplementedError


# Head Configs.


class IHeadConfig(object):

    def __init__(self, head_type):
        raise NotImplementedError
        self.head_type = head_type
        pass


class HeadNarrowingFCConfig(IHeadConfig):

    def __init__(self):
        raise NotImplementedError


class HeadLSTMConfig(IHeadConfig):

    def __init__(self):
        raise NotImplementedError


# Head Blocks.


class IHeadBlock(nn.Module):

    def __init__(self, config: IHeadConfig):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class HeadLSTM(IHeadBlock):

    def __init__(self, config: HeadLSTMConfig):
        raise NotImplementedError


def HeadNarrowingFC(IHeadBlock):

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
        self.blocks = [ ConvBlock(config.block_configs[0]) ]
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


