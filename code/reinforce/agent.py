import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from network import Network, NetworConfig, make_optimizer, \
        ConvBlockConfig, FeedForwardBlockConfig


class ConfigAgentREINFORCE(object):

    def __init__(self, request_config,
            network_config, n_actions, optimizer_config):
        self.request_config = request_config
        self.network_config = network_config
        self.opt_config = optimizer_config
        self.n_actions = n_actions


class AgentREINFORCE(object):

    def __init__(self, config: ConfigAgentREINFORCE):
        self.config = config

        self.request_config = config.request_config
        self.raw_provided = config.raw_result
        assert self.raw_provided, "Should be flat array."
        self.state_provided = config.provide_state
        self.sensor_provided = config.provide_sensor
        self.network_config = config.network_config

        self._build_network()
        self._build_optimizer()

        self.n_actions = config.n_actions

    def _build_network(self):
        self.network = Network(self.network_config)

    def _build_optimizer(self):
        self.opt = make_optimizer(
                self.optimizer_config, self.network)

    def predict_proba(self, states):
        """
        Predict action probabilities given states.
        :param states: numpy array of shape [batch, state_shape]
        :returns: numpy array of shape [batch, n_actions]
        """
        states = Variable(torch.FloatTensor(states))
        probas = F.softmax(self.network.forward(states))
        return probas.data.numpy()

    def to_one_hot(self, y, n_dims=None):
        """ Take an integer vector (tensor of variable)
        and convert it to 1-hot matrix. """
        y_tensor = y.data if isinstance(y, Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(
                torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(
                y_tensor.size()[0], n_dims).scatter_(
                        1, y_tensor, 1)
        return Variable(y_one_hot) if isinstance(
                y, Variable) else y_one_hot

    def train_on_session(self, states, actions,
            rewards, cumulative_rewards):
        """
        Takes a sequence of states, actions and rewards
        produced by generate_session.
        Updates network_agent's weights by following the
        policy gradient above.
        """
        states = Variable(torch.FloatTensor(states))
        actions = Variable(torch.IntTensor(actions))
        cumulative_returns = np.array(cumulative_rewards)
        cumulative_returns = Variable(
                torch.FloatTensor(cumulative_returns))

        logits = self.network.forward(states)
        probas = F.softmax(logits)
        logprobas = F.log_softmax(logits)

        assert all(isinstance(v, Variable) for v in [
                    logits, probas, logprobas]), \
            "please use compute using torch tensors" + \
            "and don't use predict_proba function"

        logprobas_for_actions = torch.sum(
                logprobas * to_one_hot(actions), dim = 1)

        J_hat = torch.mean(
                logprobas_for_actions * cumulative_returns)

        entropy_reg = - (probas * logprobas).sum(-1).mean()

        loss = - J_hat - 0.1 * entropy_reg

        loss.backward()
        opt.step()
        opt.zero_grad()

        return np.sum(rewards)

