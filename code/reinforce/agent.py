import gym
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(object):

    def __init__(self):
        raise NotImplementedError
        n_actions = env.action_space.n
        state_dim = env.observation_space.shape
        self.network_agent = nn.Sequential(
                nn.Linear(state_dim[0], 100),
                nn.ReLU(),
                nn.Linear(100, n_actions)
        )
        # After agent is created.
        self.opt = torch.optim.Adam(network_agent.parameters())

    def predict_proba(self, states):
        """
        Predict action probabilities given states.
        :param states: numpy array of shape [batch, state_shape]
        :returns: numpy array of shape [batch, n_actions]
        """
        states = Variable(torch.FloatTensor(states))
        probas = F.softmax(network_agent.forward(states))
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

    def train_on_session(
            self, states, actions, rewards, gamma = 0.99):
        """
        Takes a sequence of states, actions and rewards
        produced by generate_session.
        Updates network_agent's weights by following the
        policy gradient above.
        """

        # cast everything into a variable
        states = Variable(torch.FloatTensor(states))
        actions = Variable(torch.IntTensor(actions))
        cumulative_returns = np.array(
                get_cumulative_rewards(rewards, gamma))
        cumulative_returns = Variable(
                torch.FloatTensor(cumulative_returns))

        # predict logits, probas and
        # log-probas using an network_agent.
        logits = network_agent.forward(states)
        probas = F.softmax(logits)
        logprobas = F.log_softmax(logits)

        assert all(
                isinstance(v, Variable) for v in [
                    logits, probas, logprobas]), \
            "please use compute using torch tensors and don't use predict_proba function"

        # select log-probabilities for chosen actions, log pi(a_i|s_i)
        logprobas_for_actions = torch.sum(
                logprobas * to_one_hot(actions), dim = 1)

        # REINFORCE objective function
        J_hat = torch.mean(
                logprobas_for_actions * cumulative_returns)

        #regularize with entropy
        entropy_reg = - (probas * logprobas).sum(-1).mean()

        loss = - J_hat - 0.1 * entropy_reg

        # Gradient descent step
        loss.backward()
        opt.step()
        opt.zero_grad()

        # technical: return session rewards to print them later
        return np.sum(rewards)

