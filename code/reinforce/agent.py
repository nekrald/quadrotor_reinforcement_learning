import gym
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


env = gym.make("CartPole-v0").env
env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape

plt.imshow(env.render("rgb_array"))

agent = nn.Sequential(nn.Linear(state_dim[0], 100), nn.ReLU(), nn.Linear(100, n_actions) )

def predict_proba(states):
    """
    Predict action probabilities given states.
    :param states: numpy array of shape [batch, state_shape]
    :returns: numpy array of shape [batch, n_actions]
    """
    states = Variable(torch.FloatTensor(states))
    probas = F.softmax(agent.forward(states))
    return probas.data.numpy()

def generate_session(t_max=1000):
    """
    Play a full session with REINFORCE agent and train at the session end.
    Returns sequences of states, actions and rewards.
    """
    #arrays to record session
    states,actions,rewards = [],[],[]
    s = env.reset()
    for t in range(t_max):
        #action probabilities array aka pi(a|s)
        action_probas = predict_proba(np.array([s]))[0]
        a = np.random.choice(n_actions, p=action_probas)
        new_s,r,done,info = env.step(a)

        #record session history to train later
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = new_s
        if done: break
    return states, actions, rewards


states, actions, rewards = generate_session()

def get_cumulative_rewards(rewards, #rewards at each step
                           gamma = 0.99 #discount for reward
                           ):
    """
    take a list of immediate rewards r(s,a) for the whole session
    compute cumulative returns (a.k.a. G(s,a) in Sutton '16)
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    The simple way to compute cumulative rewards is to iterate from last to first time tick
    and compute G_t = r_t + gamma*G_{t+1} recurrently

    You must return an array/list of cumulative rewards with as many elements as in the initial rewards.
    """
    G = [rewards[-1]]
    for r in rewards[-2::-1]:
        G.append(r + gamma * G[-1])
    return G[::-1]


def to_one_hot(y, n_dims=None):
    """ Take an integer vector (tensor of variable) and convert it to 1-hot matrix. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


# Your code: define optimizers
opt = torch.optim.Adam(agent.parameters())

def train_on_session(states, actions, rewards, gamma = 0.99):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """

    # cast everything into a variable
    states = Variable(torch.FloatTensor(states))
    actions = Variable(torch.IntTensor(actions))
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = Variable(torch.FloatTensor(cumulative_returns))

    # predict logits, probas and log-probas using an agent.
    logits = agent.forward(states)
    probas = F.softmax(logits)
    logprobas = F.log_softmax(logits)

    assert all(isinstance(v, Variable) for v in [logits, probas, logprobas]), \
        "please use compute using torch tensors and don't use predict_proba function"

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    logprobas_for_actions = torch.sum(logprobas * to_one_hot(actions), dim = 1)

    # REINFORCE objective function
    J_hat = torch.mean(logprobas_for_actions * cumulative_returns)

    #regularize with entropy
    entropy_reg = - (probas * logprobas).sum(-1).mean()

    loss = - J_hat - 0.1 * entropy_reg

    # Gradient descent step
    loss.backward()
    opt.step()
    opt.zero_grad()

    # technical: return session rewards to print them later
    return np.sum(rewards)

# The actual training:
for i in range(100):

    rewards = [train_on_session(*generate_session()) for _ in range(100)] #generate new sessions

    print ("mean reward:%.3f"%(np.mean(rewards)))

    if np.mean(rewards) > 500:
        print ("You Win!") # but you can train even further
        break


