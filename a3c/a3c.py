#! /usr/bin/env python3

import gym
import torch

from torch import nn, optim
from torch.distributions.categorical import Categorical

from model import ActorCriticNet
from collections import namedtuple

"""Asynchronous Advantage Actor-Critic
optim: RMSProp
value based method (i.e. this) shared target network updated every 40000 frames
atari games used Mnih et al., 2015 preprocessing and action repeat of 4
network architecture of Mnih et al., 2013
  Conv 16 filters, 8x8, stride 4
  Conv 32 filters, 4x4, stride 2
  Fully Connected, 256 hidden
nonlinearity after each hidden layer is the rectifier (?)
actor critic had two set of outputs - softmax w/ one entry per action and single linear output representing the value function
discount = 0.99, RMSProp decay factor of 0.99
review section 8 for further details
"""

NUM_THREADS = 8
I_update = 5
t_max = 250  # max individual number of frames
T_max = 40000
T = 0

N = 64
learning_rate = 1e-4
gamma = 0.99

shared_net = ActorCriticNet(4, 2)

# Below this is the work that should be done by each thread
thread_net = ActorCriticNet(4, 2)
thread_net.load_state_dict(shared_net.state_dict())

value_loss_fcn = nn.MSELoss(reduction="sum")
optimizer = optim.RMSprop(thread_net.parameters(), lr=learning_rate)


env = gym.make("CartPole-v0")

while T < T_max:
    """
    TODO:
        - How do we properly do a net which can have multiple different output layers
    """
    data = []

    observation = env.reset()
    observs = torch.zeros(t_max, 4)
    actions = torch.zeros(t_max, 1)
    rewards = torch.zeros(t_max, 1)

    for t in range(t_max):
        # Prepare observation
        obs = torch.Tensor(observation).reshape(1,-1)
        # Get action from policy net
        ss = thread_net.forward_policy(obs)
        net_action = Categorical(ss).sample()
        # Save observation and the action from the net
        observs[t,:] = obs
        actions[t,:] = net_action
        # Get new observation and reward from action
        observation, r, done, _ = env.step(net_action.item())
        # Save reward from net_action
        rewards[t,] = r

        if done:
            break

    # Set global shared counter to final thread timestep
    T = t

    # Set initial reward to 0 if it was terminal state, otherwise criticize it
    R = 0 if done else thread_net.forward_critic(observation)

    accumulated_R = torch.Tensor(len(rewards))
    for i, reward in enumerate(reversed(rewards)):
        R = reward + gamma * R
        accumulated_R[t_max - i - 1] = R

    # normalize reward
    accumulated_R = (accumulated_R - accumulated_R.mean()) / (
        accumulated_R.std() + torch.finfo(torch.float32).eps
    )

    # Get a vector of values for each observation
    predicted_R = thread_net.forward_critic(observs)
    value_loss = value_loss_fcn(accumulated_R, predicted_R)
    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()

    policy_loss = torch.sum(-torch.mul(actions, accumulated_R), -1)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.zero_grad()

# Should be async update of global params
shared_net.load_state_dict(thread_net)
