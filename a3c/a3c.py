#! /usr/bin/env python3

import gym

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
learing_rate = 1e-4
gamma = 0.99

observation_action_reward = namedtuple(
    "observation_action_reward", ("observation", "action", "reward")
)



# Below this is the work that should be done by each thread
value_loss = nn.MSELoss(reduction="sum")
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
shared_net = ActorCriticNet()

env = gym.make("CartPole-v0")

while T < T_max:
    """
    TODO:
        - How do we properly do a net which can have multiple different output layers
    """

    thread_net = ActorCriticNet()
    thread_net.load_state_dict(shared_net)

    data = []

    observation = env.reset()

    for t in range(t_max):
        net_action = Categorical(thread_net.forward_policy(observation)).sample()
        oar = observation_action_reward(observation, net_action, 0)
        observation, reward, done, _ = env.step(net_action)
        oar.reward = reward
        data.append(oar)

        if done:
            break
    T = t

    # Should be V(s,\theta)
    R = 0 if done else thread_net.forward_critic(observation)

    for oar in reversed(data):
        R = oar.reward + gamma * R
        loss = value_loss(y_pred, R)
        loss.backward()
        optimizer.zero_grad()
        optimzer.step()

    # Should be async update of global params
    shared_net.load_state_dict(thread_net)

