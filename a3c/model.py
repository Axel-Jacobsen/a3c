import torch

from torch import nn


class ActorCriticNet(nn.Module):

    def __init__(self, in_dim, out_dim, n_actions):
        super(ConvNet, self).__init__()
        self.linear1 = nn.Linear(in_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.critic  = nn.Linear(128, 1)
        self.softmax = nn.Softmax(n_actions)

    def forward_critic(self, x):
        out = nn.ReLU(self.linear1(x))
        out = nn.ReLU(self.linear2(out))
        out = nn.ReLU(self.linear3(out))
        return self.critic(out)

    def forward_policy(self, x):
        out = nn.ReLU(self.linear1(x))
        out = nn.ReLU(self.linear2(out))
        return self.softmax(out)

