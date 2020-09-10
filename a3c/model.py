import torch

from torch import nn
from torch.nn import functional as f


class ActorCriticNet(nn.Module):
    def __init__(self, in_dim, n_actions):
        super(ActorCriticNet, self).__init__()
        self.linear1 = nn.Linear(in_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, n_actions)
        self.critic  = nn.Linear(32, 1)
        self.softmax = nn.Softmax(-1)

    def forward_critic(self, x):
        out = f.relu(self.linear1(x))
        out = f.relu(self.linear2(out))
        return self.critic(out)

    def forward_policy(self, x):
        out = f.relu(self.linear1(x))
        out = f.relu(self.linear2(out))
        out = f.relu(self.linear3(out))
        return self.softmax(out)
