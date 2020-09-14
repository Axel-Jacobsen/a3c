import torch

from torch import nn
from torch.nn import functional as F


class ActorCriticNet(nn.Module):
    def __init__(self, in_dim, n_actions):
        super(ActorCriticNet, self).__init__()
        self.linear1 = nn.Linear(in_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, n_actions)
        self.critic = nn.Linear(32, 1)

    def forward_critic(self, x):
        out = F.relu(self.linear1(x.float()))
        out = F.relu(self.linear2(out))
        return self.critic(out)

    def forward_policy(self, x):
        out = F.relu(self.linear1(x.float()))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        return F.softmax(out, dim=-1)
