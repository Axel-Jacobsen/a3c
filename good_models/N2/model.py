import torch

from torch import nn
from torch.nn import functional as F


class ActorCriticNet(nn.Module):
    def __init__(self, in_dim, n_actions, training=False):
        super(ActorCriticNet, self).__init__()
        self.training = training
        self.linear1 = nn.Linear(in_dim, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward_critic(self, x):
        out = F.relu(self.linear1(x.float()))
        out = F.dropout(out, training=self.training)
        out = F.relu(self.linear2(out))
        return self.critic(out)

    def forward_actor(self, x):
        out = F.relu(self.linear1(x.float()))
        out = F.dropout(out, training=self.training)
        out = F.relu(self.linear2(out))
        out = F.dropout(out, training=self.training)
        out = F.relu(self.actor(out))
        return F.softmax(out, dim=-1)
