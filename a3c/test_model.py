#! /usr/bin/env python3

import os

import gym
import torch

from torch.distributions.categorical import Categorical

from model import ActorCriticNet


def test_model(model_file: str):
    net = ActorCriticNet(4, 2)
    net.load_state_dict(torch.load(model_file))
    net.eval()

    env = gym.make("CartPole-v0")
    observation = env.reset()

    while True:
        env.render()
        cleaned_observation = torch.tensor(observation).unsqueeze(dim=0)
        action_logits = net.forward_policy(cleaned_observation)
        action = Categorical(logits=action_logits).sample()
        observation, r, done, _ = env.step(action.item())
        if done:
            break
    env.close()


def select_pth():
    files = os.listdir('pth')
    return 'pth/' + sorted(files)[-1]


if __name__ == "__main__":
    test_model(select_pth())

