#! /usr/bin/env python3

import os
import time

import gym
import torch

from torch.distributions.categorical import Categorical

from model import ActorCriticNet


def test_model(model_file: str):
    net = ActorCriticNet(4, 2)
    net.load_state_dict(torch.load(model_file))
    net.eval()

    env = gym.make("CartPole-v1")
    observation = env.reset()

    R = 0
    while True:
        env.render()
        time.sleep(0.05)
        cleaned_observation = torch.tensor(observation).unsqueeze(dim=0)
        action_logits = net.forward_actor(cleaned_observation)
        action = Categorical(logits=action_logits).sample()
        observation, r, done, _ = env.step(action.item())
        R += r
        if done:
            break
    env.close()
    print(R)


def select_pth():
    files = os.listdir("pth")
    print(sorted(files)[-1])
    return "pth/" + sorted(files)[-1]


if __name__ == "__main__":
    test_model(select_pth())
