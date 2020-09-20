#! /usr/bin/env python3

import os
import sys
import gym
import torch
import numpy as np
import torch.multiprocessing as mp

from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from model import ActorCriticNet

from collections import namedtuple

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

""" Asynchronous Advantage Actor-Critic

Following a lot of the following link because i am a pytorch noob
https://towardsdatascience.com/breaking-down-richard-suttons-policy-gradient-9768602cb63b

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


BATCH_SIZE = 512
BETA = 0.99
GAMMA = 0.99
LEARNING_RATE = 1e-3
VALUE_LOSS_CONSTANT = 0.6
SHARED_NET_UPDATE = 40000


def _select_pth():
    files = os.listdir("pth")
    return "./pth/" + sorted(files)[0]


class Trainer:
    def __init__(self, num_procs):
        self.num_procs = num_procs
        self.global_net = ActorCriticNet(4, 2, training=True)
        self.optimizer = optim.RMSprop(self.global_net.parameters(), lr=LEARNING_RATE)
        self.grad_q = mp.Queue()
        self.global_state_q = mp.Queue()

    def spawn_processes(self):
        procs = []
        for i in range(self.num_procs):
            tp = TrainerProcess(self.global_net.state_dict())
            p = mp.Process(target=tp.train, args=(self.grad_q, self.global_state_q))
            p.start()
            procs.append(p)
        return procs

    def add_grads(self, grads):
        for i, p in enumerate(self.global_net.parameters()):
            p.grad += grads[i]

    def train(self):
        procs = self.spawn_processes()
        while len(mp.active_children()) > 0:
            i = 0
            while not self.grad_q.empty():
                i += 1
                self.add_grads(self.grad_q.get())
            self.optimizer.step()
            self.optimizer.zero_grad()
            for _ in range(i):
                global_params = self.global_net.state_dict().detach()
                self.global_state_q.put(global_params)


class TrainerProcess:
    def __init__(self, model_file: str = None):
        self.thread_net = ActorCriticNet(4, 2, training=True)
        self.thread_net.load_state_dict(model_file)
        self.thread_net.eval()

        self.optimizer = optim.RMSprop(self.thread_net.parameters(), lr=LEARNING_RATE)
        self.grad_accum = [torch.zeros(p.shape) for p in self.thread_net.parameters()]
        self.env = gym.make("CartPole-v1")

        print(f'Starting process...')
        sys.stdout.flush()

    def play_episode(self):
        episode_actions = torch.empty(size=(0,), dtype=torch.long)
        episode_logits = torch.empty(size=(0, self.env.action_space.n), dtype=torch.long)
        episode_observs = torch.empty(size=(0, *self.env.observation_space.shape), dtype=torch.long)
        episode_rewards = np.empty(shape=(0,), dtype=np.float)

        observation = self.env.reset()

        t = 0
        done = False
        while not done:
            # Prepare observation
            cleaned_observation = torch.tensor(observation).unsqueeze(dim=0)
            episode_observs = torch.cat((episode_observs, cleaned_observation), dim=0)

            # Get action from policy net
            action_logits = self.thread_net.forward_actor(cleaned_observation)
            action = Categorical(logits=action_logits).sample()

            # Save observation and the action from the net
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)
            episode_actions = torch.cat((episode_actions, action), dim=0)

            # Get new observation and reward from action
            observation, r, done, _ = self.env.step(action.item())

            # Save reward from net_action
            episode_rewards = np.concatenate((episode_rewards, np.asarray([r])), axis=0)

            t += 1

        discounted_R = self.get_discounted_rewards(episode_rewards, GAMMA)
        discounted_R -= episode_rewards.mean()

        mask = F.one_hot(episode_actions, num_classes=self.env.action_space.n)
        episode_log_probs = torch.sum(mask.float() * F.log_softmax(episode_logits, dim=1), dim=1)

        values = self.thread_net.forward_critic(episode_observs)
        action_advantage = (discounted_R.float() - values).detach()
        episode_weighted_log_probs = episode_log_probs * action_advantage
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)
        sum_action_advantages = torch.sum(action_advantage).unsqueeze(dim=0)

        return (
            sum_weighted_log_probs,
            sum_action_advantages,
            episode_logits,
            np.sum(episode_rewards),
            t,
        )

    def get_discounted_rewards(self, rewards: np.array, GAMMA: float) -> torch.Tensor:
        """
        Calculates the sequence of discounted rewards-to-go.
        Args:
            rewards: the sequence of observed rewards
            GAMMA: the discount factor
        Returns:
            discounted_rewards: the sequence of the rewards-to-go

        AXEL: Directly from
        https://towardsdatascience.com/breaking-down-richard-suttons-policy-gradient-9768602cb63b
        """
        discounted_rewards = np.empty_like(rewards, dtype=np.float)
        for i in range(rewards.shape[0]):
            GAMMAs = np.full(shape=(rewards[i:].shape[0]), fill_value=GAMMA)
            discounted_GAMMAs = np.power(GAMMAs, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_GAMMAs)
            discounted_rewards[i] = discounted_reward
        return torch.from_numpy(discounted_rewards)

    def calculate_policy_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor):
        policy_loss = -torch.mean(weighted_log_probs)
        p = F.softmax(epoch_logits, dim=1)
        log_p = F.log_softmax(epoch_logits, dim=0)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=-1), dim=0)
        entropy_bonus = -1 * BETA * entropy
        return policy_loss + entropy_bonus, entropy

    def train(self, grad_q, global_state_q):
        episode, epoch, T = 0, 0, 0

        total_rewards = []
        epoch_logits = torch.empty(size=(0, self.env.action_space.n))
        epoch_action_advantage = torch.empty(size=(0,))
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float)

        while True:
            (
                episode_weighted_log_probs,
                action_advantage_sum,
                episode_logits,
                total_episode_reward,
                t,
            ) = self.play_episode()

            T += t
            episode += 1
            total_rewards.append(total_episode_reward)
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_probs), dim=0)
            epoch_action_advantage = torch.cat((epoch_action_advantage, action_advantage_sum), dim=0)

            if episode > BATCH_SIZE:

                episode = 0
                epoch += 1

                policy_loss, entropy = self.calculate_policy_loss(
                    epoch_logits=epoch_logits, weighted_log_probs=epoch_weighted_log_probs,
                )
                value_loss = torch.square(epoch_action_advantage).mean()
                total_loss = policy_loss + VALUE_LOSS_CONSTANT * value_loss

                for i, grad in enumerate(self.thread_net.parameters()):
                    self.grad_accum[i] += grad

                self.optimizer.zero_grad()

                total_loss.backward()

                self.optimizer.step()

                print(f"{os.getpid()} Epoch: {epoch}, Avg Return per Epoch: {np.mean(total_rewards):.3f}")
                sys.stdout.flush()

                # reset the epoch arrays, used for entropy calculation
                epoch_logits = torch.empty(size=(0, self.env.action_space.n))
                epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float)

                if T > SHARED_NET_UPDATE:
                    grad_q.put(self.grad_accum.detach())
                    self.thread_net.load_state_dict(global_state_q.get())
                    self.thread_net.eval()

                # check if solved
                if np.mean(total_rewards) > 200:
                    print("\nSolved!")
                    break

        self.env.close()


if __name__ == "__main__":
    NUM_PROCS = 15
    trainer = Trainer(NUM_PROCS)
    try:
        trainer.train()
    finally:
        import time
        torch.save(trainer.global_net.state_dict(), f"pth/model_{time.time()}.pth")
