#! /usr/bin/env python3

import gym
import torch
import numpy as np

from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from model import ActorCriticNet
from collections import namedtuple

"""Asynchronous Advantage Actor-Critic

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
review section 8 for further details """
NUM_THREADS = 8
I_update = 5
t_max = 250  # max individual number of frames
T_max = 40000
T = 0

BATCH_SIZE = 64
learning_rate = 1e-3
gamma = 0.99
BETA = 0.99


torch.autograd.set_detect_anomaly(True)


shared_net = ActorCriticNet(4, 2)


class Train:
    def __init__(self):
        # Below this is the work that should be done by each thread
        self.thread_net = ActorCriticNet(4, 2)
        self.thread_net.load_state_dict(shared_net.state_dict())

        self.value_loss_fcn = nn.MSELoss(reduction="sum")
        self.optimizer = optim.RMSprop(self.thread_net.parameters(), lr=learning_rate)

        self.env = gym.make("CartPole-v0")

    def play_episode(self):

        episode_actions = torch.empty(size=(0,), dtype=torch.long)
        episode_logits = torch.empty(
            size=(0, self.env.action_space.n), dtype=torch.long
        )
        episode_observs = torch.empty(
            size=(0, *self.env.observation_space.shape), dtype=torch.long
        )
        episode_rewards = np.empty(shape=(0,), dtype=np.float)

        observation = self.env.reset()

        t = 0
        while True:
            # Prepare observation
            cleaned_observation = torch.tensor(observation).unsqueeze(dim=0)
            episode_observs = torch.cat((episode_observs, cleaned_observation), dim=0)
            # Get action from policy net
            action_logits = self.thread_net.forward_policy(cleaned_observation)
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)

            action = Categorical(logits=action_logits).sample()
            # Save observation and the action from the net
            episode_actions = torch.cat((episode_actions, action), dim=0)
            # Get new observation and reward from action
            observation, r, done, _ = self.env.step(action.item())
            # Save reward from net_action
            episode_rewards = np.concatenate((episode_rewards, np.asarray([r])), axis=0)

            t += 1
            if done:
                break

        # Set initial reward to 0 if it was terminal state, otherwise criticize it
        R = 0 if done else thread_net.forward_critic(observation)

        discounted_R = self.get_discounted_rewards(episode_rewards, gamma)

        # normalize reward
        discounted_R -= episode_rewards.mean()

        mask = F.one_hot(episode_actions, num_classes=self.env.action_space.n)
        episode_log_probs = torch.sum(
            mask.float() * F.log_softmax(episode_logits, dim=1), dim=1
        )

        episode_weighted_log_probs = episode_log_probs * discounted_R.float()
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

        return (
            sum_weighted_log_probs,
            episode_logits,
            episode_observs,
            np.sum(episode_rewards),
            t,
        )

    def get_discounted_rewards(self, rewards: np.array, gamma: float) -> torch.Tensor:
        """
        Calculates the sequence of discounted rewards-to-go.
        Args:
            rewards: the sequence of observed rewards
            gamma: the discount factor
        Returns:
            discounted_rewards: the sequence of the rewards-to-go

        AXEL: Directly from
        https://towardsdatascience.com/breaking-down-richard-suttons-policy-gradient-9768602cb63b
        """
        discounted_rewards = np.empty_like(rewards, dtype=np.float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
        return torch.from_numpy(discounted_rewards)

    def calculate_policy_loss(
        self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor
    ):

        policy_loss = -torch.mean(weighted_log_probs)
        p = F.softmax(epoch_logits, dim=1)
        log_p = F.log_softmax(epoch_logits, dim=0)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=-1), dim=0)
        entropy_bonus = -1 * BETA * entropy
        return policy_loss + entropy_bonus, entropy

    def solve_env(self):
        episode = 0
        epoch = 0
        T = 0

        total_rewards = []
        epoch_logits = torch.empty(size=(0, self.env.action_space.n))
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float)

        while True:
            (
                episode_weighted_log_probs,
                episode_logits,
                episode_observs,
                total_episode_reward,
                t,
            ) = self.play_episode()

            T += t
            episode += 1
            total_rewards.append(total_episode_reward)
            epoch_weighted_log_probs = torch.cat(
                (epoch_weighted_log_probs, episode_weighted_log_probs), dim=0
            )

            if episode > BATCH_SIZE:

                episode = 0
                epoch += 1

                policy_loss, entropy = self.calculate_policy_loss(
                    epoch_logits=epoch_logits,
                    weighted_log_probs=epoch_weighted_log_probs,
                )

                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
                print(
                    f"Epoch: {epoch}, Avg Return per Epoch:"
                    f" {np.mean(total_rewards):.3f}"
                )

                # reset the epoch arrays
                # used for entropy calculation
                epoch_logits = torch.empty(size=(0, self.env.action_space.n))
                epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float)

                # check if solved
                if np.mean(total_rewards) > 200:
                    torch.save(self.thread_net.state_dict(), "model.pth")
                    print("\nSolved!")
                    break

        self.env.close()


# shared_net.load_state_dict(thread_net)

# # Get a vector of values for each observation
# predicted_R = thread_net.forward_critic(observs)
# value_loss = value_loss_fcn(discounted_R, predicted_R)

# optimizer.zero_grad()
# value_loss.backward()
# optimizer.step()

if __name__ == "__main__":
    trainer = Train()

    try:
        trainer.solve_env()
    finally:
        import time

        torch.save(trainer.thread_net.state_dict(), f"model_{time.time()}.pth")
