
import gymnasium as gym
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import cardenv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# PATH = "torchrl/jsgame_model.pt"

# env = gym.make("CartPole-v1")
env = gym.make("JSCardGame-v0")
env = gym.wrappers.FlattenObservation(env)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online = self.__build_nn(input_dim, output_dim)

        self.target = self.__build_nn(input_dim, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_nn(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )



class Agent:
    def __init__(self, env, state_dim, action_dim, save_dir):
        self.env = env

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = Net(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.sync_every = 100  # no. of experiences between Q_target & Q_online sync
        self.curr_step = 0

        self.save_every = 1e4  # no. of experiences between saving

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action = torch.tensor(self.env.action_space.sample(), device=self.device, dtype=torch.long)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            action_values = self.net(state, model="online")
            # print(action_values)
            action = action_values

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action

    def cache(self, state, next_state, action, reward, done):
        """Add the experience to memory"""
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(done)

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """Sample experiences from memory"""
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        print(state)
        print(action)
        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def td_estimate(self, state, action):
        print(state.shape)
        print(action.shape)
        current_Q = self.net(state, model="online")  # Q_online(s,a)
        print(current_Q.shape)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"jsgame_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"JsGameNet saved to {save_path} at step {self.curr_step}")


class Logger():
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.writer = SummaryWriter(self.save_dir)

        self.episode_num = 0

    def log_episode(self, i_episode, reward):
        self.writer.add_scalar('reward',
                          reward,
                          i_episode)

        self.episode_num = i_episode


log_path = 'torchrl/runs/jscardgame'
checkpoint_dir = 'torchrl/checkpoints'

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

num_episodes = 1000
log_freq = 100

agent = Agent(env, state_dim=len(state), action_dim=env.action_space.n, save_dir=checkpoint_dir)

logger = Logger(log_path)

for e in range(num_episodes):

    state = env.reset()
    # Play the game!
    while True:

        # Run agent on the state
        action = agent.act(state)
        # print(action)
        # Agent performs action
        next_state, reward, terminated, trunc, info = env.step(action)
        done = terminated or trunc

        # Remember
        agent.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = agent.learn()

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break

    if (e % log_freq == 0) or (e == num_episodes - 1):
        logger.log_episode(e, reward)
