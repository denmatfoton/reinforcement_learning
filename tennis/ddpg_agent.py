import numpy as np
import random

from rl_utils.memory import PrioritizedReplayBuffer, SimpleReplayBuffer
from rl_utils.noise import OUNoise, NormalNoise
from ddpg_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024       # mini batch size
GAMMA = 0.9             # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
TRAIN_EVERY = 4         # How many iterations to wait before starting training
TRAIN_ITERATIONS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, num_agents, seed):
        """Initialize an Agent object.
        
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        print("Using device: " + str(device))

        # Actor Network
        actor_hidden_layers = [64, 32]
        self.actor_local = Actor(state_size, action_size, seed, actor_hidden_layers).to(device)
        self.actor_target = Actor(state_size, action_size, seed, actor_hidden_layers).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        critic_hidden_layers = [64, 32]
        self.critic_local = Critic(state_size, action_size, seed, critic_hidden_layers).to(device)
        self.critic_target = Critic(state_size, action_size, seed, critic_hidden_layers).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), seed, sigma=0.2)
        # self.noise = OUNoise(action_size, seed)

        # Replay memory
        # self.memory = SimpleReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.step_num = 0

    @staticmethod
    def get_hyper_params():
        return "BUFFER_SIZE: {}\nBATCH_SIZE: {}\nGAMMA: {}\nTAU: {}\nLR_ACTOR: {}\nLR_CRITIC: {}".format(
            BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC)

    # step for a set of agents
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        if type(self.memory) == PrioritizedReplayBuffer:
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self.memory.add(1000, state, action, reward, next_state, done)
            # self.memory.add(1000, states, actions, rewards, next_states, dones)
        else:
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self.memory.add(state, action, reward, next_state, done)
            # self.memory.add(states, actions, rewards, next_states, dones)

        self.step_num = (self.step_num + 1) % TRAIN_EVERY
        if self.step_num == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for _ in range(TRAIN_ITERATIONS):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, sigma=0.):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if sigma != 0.:
            action += self.noise.sample()
            # action += np.random.normal(0, sigma, action.shape)
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if type(self.memory) == PrioritizedReplayBuffer:
            batch, idxs, is_weights = experiences
        else:
            batch = experiences
        states, actions, rewards, next_states, dones = extract(batch)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next).detach()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        if type(self.memory) == PrioritizedReplayBuffer:
            errors = torch.abs(Q_expected - Q_targets).mean(1).cpu().data.numpy()
            for idx, error in zip(idxs, errors):
                self.memory.update(idx, error)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()
        
        # ------------------- update target networks ------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def extract(batch):
    states = torch.from_numpy(np.vstack([e[0] for e in batch])).float().to(device)
    actions = torch.from_numpy(np.vstack([e[1] for e in batch])).float().to(device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in batch])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in batch])).float().to(device)
    dones = torch.from_numpy(np.vstack([e[4] for e in batch]).astype(np.uint8)).float().to(device)
    return states, actions, rewards, next_states, dones
