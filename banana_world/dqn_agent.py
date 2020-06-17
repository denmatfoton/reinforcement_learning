import numpy as np
import random
from rl_utils.memory import *

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ENABLE_DDQN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
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

        # Q-Network
        hidden_layers = [64, 32]
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_layers).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_layers).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        # self.memory = SimpleReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def get_hyper_params(self):
        return "BUFFER_SIZE: {}\nBATCH_SIZE: {}\nGAMMA: {}\nTAU: {}\nLR: {}\nUPDATE_EVERY: {}".format(
            BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_EVERY)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if type(self.memory) == PrioritizedReplayBuffer:
            '''
            states = torch.from_numpy(state).float().to(device).unsqueeze(0)
            actions = torch.from_numpy(np.array([action])).long().to(device).unsqueeze(0)
            rewards = torch.from_numpy(np.array([reward])).float().to(device).unsqueeze(0)
            next_states = torch.from_numpy(next_state).float().to(device).unsqueeze(0)
            dones = torch.from_numpy(np.array([done]).astype(np.uint8)).float().to(device).unsqueeze(0)

            if ENABLE_DDQN:
                Q_targets_arg = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
                Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_targets_arg)
            else:
                Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + GAMMA * Q_targets_next * (1 - dones)
            Q_expected = self.qnetwork_local(states).gather(1, actions)
            error = torch.abs(Q_expected - Q_targets).mean(1).cpu().data.numpy()
            '''

            self.memory.add(1000, state, action, reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                batch = self.memory.sample()
                self.learn(batch, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if type(self.memory) == PrioritizedReplayBuffer:
            batch, idxs, is_weights = experiences
        else:
            batch = experiences
        states, actions, rewards, next_states, dones = extract(batch)

        if ENABLE_DDQN:
            Q_targets_arg = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_targets_arg)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if type(self.memory) == PrioritizedReplayBuffer:
            errors = torch.abs(Q_expected - Q_targets).mean(1).cpu().data.numpy()
            for idx, error in zip(idxs, errors):
                self.memory.update(idx, error)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

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
    #'''
    states = torch.from_numpy(np.vstack([e[0] for e in batch])).float().to(device)
    actions = torch.from_numpy(np.vstack([e[1] for e in batch])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in batch])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in batch])).float().to(device)
    dones = torch.from_numpy(np.vstack([e[4] for e in batch]).astype(np.uint8)).float().to(device)
    '''
    states = torch.stack([e[0] for e in batch if e is not None])
    actions = torch.stack([e[1] for e in batch if e is not None])
    rewards = torch.stack([e[2] for e in batch if e is not None])
    next_states = torch.stack([e[3] for e in batch if e is not None])
    dones = torch.stack([e[4] for e in batch if e is not None])
    #'''
    return states, actions, rewards, next_states, dones