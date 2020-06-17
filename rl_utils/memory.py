import random
import numpy as np
import torch
from collections import namedtuple, deque
from .sum_tree import SumTree


class ReplayBuffer:
    def __init__(self, capacity, batch_size, seed):
        self.capacity = capacity
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class SimpleReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity, batch_size, seed):
        """Initialize a SimpleReplayBuffer object.

        Params
        ======
            capacity (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        super().__init__(capacity, batch_size, seed)
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(self.memory, k=self.batch_size)
        return batch

    def __len__(self):
        """Return the number of elements in internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    eps = 0.01

    def __init__(self, capacity, batch_size, seed, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        super().__init__(capacity, batch_size, seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def add(self, error, state, action, reward, next_state, done):
        # e = self.experience(state, action, reward, next_state, done)
        p = self._get_priority(error)
        self.tree.add(p, (state, action, reward, next_state, done))

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        """Return the number of elements in internal memory."""
        return self.tree.size
