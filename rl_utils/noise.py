import numpy as np
import copy
import random


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, shape, seed, mu=0., theta=0.15, sigma=0.08):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.shp = shape
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.shp)
        self.state = x + dx
        return self.state


class NormalNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, shape, seed, sigma=0.5):
        """Initialize parameters and noise process."""
        self.shape = shape
        self.sigma = sigma
        self.seed = random.seed(seed)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        return np.random.normal(0., self.sigma, self.shape)
