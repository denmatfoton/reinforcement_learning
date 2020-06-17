import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class FcNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, hidden_layers, act_func):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers: array of dimensions of hidden layers
            act_func: activation function in layers
        """
        super(FcNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_layers])
        
        self.output = nn.Linear(hidden_layers[-1], action_size)
        self.act_func = act_func
        self.reset_parameters()

    def reset_parameters(self):
        for hl in self.hidden_layers:
            hl.weight.data.uniform_(*hidden_init(hl))
        self.output.weight.data.uniform_(-3e-3, 3e-3)


class Actor(FcNetwork):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, act_func=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers: array of dimensions of hidden layers
            act_func: activation function in layers
        """
        super(Actor, self).__init__(state_size, action_size, seed, hidden_layers, act_func)

    def forward(self, x):
        for linear, bn in zip(self.hidden_layers, self.batch_norm):
            x = self.act_func(bn(linear(x)))
        x = torch.tanh(self.output(x))
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, act_func=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers: array of dimensions of hidden layers
            act_func: activation function in layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(d) for d in hidden_layers])
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        self.hidden_layers.append(nn.Linear(hidden_layers[0] + action_size, hidden_layers[1]))

        self.output = nn.Linear(hidden_layers[-1], action_size)

        self.act_func = act_func
        self.reset_parameters()

    def reset_parameters(self):
        for hl in self.hidden_layers:
            hl.weight.data.uniform_(*hidden_init(hl))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.act_func(self.batch_norm[0](self.hidden_layers[0](state)))
        x = torch.cat((xs, action), dim=1)
        x = self.act_func(self.batch_norm[1](self.hidden_layers[1](x)))
        return self.output(x)
