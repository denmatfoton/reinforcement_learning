import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, hidden_layers, drop_p=0.5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        """ For additional research
        gru_input_size = state_size + action_size
        gru_hidden_size = action_size
        gru_layers = 1
        self.gru = nn.GRU(gru_input_size, gru_hidden_size, gru_layers, batch_first=False)
        """
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        x = self.output(x)
        return x
