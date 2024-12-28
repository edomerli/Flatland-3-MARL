import torch
from torch import nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, num_layers, activation=nn.Tanh):
        """Multi-layer perceptron network.

        Args:
            state_size (int): the size of the input state, i.e. the observation space size for an agent
            action_size (int): the number of possible actions an agent can take
            hidden_size (int): the size of the hidden layers
            num_layers (int): the number of *hidden* layers
            activation (function, optional): the activation function for all layers except the final one. Defaults to nn.Tanh.
        """
        super(MLP, self).__init__()

        self.actor_flag = action_size > 1

        self.first = nn.Sequential(nn.Linear(state_size, hidden_size), activation())
        self.hidden_layers = nn.ModuleList()

        in_channels = hidden_size
        assert num_layers >= 2, "Number of layers must be at least 2 (first layer to map input_size -> hidden_size, last layer to map hidden_size -> action_size)"
        hidden_channels = [hidden_size] * (num_layers-2)

        for out_channels in hidden_channels:
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(in_channels, out_channels), 
                activation()
            ))

            in_channels = out_channels

        if self.actor_flag:
            self.final = layer_init(nn.Linear(in_channels, action_size), std=0.01)
        else:
            self.final = layer_init(nn.Linear(in_channels, action_size), std=1.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.nn.Tensor): the input tensor, of shape [batch_size, num_agents, state_size]

        Returns:
            output: the output tensor, of shape [batch_size, num_agents, action_size] if it's an Actor network, [batch_size] if it's a Critic network
        """
        x = self.first(x)

        for layer in self.hidden_layers:
            x = layer(x) + x

        x = self.final(x)

        if self.actor_flag:
            output = torch.distributions.Categorical(logits=x)
        else:
            output = x.squeeze()
            # average over the agents dimension
            output = output.mean(dim=-1)

        return output