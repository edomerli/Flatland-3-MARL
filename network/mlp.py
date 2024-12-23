import torch
from torch import nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, num_layers, activation=nn.Tanh):
        super(MLP, self).__init__()

        self.actor_flag = action_size > 1

        self.first = nn.Sequential(nn.Linear(state_size, hidden_size), activation())
        self.hidden_layers = nn.ModuleList()

        in_channels = hidden_size
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
        x = self.first(x)

        for layer in self.hidden_layers:
            x = layer(x) + x    # TODO: test without residual connection

        x = self.final(x)

        if self.actor_flag:
            output = torch.distributions.Categorical(logits=x)
        else:
            output = x.squeeze()
            # average over the agents dimension
            output = output.mean(dim=-1)

        return output