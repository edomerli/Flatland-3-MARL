
import torch
from torch import nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class RailTranformer(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, num_layers, activation=nn.ReLU):
        super(RailTranformer, self).__init__()

        self.actor_flag = action_size > 1

        if not self.actor_flag:
            # create a [VALUE] token for the critic network
            self.value_token = nn.Parameter(torch.randn(1, 1, state_size))
            torch.nn.init.normal_(self.value_token, std=0.02)

        ### EMBEDDER ###
        self.first = nn.Sequential(nn.Linear(state_size, hidden_size), activation())
        self.embedder = nn.ModuleList()

        in_channels = hidden_size
        hidden_channels = [hidden_size] * (num_layers-1)

        for out_channels in hidden_channels:
            self.embedder.append(nn.Sequential(
                nn.Linear(in_channels, out_channels), 
                activation()
            ))

            in_channels = out_channels

        ### ATTENTION ###
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 4, hidden_size * 4, norm_first=False, batch_first=True),
            num_layers=3,
            norm=nn.LayerNorm(hidden_size),
        )

        ### ACTOR or CRITIC HEAD ###
        self.head = nn.ModuleList()

        in_channels = hidden_size
        hidden_channels = [hidden_size] * num_layers

        for out_channels in hidden_channels:
            self.head.append(nn.Sequential(
                layer_init(nn.Linear(in_channels, out_channels)), 
                activation() 
            ))

            in_channels = out_channels

        if self.actor_flag:
            self.final = layer_init(nn.Linear(in_channels, action_size), std=0.01)
        else:
            self.final = layer_init(nn.Linear(in_channels, action_size), std=1.0)


    def forward(self, x):
        # append the [VALUE] token to the beginning of the sequence if it's a Critic network
        if not self.actor_flag:
            x = torch.cat([self.value_token.expand(x.shape[0], -1, -1), x], dim=1)

        # Embedding
        x = self.first(x)

        for layer in self.embedder:
            x = layer(x) + x

        # Attention
        x = self.transformer(x)

        # Head
        # TODO: è corretto lasciare la residual connection anche all'ultimo layer (btw, quelli del paper "skip connections eliminate singularities" la mettono anche all'ultimo!)? 
        # -> TESTA CON E SENZA! Sono troppo curioso ahahahahha
        if self.actor_flag:
            # if it's the actor -> apply to all tokens
            for layer in self.head:
                x = layer(x) + x
        else:
            # if it's the critic -> apply only to the first token
            x = x[:, 0, :]
            for layer in self.head:
                x = layer(x)

        x = self.final(x)

        if self.actor_flag:
            output = torch.distributions.Categorical(logits=x)
        else:
            output = x.squeeze()

        return output   