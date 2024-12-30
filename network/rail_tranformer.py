
import torch
from torch import nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class RailTranformer(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, num_layers, activation=nn.Tanh):
        """Rail Transformer network.

        It's composed of an Embedder, a Transformer and a Head.
        The Embedder is a series of residual fully connected layers with the same hidden size, which compute the embedding for each token (i.e. each agent's observation).
        The Transformer is a set of Transformer encoder layers (modeling the communication between agents).
        The Head is a series of residual fully connected layers with the same hidden size, except for the final layer which maps to the action_size.

        If it's used as an Actor, the final layer's activations are passed through a Categorical distribution as logits.
        If it's used as a Critic, we append a [VALUE] token to the beginning of the sequence and the final layer is applied only to this token's final representation to compute the value.

        Args:
            state_size (int): the size of the input state, i.e. the observation space size for an agent
            action_size (int): the number of possible actions an agent can take
            hidden_size (int): the size of the hidden layers
            num_layers (int): the number of Embedding layers, Transformer layers and Head layers (all the same)
            activation (function, optional): the activation function for all the Embedding and Head layers, except the final layer. Defaults to nn.Tanh.
        """
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
        hidden_channels = [hidden_size] * (num_layers-1)    # -1 because the first layer is already defined

        for out_channels in hidden_channels:
            self.embedder.append(nn.Sequential(
                nn.Linear(in_channels, out_channels), 
                activation()
            ))

            in_channels = out_channels

        ### ATTENTION ###
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 4, hidden_size * 4, norm_first=False, batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
        )

        ### ACTOR or CRITIC HEAD ###
        self.head = nn.ModuleList()

        in_channels = hidden_size
        hidden_channels = [hidden_size] * (num_layers - 1)   # -1 because we apply a final layer at the end

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
        """Forward pass.

        Args:
            x (torch.nn.Tensor): the input tensor, of shape [batch_size, num_agents, state_size]

        Returns:
            output: the output tensor, of shape [batch_size, num_agents, action_size] if it's an Actor network, [batch_size] if it's a Critic network
        """
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
        if self.actor_flag:
            # if it's the actor -> apply to all tokens
            for layer in self.head:
                x = layer(x) + x
        else:
            # if it's the critic -> apply only to the first token
            x = x[:, 0, :]
            for layer in self.head:
                x = layer(x) + x

        x = self.final(x)   

        if self.actor_flag:
            output = torch.distributions.Categorical(logits=x)
        else:
            output = x.squeeze()

        return output   