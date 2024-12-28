import itertools

class ActorCritic:
    def __init__(self, policy_network, value_network, config):
        """Actor-Critic model. It is composed of a policy network and a value network.

        Args:
            policy_network (torch.nn.Module): the policy network
            value_network (torch.nn.Module): the value network
            config (dict): the configuration dictionary
        """
        self.policy_net = policy_network
        self.value_net = value_network

        self.normalize_v_targets = config.normalize_v_targets

        if self.normalize_v_targets:
            self.value_mean = 0
            self.value_std = 1
            self.values_count = 0

    def act(self, state):
        """Get an action from the policy network (more precisely, an action for each agent). Used only at inference time, i.e. during actual play.

        Args:
            state (torch.nn.Tensor): the input state, of shape [batch_size, num_agents, state_size]

        Returns:
            action: the sampled action
        """
        dist = self.policy_net(state)
        action = dist.sample()

        return action
    
    def value(self, state):
        """Get the value estimate from the value network.

        Args:
            state (torch.nn.Tensor): the input state, of shape [batch_size, num_agents, state_size]

        Returns:
            value: the value estimate
        """
        value = self.value_net(state)

        if self.normalize_v_targets:
            # denormalize value
            value = value * max(self.value_std, 1e-6) + self.value_mean

        return value

    def action_and_value(self, state, action=None, agents_mask=None):
        """Using the policy network get a set of actions, their log probability and respective entropy. Also get the value estimate from the value network.

        Args:
            state (torch.nn.Tensor): the input state, of shape [batch_size, num_agents, state_size]
            action (int, optional): the action of which to compute the log probability. If None, the action is sampled from the policy network's output distribution. Defaults to None.
            agents_mask (torch.nn.Tensor, optional): the mask that specifies which agents are are required to act. Those that are not required to act will have their action set to 0. Defaults to None.

        Returns:
            action, log_prob, entropy, value: the sampled action, its log probability, the entropy of the policy distribution and the value estimate
        """
        dist = self.policy_net(state)
        if action is None:
            action = dist.sample()

        if agents_mask is not None:
            action = action * agents_mask

        value = self.value(state)

        return action, dist.log_prob(action), dist.entropy(), value
      
    def to(self, device):
        self.policy_net.to(device)
        self.value_net.to(device)

    def eval(self):
        self.policy_net.eval()
        self.value_net.eval()

    def train(self):
        self.policy_net.train()
        self.value_net.train()

    def parameters(self):
        return itertools.chain(self.policy_net.parameters(), self.value_net.parameters())

    def update_v_target_stats(self, v_targets):
        """If normalize_v_targets is True, will be called to update the mean and std of value targets. This is used to normalize value targets during training.
        
        Args:
            v_targets (torch.nn.Tensor): the value targets, of shape [batch_size]
        """
        new_values_count = self.values_count + len(v_targets)
        
        self.value_mean = self.value_mean * (self.values_count / (new_values_count + 1e-6)) + v_targets.mean() * (len(v_targets) / (new_values_count + 1e-6))
        self.value_std = self.value_std * (self.values_count / (new_values_count + 1e-6)) + v_targets.std() * (len(v_targets) / (new_values_count + 1e-6))
        self.values_count = new_values_count

    def policy_state_dict(self):
        return self.policy_net.state_dict()
    
    def value_state_dict(self):
        return self.value_net.state_dict()
    
    def load_policy_state_dict(self, state_dict):
        self.policy_net.load_state_dict(state_dict)

    def load_value_state_dict(self, state_dict):
        self.value_net.load_state_dict(state_dict)
