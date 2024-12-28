import itertools

class ActorCritic:
    """
    It contains the policy and value networks, and the methods call them and get actions/value estimates.
    """
    def __init__(self, policy_network, value_network, config):
        self.policy_net = policy_network
        self.value_net = value_network

        self.normalize_v_targets = config.normalize_v_targets

        if self.normalize_v_targets:
            self.value_mean = 0
            self.value_std = 1
            self.values_count = 0

    # act(), value() and action_and_value() are used during play, hence a single value (.item()) is returned
    def act(self, state):
        dist = self.policy_net(state)
        action = dist.sample()

        return action.item()
    
    def value(self, state):
        value = self.value_net(state)

        if self.normalize_v_targets:
            # denormalize value -> TODO: controlla di starlo facendo bene, ma direi di si
            value = value * max(self.value_std, 1e-6) + self.value_mean

        return value

    def action_and_value(self, state, action=None, agents_mask=None):
        dist = self.policy_net(state)
        if action is None:
            action = dist.sample()

        if agents_mask is not None:
            action = action * agents_mask

        value = self.value(state)

        return action, dist.log_prob(action), dist.entropy(), value
    
    # TODO: remove
    # actions_dist() and actions_dist_and_v() are used during training, hence the full distributions and values are returned
    # def actions_dist(self, state):
    #     return self.policy_net(state)
    
    # def actions_dist_and_v(self, state):
    #     dist = self.policy_net(state)
    #     value = self.value_net(state)

    #     return dist, value
      
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
        """If normalize_v_targets is True, will be called to update the mean and std of value targets. This is used to normalize value targets during training."""
        new_values_count = self.values_count + len(v_targets)
        
        self.value_mean = self.value_mean * (self.values_count / (new_values_count + 1e-6)) + v_targets.mean() * (len(v_targets) / (new_values_count + 1e-6))
        self.value_std = self.value_std * (self.values_count / (new_values_count + 1e-6)) + v_targets.std() * (len(v_targets) / (new_values_count + 1e-6))
        self.values_count = new_values_count

    def policy_state_dict(self):
        return self.policy_net.state_dict()
    
    def load_policy_state_dict(self, state_dict):
        self.policy_net.load_state_dict(state_dict)
