import numpy as np
import random


class Policy(object):
    """Abstract base class for all implemented policies.
    Each policy helps with selection of action to take on an environment.
    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:
    - `select_action`
    # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError()

    def get_config(self):
        """Return configuration of the policy
        # Returns
            Configuration as dict
        """
        return {}


class NewEpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy
    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, eps=.1):
        super(NewEpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values, valid_actions):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1

        if np.random.uniform() < self.eps:
            action = random.choice(valid_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyQPolicy
        # Returns
            Dict of config
        """
        config = super(NewEpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config
