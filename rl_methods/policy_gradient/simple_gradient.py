# RL IMPLEMENTATIONS - SIMPLE GRADIENT
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# Simplest version of a Policy Gradient Algorithm, using the most basic
# gradient calculation without advantages

# IMPORTS #
from torch.nn import Module, ReLU, Identity

from rl_methods import BaseAlgorithm, mlp

# CLASS DEFINITION #
class SimpleGradient(BaseAlgorithm):
    """
    Simplest version of a Policy Gradient Algorithm, using the most basic
    gradient calculation without advantages

    Parameters
    ----------
    env : Env
        A generic Gym environment
    """

    # NETWORKS
    # Policy neural network
    policy_net: Module

    # ATTRIBUTES
    # Flag indicating if the

    # CONSTRUCTOR
    def __init__(self, env):

        # Prepare the environment
        super().__init__(env)

        # Instantiate a network based on the input type
        if len(self.obs_shape) > 1:
            # Shape is bigger than 1 - CNN for images
            pass
        else:
            # Shape is 1 - MLP for simple inputs
            self.policy_net = mlp(self.obs_shape[0], [32], ReLU, self.act_shape, Identity)


    # MAIN METHODS
    def train(self, total_epochs, steps_per_epoch):
        pass

    def eval(self, total_steps):
        pass

    # HELPER METHODS
