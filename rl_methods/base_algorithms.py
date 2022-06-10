# RL IMPLEMENTATIONS - BASE ALGORITHMS
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# This file implements:
#   * The base implementation of a RL Method
#   * A base implementation for Policy Gradient methods
#   * A base implementation for Value Based methods

# IMPORTS #
from typing import Tuple, Any

from gym import Env, Space
from gym.spaces import Box

import torch

from memories import ReplayBuffer


class BaseAlgorithm:
    """
    Base class used by all other implemented RL algorithsm, defining
    the base methods to be shared by all classes.

    Further subclasses (PolicyGradientAlgorithm and ValueBasedAlgorithm) introduce
    additional restrictions and shared code for each specific variant

    Parameters
    ----------
    env : Env
        A generic Gym environment
    """

    # ATTRIBUTES
    # Gym environment
    env: Env
    # Shape of the observation space
    # If the length of the shape is bigger than 1, an image is assumed
    obs_shape: Tuple[int, ...]
    # Type of action space
    # If the space is a Box, a continuous action space is assumed
    # Otherwise, a discrete action space is assumed
    act_space: Space
    # Shape of the action space
    # This shape is always assumed to be one-dimensional, regardless of the action space type
    act_shape: int

    # Device where PyTorch will be executed
    # Can either be "cuda" or "cpu"
    device: Any

    # CONSTRUCTOR
    def __init__(self, env):

        # Store the environment and extract the observation and action space
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.act_space = env.action_space

        # Get the length of the action space, depending on the action type
        if isinstance(self.act_space, Box):
            self.act_shape = self.act_space.shape
        else:
            self.act_shape = self.act_space.n

        # Check the proper device for the neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: " + self.device)

    # MAIN METHODS #
    def train(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    # HELPER METHODS #
    def _to_tensor(self, tensor):
        """
        Given an element or a list, converts it to a PyTorch Tensor and sends it to the proper device

        Parameters
        ----------
        tensor: Any

        Returns
        -------
        torch.Tensor
        """

        # TODO Convertir imagenes de forma adecuada
        return torch.as_tensor(tensor, device=self.device)


class PolicyGradientAlgorithm(BaseAlgorithm):
    """
    Base class used by Policy Gradient algorithms, defining more information
    about the specific train and eval parameters required

    Parameters
    ----------
    env : Env
        A generic Gym environment

    Attributes
    ----------
    replay_buffer : ReplayBuffer
        The Replay Buffer used by the policy gradient algorithm
    """

    # ATTRIBUTES
    # The Replay Buffer used by the policy gradient algorithm
    replay_buffer: ReplayBuffer

    # CONSTRUCTOR #
    def __init__(self, env):

        # Super constructor call
        super().__init__(env)

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer()

    # MAIN METHODS #
    def train(self, total_epochs, steps_per_epoch):
        raise NotImplementedError

    def eval(self, total_steps):
        raise NotImplementedError

    def act(self, observation):
        raise NotImplementedError
