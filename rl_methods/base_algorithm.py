# RL IMPLEMENTATIONS - BASE ALGORITHMS
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# This class represents the base algorithm used for compatibility by all other
# implemented RL rl_methods

# IMPORTS #
from typing import Tuple, Union

from gym import Env, Space
from gym.spaces import Box


class BaseAlgorithm:
    """
    Base class used by all other implemented RL algorithsm, defining
    the base methods to be shared by all classes.

    Also includes helper methods related to utilities and Gym

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

    # ABSTRACT METHODS #
    def train(self, total_epochs, steps_per_epoch):
        raise NotImplementedError

    def eval(self, total_steps):
        raise NotImplementedError
