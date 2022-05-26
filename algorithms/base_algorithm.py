# RL IMPLEMENTATIONS - BASE ALGORITHMS
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# This class represents the base algorithm used for compatibility by all other
# implemented RL algorithms

# IMPORTS #
from gym import Env


class BaseAlgorithm:
    """
    Base class used by all other implemented RL algorithsm, defining
    the base methods to be shared by all classes.

    Also includes helper methods related to utilities and Gym
    """

    # ABSTRACT METHODS #
    def train(self, total_epochs, steps_per_epoch):
        raise NotImplementedError

    def eval(self, total_steps):
        raise NotImplementedError

    # HELPER METHODS #
    # TODO
    def extract_env_input_output(self, environment):
        """
        Given an Env, extracts both the input and output shape

        Parameters
        ----------
        environment : Env
            A generic Gym environment

        Returns
        -------

        """
        pass

    # TODO
    def extract_space_shape(self, space):
        """
        Given a Space, extracts the proper shape according to the Space

        The shape is given as a list of integer values.
        - If the list has a single element, a 1D shape is assumed.
        - Otherwise, each element in the list represents the size of the corresponding dimension

        Parameters
        ----------
        environment : Env
            A generic Gym environment

        Returns
        -------
        """
        pass