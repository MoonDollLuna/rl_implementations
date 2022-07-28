# RL IMPLEMENTATIONS - POLICY GRADIENT METHOD UTILS
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# This file contains helper method and utilities for Policy Gradient methods, including:
#   * Policy creation (with Categorical and Gaussian policies)

# IMPORTS #

import torch
from torch.distributions import Categorical, Normal


# POLICY CREATION METHODS
def get_categorical_policy(logits):
    """
    Given some logits (as a Tensor), return a categorical policy to sample

    Parameters
    ----------
    logits: Tensor

    Returns
    -------
    Categorical
    """

    # Return the proper categorical distribution
    return Categorical(logits=logits)
