# BASE NN CREATION
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spinning Up
#
# This class includes the base methods to create the main types of NNs used by the algorithms
# (mostly MLPs and CNNs)

# IMPORTS
import torch
import torch.nn as nn


def mlp(input_size, hidden_sizes, hidden_activations, output_size, output_activation):
    """
    Creates a MLP with specified shape and activation functions

    Parameters
    ----------
    input_size : int
        Number of inputs
    hidden_sizes : list[int]
        List of neurons in each hidden layer
    hidden_activations : any
        Activation function used for each hidden layer
    output_size : int
        Size of the output layer
    output_activation : any
        Activation function used by the output layer

    Returns
    -------
    nn.Module
    """

    # Store the layers in a list
    layers = []

    # Create the first linear layer (Input -> Hidden)
    layers += [nn.Linear(input_size, hidden_sizes[0]), hidden_activations()]

    # If necessary, create additional hidden layers
    for i in range(1, len(hidden_sizes)):
        layers += [nn.Linear(hidden_sizes[i-1], hidden_sizes[i]), hidden_activations()]

    # Create the output layer
    layers += [nn.Linear(hidden_sizes[-1], output_size), output_activation()]

    # Return the created module
    return nn.Sequential(*layers)
