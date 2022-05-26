# RL IMPLEMENTATIONS - SIMPLE GRADIENT
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# Simplest version of a Policy Gradient Algorithm, using the most basic gradient calculation
# without advantages

# IMPORTS #
import torch
from algorithms.base_algorithm import BaseAlgorithm

# CLASS DEFINITION #
class SimpleGradient(BaseAlgorithm):

    def __init__(self):
        pass

    def train(self, total_epochs, steps_per_epoch):
        pass

    def eval(self, total_steps):