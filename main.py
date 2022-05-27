import gym
import torch.nn as nn
from rl_methods import mlp

# CURRENTLY TESTING
# Create a NN and check that it was created properly


env = gym.make("ALE/Pong-v5")
print(env.observation_space.shape)
print(env.action_space)
print(env.action_space.n)

env = gym.make("HalfCheetah-v2")
print(env.observation_space.shape)
print(env.action_space)
print(env.action_space.shape)