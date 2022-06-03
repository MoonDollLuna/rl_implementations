"""
This module implements the most typical memory structures used by RL algorithms:
    * ReplayBuffer implements a buffer for on-policy methods (such as policy gradient), that gets flushed
      after each epoch.
    * ReplayMemory and its variants implement a memory for off-policy methods (such as value methods),
      that continually store past experiences
"""

# IMPORTS
from .experience import Experience, unpack_experiences
