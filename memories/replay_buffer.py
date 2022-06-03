# RL IMPLEMENTATIONS - REPLAY BUFFER (AND HELPERS)
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# This file contains:
#   - A Replay Buffer to be used as a memory for all policy gradient methods
#   - An Episode class, containing all necessary information within an episode

# TODO MEASURE TIME


# IMPORTS #
from typing import List, Any, Tuple, Union

from memories import Experience, unpack_experiences


# EPISODE #
class Episode:
    """
    An Episode represents a complete episode performed by an agent in a RL environment

    To be more precise, an Episode contains all Experiences performed from the Episode start to the
    Episode end, and several helper methods to extract and manage info related to said experiences.

    Attributes
    ----------
    experiences: list[Experience]
        Current list of experiences within the episode
    finished: bool
        Whether this episode is "complete" or not
    states: list[Any]
        List of current states s (after episode completion)
    actions: list[int or tuple]
        List of actions a (after episode completion)
    rewards: list[float]
        List of rewards r (after episode completion)
    next_states: list[Any]
        List of next states s' (after episode completion)
    final_flags: list[bool]
        List of final flags f (after episode completion)
    episode_reward: list[float]
        Total reward obtained by this episode / trajectory
    """

    # ATTRIBUTES #

    # Current list of experiences within the episode
    experiences: List[Experience]
    # Whether this episode is "complete" or not
    finished: bool

    # List of current states s (after episode completion)
    states: List[Any]
    # List of actions a (after episode completion)
    actions: List[Union[int, Tuple]]
    # List of rewards r (after episode completion)
    rewards: List[float]
    # List of next states s' (after episode completion)
    next_states: List[Any]
    # List of final flags f (after episode completion)
    final_flags: List[bool]
    # Total reward obtained by this episode / trajectory, repeated for all experiences
    episode_reward: List[float]

    # CONSTRUCTOR #
    def __init__(self):

        # Episodes start empty and unfinished
        self.experiences = []
        self.finished = False

        # The unrolled lists of experiences, general reward and reward-to-go are computed after the episode ends
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.final_flags = []
        self.episode_reward = []
        self.rewards_to_go = []

    # METHODS #

    def insert_experience(self, state, action, reward, next_state):
        """
        Inserts an experience into the episode. Experiences are, by default, unfinished.

        Parameters
        ----------
        state: Any
            Initial state s
        action: int
            Action a performed in state s
        reward: float
            Reward r obtained after performing action a in state s
        next_state: Any
            Next state s' reached after applying action a to state s
        """

        # Create and append a new experience
        experience = Experience(state, action, reward, next_state, False)
        self.experiences.append(experience)

    def finish_episode(self, completed):
        """
        Marks the episode as finished and computes:
            * The unrolled lists for each episode element
            * The episode reward
            * The rewards-to-go of each episode

        Parameters
        ----------
        completed: bool
            Whether the final experience was final (True) or the episode was cut short (False)
        """

        # Mark the episode as finished
        self.finished = True

        # If specified, mark the last experience as a final experience
        if completed:
            self.experiences[-1].final = True

        # Unfold the experiences
        self.states, self.actions, self.rewards, self.next_states, self.final_flags = unpack_experiences(self.experiences)

        # Compute the complete episode reward
        self.episode_reward = [sum(self.rewards)] * len(self.rewards)

        # Compute the reward-to-go for each experience
        for i in range(len(self.rewards)):
            reward_to_go = sum(self.rewards[i:])
            self.rewards_to_go.append(reward_to_go
