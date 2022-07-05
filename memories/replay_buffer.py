# RL IMPLEMENTATIONS - REPLAY BUFFER (AND HELPERS)
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# This file contains:
#   - A Replay Buffer to be used as a memory for all policy gradient methods
#   - An Episode class, containing all necessary information within an episode


# IMPORTS #
from typing import List, Any, Tuple, Union, Optional

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
    rewards_to_go: List[float]
        Rewards to go for each experience within the episode
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
    # Rewards to go for each experience within the episode
    rewards_to_go: List[float]

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
            self.rewards_to_go.append(reward_to_go)


# REPLAY BUFFER
class ReplayBuffer:
    """
    A ReplayBuffer represents the memory buffer used by actor-critic reinforcement-learning methods, used
    to store and handle the experiences from the agent episodes.

    Unlike Replay Memory (used by value-based methods such as DQN), Replay Buffers work on-policy, by storing
    experiences performed with the current policy and being flushed after use.

    current_episode: Episode, optional
        Current episode. If None, there is no currently started episode
    episode_list: List[Episode]
        List of all episodes, not including the current episode
    states: list[Any]
        List of current states s of ALL episodes
    actions: list[int or tuple]
        List of actions a of ALL episodes
    rewards: list[float]
        List of rewards r of ALL episodes
    next_states: list[Any]
        List of next states s' of ALL episodes
    final_flags: list[bool]
        List of final flags f of ALL episodes
    episode_reward: list[float]
        Total reward for each episode, for ALL episodes
        Note: experiences of the same episode have the same reward
    rewards_to_go: list[float]
        Rewards to go for each experience within the episode, for ALL episodes
    """

    # ATTRIBUTES #

    # Current episode. If None, there is no currently started episode
    current_episode: Optional[Episode]
    # List of all episodes, not including the current episode
    episode_list: List[Episode]

    # List of current states s of ALL episodes
    states: List[Any]
    # List of actions a of ALL episodes
    actions: List[Union[int, Tuple]]
    # List of rewards r of ALL episodes
    rewards: List[float]
    # List of next states s' of ALL episodes
    next_states: List[Any]
    # List of final flags f of ALL episodes
    final_flags: List[bool]
    # Total reward for each episode, for ALL episodes
    # Note: experiences of the same episode have the same reward
    episode_reward: List[float]
    # Rewards to go for each experience within the episode, for ALL episodes
    rewards_to_go: List[float]

    # CONSTRUCTOR #
    def __init__(self):

        # The Replay Buffer starts without a current episode
        self.current_episode = None

        # All lists start empty
        self.episode_list = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.final_flags = []
        self.episode_reward = []
        self.rewards_to_go = []

    # METHODS #

    # Episode management
    def start_episode(self):
        """
        Creates a new episode and sets it as the current episode
        """

        # Ignore this method if an episode already exists
        if not self.current_episode:
            self.current_episode = Episode()

    def finish_episode(self, completed):
        """
        Finishes the episode, removing it as the current episode and storing the episode values
        within the buffer for faster usage

        If the episode has been "cut out" (the episode is not actually finished but the training process stops
        earlier due to cut off), the final experience is not marked as final

        Parameters
        ----------
        completed: bool
            True if the episode has finished naturally, False otherwise
        """

        # Ignore this method if there is no current episode
        if self.current_episode:

            # Mark the episode as finished
            self.current_episode.finish_episode(completed)

            # Store the episode info within the buffer lists
            self.states.extend(self.current_episode.states)
            self.actions.extend(self.current_episode.actions)
            self.rewards.extend(self.current_episode.rewards)
            self.next_states.extend(self.current_episode.next_states)
            self.final_flags.extend(self.current_episode.final_flags)
            self.episode_reward.extend(self.current_episode.episode_reward)
            self.rewards_to_go.extend(self.current_episode.rewards_to_go)

            # Store the current episode within the buffer and remove it from the current episode variable
            self.episode_list.append(self.current_episode)
            self.current_episode = None

    # Experience management
    def insert_experience(self, state, action, reward, next_state):
        """
        Inserts an experience into the current episode. Experiences are, by default, unfinished.

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

        # Ignore this method if there is no current episode
        if self.current_episode:
            self.current_episode.insert_experience(state, action, reward, next_state)

    # Replay buffer management
    def empty(self):
        """
        Flushes the replay buffer to empty the information, preparing it for the next training epoch
        """
        
        # The Replay Buffer starts without a current episode
        self.current_episode = None

        # All lists start empty
        self.episode_list = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.final_flags = []
        self.episode_reward = []
        self.rewards_to_go = []

    def get_epoch_info(self):
        """
        Returns all the info for the current epoch, to be used during training

        Returns
        -------
        (list[Any], list[int], list[float], list[Any], list[bool], list[float], list[float])
        """

        return self.states, self.actions, self.rewards, self.next_states, \
               self.final_flags, self.episode_reward, self.rewards_to_go
