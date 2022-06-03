# RL IMPLEMENTATIONS - SIMPLE GRADIENT
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# Experience to be re-used by all memory structures in RL algorithms

# IMPORTS #
from typing import Any, Union, Tuple


class Experience:
    """
    An experience represents previous behaviours by the agent, to be used during RL training.

    More specifically, an experience contains:
        * The initial state `s`
        * The action `a` performed on state `s`
        * The reward `r` obtained after performing action `a` in state `s`
        * The reached state `s'`
        * Whether the state is final (TRUE) or not (FALSE), `f`

    Parameters
    ----------
    state: Any
        Initial state
    action: int, tuple
        Action performed
    reward: float
        Reward obtained
    next_state: Any
        State reached
    final: bool
        Final flag
    """

    # ATTRIBUTES #
    # Initial state s
    state: Any
    # Chosen action a
    action: Union[int, Tuple[Any]]
    # Reward obtained r
    reward: float
    # Reached state s'
    next_state: Any
    # Final flag f
    final: bool

    # CONSTRUCTOR #
    def __init__(self, state, action, reward, next_state, final):

        # Store the information
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.final = final


# STATIC METHODS

def unpack_experiences(experience_list):
    """
    Given a list of Experiences, return five lists with:
        * The initial state `s`
        * The action `a` performed on state `s`
        * The reward `r` obtained after performing action `a` in state `s`
        * The reached state `s'`
        * Whether the state is final (TRUE) or not (FALSE), `f`

    Parameters
    ----------
    experience_list: list[Experience]
        List of experiences to be unpacked

    Return
    ------
    (list[Any], list[int or tuple], list[float], list[Any], list[bool])
    """

    # Create lists to store the unpacked experiences
    states = []
    actions = []
    rewards = []
    next_states = []
    final_flags = []

    # Unpack each experience
    for experience in experience_list:
        states.append(experience.state)
        actions.append(experience.action)
        rewards.append(experience.reward)
        next_states.append(experience.next_state)
        final_flags.append(experience.final)

    return states, actions, rewards, next_states, final_flags
