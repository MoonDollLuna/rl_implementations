# RL IMPLEMENTATIONS - SIMPLE GRADIENT
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# Experience to be re-used by all memory structures in RL algorithms
from typing import Any


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
    action: int
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
    action: int
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