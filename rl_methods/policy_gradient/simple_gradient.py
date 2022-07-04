# RL IMPLEMENTATIONS - SIMPLE GRADIENT
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# Simplest version of a Policy Gradient Algorithm, using the most basic
# gradient calculation without advantages

# IMPORTS #
import torch
from torch import Tensor
from torch.nn import Module, ReLU, Identity
from torch.distributions import Categorical
from torch.optim import Adam

from rl_methods import PolicyGradientAlgorithm, mlp
from memories import ReplayBuffer


# TODO MEASURE TIME


# CLASS DEFINITION #
class SimpleGradient(PolicyGradientAlgorithm):
    """
    Simplest version of a Policy Gradient Algorithm, using the most basic
    gradient calculation without advantages

    Parameters
    ----------
    env : Env
        A generic Gym environment
    """

    # NETWORKS AND MEMORIES
    # Policy neural network
    policy_net: Module

    # Replay Buffer is created in the parent class

    # CONSTRUCTOR
    def __init__(self, env):

        # Prepare the environment, replay buffer and device for Torch
        super().__init__(env)

        # Instantiate the policy network based on the input type
        # Simple gradient does not have a separate critic network.
        if len(self.obs_shape) > 1:
            # Shape is bigger than 1 - CNN for images
            pass
        else:
            # Shape is 1 - MLP for simple inputs
            self.policy_net = mlp(self.obs_shape[0], [32], ReLU, self.act_shape, Identity)

        # Send the neural network to the proper device
        self.policy_net.to(self.device)

    # MAIN METHODS
    def train(self, total_epochs, steps_per_epoch):
        """
        Trains the agent for total_epochs. The agent runs for steps_per_epoch steps, and then performs training
        based on the on-policy experiences, updating the network weights

        Parameters
        ----------
        total_epochs: int
            Total number of epochs to train
        steps_per_epoch: int
            How many steps are performed in each epoch. This number is approximate - epochs will be this size
            or slightly larger but never smaller, since the last episode within the epoch will not be finished
        """

        # Prepare the logger for training
        # TODO - CREATE LOGGER

        # Prepare the optimizer
        # ADAM is used for simplicity
        optimizer = Adam()

        # Perform each epoch separately
        for epoch in range(total_epochs):
            # Handle the epoch by letting the agent run for the specified number of steps
            self._epoch(steps_per_epoch)

            # Extract the info from the replay buffer
            states, actions, rewards, \
            next_states, final_flags, \
            episode_reward, rewards_to_go = self.replay_buffer.get_epoch_info()

            # Reset the optimizer gradients
            optimizer.zero_grad()

            # Obtain the loss (gradients)
            loss = self._compute_losses(states, actions, episode_reward)

            # Perform gradient descent
            loss.backward()
            optimizer.step()

            # Flush the replay buffer after the update
            self.replay_buffer.empty()

    def eval(self, total_steps):
        pass

    def act(self, observation):
        """
        Given an observation, sample and return an action to perform, identified by an int ID

        Parameters
        ----------
        observation: Tensor

        Returns
        -------
        Categorical
        """

        # TODO Currently this method assumes a categorical policy

        # Convert the observation into a tensor
        observation = self._to_tensor(observation)

        # Create the policy from the policy network
        policy = self._get_categorical_policy(observation)

        # Return a sampled action from said policy
        return policy.sample().item()

    # HELPER METHODS #
    def _get_categorical_policy(self, observation):
        """
        Given an observation (in Tensor form), return a categorical policy to sample

        Parameters
        ----------
        observation: Tensor

        Returns
        -------
        Categorical
        """

        # Obtain the logits from the policy network
        logits = self.policy_net(observation)

        # Return the proper categorical distribution
        return Categorical(logits=logits)

    def _epoch(self, total_steps):
        """
        Runs the agent for "total_steps" as a single epoch

        Parameters
        ----------
        total_steps: int
        """

        # Counter of the current amount of steps
        current_steps = 0
        # Whether the current episode is finished or not
        finished = False

        # Prepare the environment and the buffer for training, obtaining the first observations / state
        current_obs = self.env.reset()
        self.replay_buffer.start_episode()

        # The epoch will run until an episode is finished AFTER total_steps has been reached
        while not finished and current_steps < total_steps:

            # Mark episode as not finished
            finished = False

            # Find the proper action for the agent
            act = self.act(current_obs)

            # Act in the environment
            next_obs, reward, done, _ = self.env.step(act)

            # Store the current experience and mark the next observation as the current state
            self.replay_buffer.insert_experience(current_obs, act, reward, next_obs)

            # Check if the episode is over
            if done:
                # Mark the episode as finished
                finished = True

                # Reset the environment
                current_obs = self.env.reset()

                # Finish the episode in the buffer and start the next episode
                self.replay_buffer.finish_episode(done)
                self.replay_buffer.start_episode()

    def _compute_losses(self, observations, actions, rewards):
        """
        Computes the loss (gradient descent) for each state-action pair

        The method internally transforms all necessary lists into tensors

        Parameters
        ----------
        observations: list[Any]
            List of all observations (states) in an epoch
        actions: list[int]
            List of all actions for each corresponding state
        rewards: list[float]
            List of EPISODE rewards for each state

        Returns
        -------
        Tensor
        """

        # Convert all lists into tensors
        observations = self._to_tensor(observations)
        actions = self._to_tensor(actions)
        rewards = self._to_tensor(rewards)

        # Compute the log-probability of all state-action pairs
        log_probs = self._get_categorical_policy(observations).log_prob(actions)

        # Obtain the gradient and return it
        # Negative value is used to perform gradient ascent
        # Mean is taken since the expected mean value is used
        gradients = -(log_probs * rewards).mean()

        return gradients
