# RL IMPLEMENTATIONS - LOGGERS
#
# Developed by Luna Jimenez Fernandez
# Based on OpenAI Spin Up
#
# This file implements loggers to be used to:
#   * Display information about the training process
#   * Store said information into a .log file

# IMPORTS #
import time


class BaseLogger:
    """
    Base class for Loggers, used to store and display information about the training process

    This class especifically handles the timing aspects
    """

    # ATTRIBUTES

    # Initial training time
    init_time: float
    # Time of the last timestamp
    time_stamp: float

    # CONSTRUCTOR
    def __init__(self):

        # Extract the initial time and store it
        self.init_time = time.time()
        self.time_stamp = self.init_time

    # HELPER METHODS
    def timestamp(self):
        """
        Returns, in order, the time since the last timestamp call and the total training time

        Returns
        -------
        (float, float)
        """

        # Get the current time
        current_time = time.time()

        # Get both the total and the timestamp times
        total_time = current_time - self.init_time
        timestamp_time = current_time - self.time_stamp

        # Update the timestamp
        self.time_stamp = current_time

        return timestamp_time, total_time


class PolicyGradientLogger(BaseLogger):
    """
    Logger designed for Policy Gradient methods.

    This logger, in addition to all BaseLogger functionality, includes the necessary methods
    to showcase the progress during training
    """

