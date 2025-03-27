"""
Base controller interface defining the common methods for all controllers.
"""

from abc import ABC, abstractmethod

import numpy as np

from swarm_squad.models.swarm_state import SwarmState


class BaseController(ABC):
    """
    Abstract base class for all controllers.

    Controllers must implement the compute_control method which calculates
    control inputs for each agent in the swarm.
    """

    def __init__(self, swarm_state: SwarmState):
        """
        Initialize the controller with a reference to the swarm state.

        Args:
            swarm_state: Reference to the swarm state object
        """
        self.swarm_state = swarm_state

    @abstractmethod
    def compute_control(self) -> np.ndarray:
        """
        Calculate control inputs for all agents in the swarm.

        Returns:
            A numpy array of shape (swarm_size, 2) containing the control
            inputs for each agent in the swarm.
        """
        pass

    def apply_control(self, control_inputs: np.ndarray = None):
        """
        Apply computed control inputs to update agent positions.

        Args:
            control_inputs: Optional control inputs to apply. If None,
                compute_control() will be called to get control inputs.
        """
        if control_inputs is None:
            control_inputs = self.compute_control()

        # Apply the control inputs to the swarm state
        self.swarm_state.swarm_control_ui = control_inputs
        self.swarm_state.swarm_position += control_inputs
