"""
Controller factory to manage different controller types and provide integration
hooks for LLM intervention.
"""

from enum import Enum
from typing import Dict

import numpy as np

from swarm_squad.controllers.base_controller import BaseController
from swarm_squad.controllers.behavior_controller import BehaviorController
from swarm_squad.controllers.communication_controller import CommunicationController
from swarm_squad.models.swarm_state import SwarmState


class ControllerType(Enum):
    """Enum for different controller types"""

    COMMUNICATION = "communication"
    BEHAVIOR = "behavior"
    COMBINED = "combined"
    LLM = "llm"  # Future LLM controller


class ControllerFactory:
    """
    Factory for creating and managing different types of controllers.

    This class serves as a bridge between the simulation and controllers,
    allowing for dynamic switching between controller types and providing
    hooks for LLM intervention.
    """

    def __init__(self, swarm_state: SwarmState):
        """
        Initialize the controller factory.

        Args:
            swarm_state: Reference to the swarm state object
        """
        self.swarm_state = swarm_state
        self.controllers: Dict[ControllerType, BaseController] = {}
        self.active_controller_type = ControllerType.COMMUNICATION

        # Initialize controllers
        self._init_controllers()

    def _init_controllers(self):
        """Initialize all available controllers"""
        self.controllers[ControllerType.COMMUNICATION] = CommunicationController(
            self.swarm_state
        )
        self.controllers[ControllerType.BEHAVIOR] = BehaviorController(self.swarm_state)
        # LLM controller would be initialized here in the future

    def get_controller(self, controller_type: ControllerType) -> BaseController:
        """
        Get a specific controller by type.

        Args:
            controller_type: Type of controller to retrieve

        Returns:
            The requested controller instance
        """
        return self.controllers[controller_type]

    def set_active_controller(self, controller_type: ControllerType):
        """
        Set the active controller.

        Args:
            controller_type: Type of controller to activate
        """
        print(f"DEBUG: Setting active controller to {controller_type}")

        if controller_type == ControllerType.COMBINED:
            self.active_controller_type = ControllerType.COMBINED
            print(f"DEBUG: Active controller is now {self.active_controller_type}")
        elif controller_type in self.controllers:
            self.active_controller_type = controller_type
            print(f"DEBUG: Active controller is now {self.active_controller_type}")
        else:
            print(
                f"WARNING: Controller type {controller_type} not found in available controllers"
            )
            print(f"Available controllers: {list(self.controllers.keys())}")

    def compute_control(self) -> np.ndarray:
        """
        Compute control inputs using the active controller.

        Returns:
            Control inputs for all agents
        """
        if self.active_controller_type == ControllerType.COMBINED:
            # Special case for combined controller
            return self._compute_combined_control()

        return self.controllers[self.active_controller_type].compute_control()

    def _compute_combined_control(self) -> np.ndarray:
        """
        Compute control inputs by combining multiple controllers.

        DEPRECATED: This method is no longer used. The step() method now directly
        selects the appropriate controller based on Jn_converged flag.

        Returns:
            Combined control inputs for all agents
        """
        # If formation has converged, use behavior control
        if self.swarm_state.Jn_converged:
            print(
                f"DEBUG: Using BEHAVIOR controller at iteration {self.swarm_state.iteration}"
            )
            return self.controllers[ControllerType.BEHAVIOR].compute_control()

        # Otherwise use communication-aware control
        print(
            f"DEBUG: Using COMMUNICATION controller at iteration {self.swarm_state.iteration}"
        )
        return self.controllers[ControllerType.COMMUNICATION].compute_control()

    def step(self):
        """
        Perform a control step using the active controller.

        This method computes control inputs, applies them, and updates
        the swarm state for the next iteration.
        """

        # Special case for combined controller
        if self.active_controller_type == ControllerType.COMBINED:
            # Update matrices regardless of which controller we use
            self.swarm_state.update_matrices()

            control_inputs = np.zeros((self.swarm_state.swarm_size, 2))

            if self.swarm_state.Jn_converged:
                # After convergence, use both controllers
                comm_controller = self.controllers[ControllerType.COMMUNICATION]
                behav_controller = self.controllers[ControllerType.BEHAVIOR]

                print(
                    "DEBUG: Using BOTH controllers (communication + behavior) after convergence"
                )

                # Get control inputs from both controllers
                comm_inputs = comm_controller.compute_control()
                behav_inputs = behav_controller.compute_control()

                # Combine control inputs (weighted sum)
                control_inputs = 0.3 * comm_inputs + 0.7 * behav_inputs

                # Apply combined control inputs using the base method
                self.swarm_state.swarm_control_ui = control_inputs
                self.swarm_state.swarm_position += control_inputs
            else:
                # Before convergence, use only communication controller
                comm_controller = self.controllers[ControllerType.COMMUNICATION]
                print("DEBUG: Using ONLY communication controller before convergence")
                control_inputs = comm_controller.compute_control()
                comm_controller.apply_control(control_inputs)

            # Update performance metrics
            self.swarm_state.update_performance_metrics()

            # Store current positions for trajectory visualization
            self.swarm_state.update_swarm_paths()

            # Increment iteration counter
            self.swarm_state.iteration += 1

        # For single controllers
        elif self.active_controller_type == ControllerType.COMMUNICATION:
            self.controllers[ControllerType.COMMUNICATION].update_swarm_state()
        else:
            # For other controllers without specific update methods
            self.swarm_state.update_matrices()
            control_inputs = self.compute_control()
            self.controllers[self.active_controller_type].apply_control(control_inputs)
            self.swarm_state.update_performance_metrics()
            self.swarm_state.update_swarm_paths()
            self.swarm_state.iteration += 1

    # LLM intervention hooks - to be implemented in the future
    def llm_override_control(self, agent_indices, control_inputs):
        """
        Hook for LLM to override control for specific agents.

        Args:
            agent_indices: Indices of agents to override
            control_inputs: New control inputs for these agents
        """
        # This is a placeholder for future LLM integration
        pass
