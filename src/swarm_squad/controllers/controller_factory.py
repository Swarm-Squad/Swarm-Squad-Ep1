"""
Controller factory to manage different controller types and provide integration
hooks for LLM intervention.
"""

from enum import Enum
from typing import Dict

import numpy as np

import swarm_squad.config as config
from swarm_squad.controllers.base_controller import BaseController
from swarm_squad.controllers.behavior_controller import BehaviorController
from swarm_squad.controllers.formation_controller import FormationController
from swarm_squad.controllers.llm_controller import LLMController
from swarm_squad.models.swarm_state import SwarmState


class ControllerType(Enum):
    """Enum for different controller types"""

    FORMATION = "formation"
    BEHAVIOR = "behavior"
    COMBINED = "combined"
    LLM = "llm"  # LLM controller


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
        self.active_controller_type = ControllerType.FORMATION

        # Initialize controllers
        self._init_controllers()

    def _init_controllers(self):
        """Initialize all available controllers"""
        # Core controllers
        self.controllers[ControllerType.FORMATION] = FormationController(
            self.swarm_state
        )
        self.controllers[ControllerType.BEHAVIOR] = BehaviorController(self.swarm_state)

        # Initialize LLM controller - with debug print
        print("### Initializing LLM controller")
        llm_controller = LLMController(self.swarm_state)

        # Set combined controller as default for LLM controller
        # to fall back on when not actively providing control
        llm_controller.set_default_controller(self)

        self.controllers[ControllerType.LLM] = llm_controller
        print("### LLM controller initialized")

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
        # print(f"DEBUG: Setting active controller to {controller_type}")

        if controller_type == ControllerType.COMBINED:
            self.active_controller_type = ControllerType.COMBINED
            # print(f"DEBUG: Active controller is now {self.active_controller_type}")
        elif controller_type in self.controllers:
            self.active_controller_type = controller_type
            # print(f"DEBUG: Active controller is now {self.active_controller_type}")
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
                # f"DEBUG: Using BEHAVIOR controller at iteration {self.swarm_state.iteration}"
            )
            return self.controllers[ControllerType.BEHAVIOR].compute_control()

        # Otherwise use communication-aware formation control
        print(
            # f"DEBUG: Using FORMATION controller at iteration {self.swarm_state.iteration}"
        )
        return self.controllers[ControllerType.FORMATION].compute_control()

    def step(self):
        """
        Perform a control step using the active controller.

        This method computes control inputs, applies them, and updates
        the swarm state for the next iteration.
        """
        # Always update LLM controller to receive feedback, regardless of whether
        # it's the active controller - add debug print
        llm_controller = self.controllers[ControllerType.LLM]
        # print(f"### Controller Factory step() at iteration {self.swarm_state.iteration}, getting LLM controller")

        # Call LLM controller's compute_control method to let it process feedback
        # This is separate from actual control execution
        _ = llm_controller.compute_control()

        # Special case for combined controller
        if self.active_controller_type == ControllerType.COMBINED:
            # Update matrices regardless of which controller we use
            self.swarm_state.update_matrices()

            control_inputs = np.zeros((self.swarm_state.swarm_size, 2))

            # For high power jamming, let formation controller handle it for returning agents
            # but still use behavior controller for destination reaching after convergence
            if config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING:
                # After convergence, use both controllers for active agents
                if self.swarm_state.Jn_converged:
                    # Get formation control inputs
                    comm_controller = self.controllers[ControllerType.FORMATION]
                    comm_inputs = comm_controller.compute_control()

                    # Get behavior (destination) control inputs
                    behav_controller = self.controllers[ControllerType.BEHAVIOR]
                    behav_inputs = behav_controller.compute_control()

                    # Combine control inputs (weighted sum)
                    control_inputs = 0.3 * comm_inputs + 0.7 * behav_inputs

                    # Apply combined control inputs using the base method
                    self.swarm_state.swarm_control_ui = control_inputs
                    self.swarm_state.swarm_position += control_inputs
                else:
                    # Before convergence, use only formation controller
                    comm_controller = self.controllers[ControllerType.FORMATION]
                    comm_controller.update_swarm_state()

            # For all other obstacle modes (hard and low power jamming)
            else:
                # After convergence, use both controllers for improved obstacle avoidance and destination reaching
                if self.swarm_state.Jn_converged:
                    # After convergence, use both controllers
                    comm_controller = self.controllers[ControllerType.FORMATION]
                    behav_controller = self.controllers[ControllerType.BEHAVIOR]

                    print(
                        # "DEBUG: Using BOTH controllers (formation + behavior) after convergence"
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
                    # Before convergence, use only formation controller
                    comm_controller = self.controllers[ControllerType.FORMATION]
                    # print("DEBUG: Using ONLY formation controller before convergence")
                    control_inputs = comm_controller.compute_control()
                    comm_controller.apply_control(control_inputs)

            # Update performance metrics
            self.swarm_state.update_performance_metrics()

            # Store current positions for trajectory visualization
            self.swarm_state.update_swarm_paths()

            # Increment iteration counter
            self.swarm_state.iteration += 1

        # For single controllers
        elif self.active_controller_type == ControllerType.FORMATION:
            self.controllers[ControllerType.FORMATION].update_swarm_state()
        else:
            # For other controllers without specific update methods
            self.swarm_state.update_matrices()
            control_inputs = self.compute_control()
            self.controllers[self.active_controller_type].apply_control(control_inputs)
            self.swarm_state.update_performance_metrics()
            self.swarm_state.update_swarm_paths()
            self.swarm_state.iteration += 1

    # LLM intervention hooks
    def llm_override_control(self, agent_indices, control_inputs):
        """
        Hook for LLM to override control for specific agents.

        Args:
            agent_indices: Indices of agents to override
            control_inputs: New control inputs for these agents
        """
        # Store the original controls
        original_controls = self.swarm_state.swarm_control_ui.copy()

        # Override controls for specified agents
        for idx, control in zip(agent_indices, control_inputs):
            if idx < self.swarm_state.swarm_size:
                self.swarm_state.swarm_control_ui[idx] = control

        # Return original controls for reference
        return original_controls
