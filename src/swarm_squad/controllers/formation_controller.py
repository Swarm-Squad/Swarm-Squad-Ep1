"""
Communication-aware controller that implements formation control logic.
"""

import numpy as np

import swarm_squad.config as config
import swarm_squad.utils as utils
from swarm_squad.controllers.base_controller import BaseController
from swarm_squad.models.swarm_state import SwarmState


class FormationController(BaseController):
    """
    Controller that implements communication-aware formation control logic.

    This controller maintains communication quality between agents by adjusting
    their positions to ensure connectivity while keeping a desired formation.
    """

    def __init__(self, swarm_state: SwarmState):
        """
        Initialize the formation controller.

        Args:
            swarm_state: Reference to the swarm state object
        """
        print("BREAKPOINT: FormationController initialized")
        super().__init__(swarm_state)

    def compute_control(self) -> np.ndarray:
        """
        Calculate control inputs for formation control.

        Returns:
            A numpy array of shape (swarm_size, 2) containing the control
            inputs for each agent in the swarm.
        """
        # print(
        #     f"BREAKPOINT: FormationController.compute_control called at iteration {self.swarm_state.iteration}"
        # )
        # Reset control inputs
        control_inputs = np.zeros((self.swarm_state.swarm_size, 2))

        # Formation control - only for active agents
        for i in range(self.swarm_state.swarm_size):
            # Skip agents affected by high-power jamming (returning to launch)
            if not self.swarm_state.agent_status[i]:
                continue

            for j in [
                x
                for x in range(self.swarm_state.swarm_size)
                if x != i and self.swarm_state.agent_status[x]
            ]:
                rij = utils.calculate_distance(
                    self.swarm_state.swarm_position[i],
                    self.swarm_state.swarm_position[j],
                )
                aij = utils.calculate_aij(
                    config.ALPHA, config.DELTA, rij, config.R0, config.V
                )

                # Only apply formation control if communication quality is above threshold
                if aij >= config.PT:
                    rho_ij = utils.calculate_rho_ij(
                        config.BETA, config.V, rij, config.R0
                    )
                else:
                    rho_ij = 0

                qi = self.swarm_state.swarm_position[i, :]
                qj = self.swarm_state.swarm_position[j, :]
                eij = (qi - qj) / np.sqrt(rij)

                # Formation control input
                control_inputs[i] += rho_ij * eij

        return control_inputs

    def update_swarm_state(self):
        """
        Update the swarm state based on agent interactions.

        This method calculates and updates the communication quality,
        distance matrices, and other state information.
        """
        # Update communication matrices
        self.swarm_state.update_matrices()

        # Compute and apply control inputs
        control_inputs = self.compute_control()

        # For agents affected by high-power jamming, get return-to-launch control
        # from the behavior controller
        if config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING:
            # Import here to avoid circular imports
            from swarm_squad.controllers.behavior_controller import BehaviorController

            rtl_controller = BehaviorController(self.swarm_state)

            for i in range(self.swarm_state.swarm_size):
                if not self.swarm_state.agent_status[i]:
                    # Calculate RTL control for returning agents
                    rtl_inputs = np.zeros((self.swarm_state.swarm_size, 2))
                    rtl_controller._add_rtl_behavior(rtl_inputs, i)
                    control_inputs[i] = rtl_inputs[i]

        self.apply_control(control_inputs)

        # Update performance metrics
        self.swarm_state.update_performance_metrics()

        # Store current positions for trajectory visualization
        self.swarm_state.update_swarm_paths()

        # Increment iteration counter
        self.swarm_state.iteration += 1
