"""
SwarmState class to manage the state of the swarm.
"""

import time
from typing import List, Tuple

import numpy as np

import swarm_squad.config as config
import swarm_squad.utils as utils


class SwarmState:
    """
    Manages the state of the swarm, including positions, communication quality,
    performance metrics, and obstacles.
    """

    def __init__(self):
        # Initialize swarm positions and parameters
        self.swarm_position = config.INITIAL_SWARM_POSITIONS.copy()
        self.swarm_destination = config.DEFAULT_DESTINATION.copy()
        self.swarm_size = self.swarm_position.shape[0]
        self.swarm_control_ui = np.zeros((self.swarm_size, 2))

        # Store initial positions for return-to-launch behavior
        self.initial_positions = config.INITIAL_SWARM_POSITIONS.copy()

        # Agent status tracking
        self.agent_status = np.ones(
            self.swarm_size, dtype=bool
        )  # True = active, False = returning
        self.jamming_affected = np.zeros(
            self.swarm_size, dtype=bool
        )  # Track which agents are affected by jamming

        # Performance indicators
        self.Jn = []
        self.rn = []
        self.t_elapsed = []
        self.start_time = time.time()

        # Initialize matrices
        self.communication_qualities_matrix = np.zeros(
            (self.swarm_size, self.swarm_size)
        )
        self.distances_matrix = np.zeros((self.swarm_size, self.swarm_size))
        self.neighbor_agent_matrix = np.zeros((self.swarm_size, self.swarm_size))

        # Paths and obstacles
        self.swarm_paths = []
        self.obstacles: List[Tuple[float, float, float]] = []  # (x, y, radius)

        # Simulation state
        self.iteration = 0
        self.Jn_converged = False
        self.line_colors = np.random.rand(self.swarm_size, self.swarm_size, 3)

    def reset(self):
        """Reset the swarm state to initial conditions"""
        self.swarm_position = config.INITIAL_SWARM_POSITIONS.copy()
        self.swarm_control_ui = np.zeros((self.swarm_size, 2))

        # Reset agent status
        self.agent_status = np.ones(self.swarm_size, dtype=bool)
        self.jamming_affected = np.zeros(self.swarm_size, dtype=bool)

        # Reset performance indicators
        self.Jn = []
        self.rn = []
        self.t_elapsed = []
        self.start_time = time.time()

        # Reset matrices
        self.communication_qualities_matrix = np.zeros(
            (self.swarm_size, self.swarm_size)
        )
        self.distances_matrix = np.zeros((self.swarm_size, self.swarm_size))
        self.neighbor_agent_matrix = np.zeros((self.swarm_size, self.swarm_size))

        # Reset paths
        self.swarm_paths = []

        # Reset simulation state
        self.iteration = 0
        self.Jn_converged = False

    def add_obstacle(self, x: float, y: float, radius: float):
        """Add an obstacle to the environment"""
        self.obstacles.append((x, y, radius))

    def remove_last_obstacle(self):
        """Remove the last added obstacle"""
        if self.obstacles:
            self.obstacles.pop()

    def update_swarm_paths(self):
        """Store the current positions for trajectory visualization"""
        self.swarm_paths.append(self.swarm_position.copy())

    def update_matrices(self):
        """Update distance and communication matrices"""
        # First, apply obstacle effects to update agent statuses
        self.apply_obstacle_effects()

        # Clear matrices
        self.communication_qualities_matrix = np.zeros(
            (self.swarm_size, self.swarm_size)
        )
        self.distances_matrix = np.zeros((self.swarm_size, self.swarm_size))
        self.neighbor_agent_matrix = np.zeros((self.swarm_size, self.swarm_size))

        # Calculate matrices only for active agents
        for i in range(self.swarm_size):
            # For high power jamming, completely exclude inactive agents
            if (
                config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING
                and not self.agent_status[i]
            ):
                continue

            for j in range(self.swarm_size):
                if i != j:
                    # For high power jamming, completely exclude inactive agents
                    if (
                        config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING
                        and not self.agent_status[j]
                    ):
                        continue

                    rij = utils.calculate_distance(
                        self.swarm_position[i], self.swarm_position[j]
                    )
                    aij = utils.calculate_aij(
                        config.ALPHA, config.DELTA, rij, config.R0, config.V
                    )
                    gij = utils.calculate_gij(rij, config.R0)

                    # Record matrices
                    phi_rij = gij * aij
                    self.communication_qualities_matrix[i, j] = phi_rij
                    self.distances_matrix[i, j] = rij
                    self.neighbor_agent_matrix[i, j] = aij

        # For low power jamming, apply degradation after calculation
        if (
            self.obstacles
            and config.OBSTACLE_MODE == config.ObstacleMode.LOW_POWER_JAMMING
        ):
            self.apply_lowpower_jamming()

    def apply_obstacle_effects(self):
        """Apply the effects of obstacles based on the current obstacle mode"""
        # Reset jamming effects
        self.jamming_affected.fill(False)

        # Skip if no obstacles
        if not self.obstacles:
            return

        affected_agents = []

        # Update agent statuses based on obstacle mode
        for i in range(self.swarm_size):
            for obstacle in self.obstacles:
                obstacle_pos = np.array([obstacle[0], obstacle[1]])
                obstacle_radius = obstacle[2]
                jamming_radius = obstacle_radius * config.JAMMING_RADIUS_MULTIPLIER

                # Calculate distance to obstacle center
                dist_to_center = np.linalg.norm(self.swarm_position[i] - obstacle_pos)

                # Apply effects based on obstacle mode
                if config.OBSTACLE_MODE == config.ObstacleMode.LOW_POWER_JAMMING:
                    if dist_to_center < jamming_radius:
                        # Mark as affected by jamming
                        self.jamming_affected[i] = True
                        affected_agents.append(i)

                        # Store distance to jamming center for gradual effect (normalized)
                        # Closer to center = higher interference = lower value
                        # This will be used in apply_lowpower_jamming
                        penetration_depth = 1.0 - max(
                            0,
                            (dist_to_center - obstacle_radius)
                            / (jamming_radius - obstacle_radius),
                        )

                        # Store the penetration depth for this agent (used later in apply_lowpower_jamming)
                        if not hasattr(self, "jamming_depth"):
                            self.jamming_depth = np.zeros(self.swarm_size)

                        # Use the maximum depth if multiple obstacles affect the same agent
                        self.jamming_depth[i] = max(
                            self.jamming_depth[i], penetration_depth
                        )

                elif config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING:
                    if dist_to_center < jamming_radius:
                        # Mark agent as returning to launch position
                        self.agent_status[i] = False
                        self.jamming_affected[i] = True
                        affected_agents.append(i)

        # Print debug information about affected agents
        if affected_agents:
            jamming_type = (
                "Low Power"
                if config.OBSTACLE_MODE == config.ObstacleMode.LOW_POWER_JAMMING
                else "High Power"
            )
            print(f"DEBUG: {jamming_type} Jamming affects agents: {affected_agents}")

            if config.OBSTACLE_MODE == config.ObstacleMode.LOW_POWER_JAMMING:
                depths = [f"{i}: {self.jamming_depth[i]:.2f}" for i in affected_agents]
                print(f"      Penetration depths: {', '.join(depths)}")
            else:
                print("      Affected agents returning to launch position")

    def apply_lowpower_jamming(self):
        """Apply low power jamming effects to degrade communication quality with gradual effect"""
        # Reset jamming depths if switching from other modes
        if not hasattr(self, "jamming_depth"):
            self.jamming_depth = np.zeros(self.swarm_size)
        elif config.OBSTACLE_MODE != config.ObstacleMode.LOW_POWER_JAMMING:
            self.jamming_depth.fill(0)

        for i in range(self.swarm_size):
            if self.jamming_affected[i]:
                # Get the penetration depth for this agent
                depth = self.jamming_depth[i]

                # Calculate degradation factor based on penetration depth
                # More penetration = more degradation
                # Min degradation = LOWPOWER_JAMMING_DEGRADATION (at edge)
                # Max degradation = LOWPOWER_JAMMING_DEGRADATION * 0.2 (at center)
                degradation_factor = config.LOWPOWER_JAMMING_DEGRADATION + (
                    1.0 - config.LOWPOWER_JAMMING_DEGRADATION
                ) * (1.0 - depth)

                # Ensure degradation doesn't go below minimum
                degradation_factor = max(0.2, degradation_factor)

                # Apply graduated degradation to communication quality
                for j in range(self.swarm_size):
                    if i != j:
                        # Only degrade if quality is above threshold
                        if self.communication_qualities_matrix[i, j] > 0:
                            self.communication_qualities_matrix[i, j] *= (
                                degradation_factor
                            )
                            self.communication_qualities_matrix[j, i] *= (
                                degradation_factor
                            )

                            # Debug output to show gradual degradation
                            if j == 0 and i == 1:  # Just show one pair as example
                                print(
                                    f"DEBUG: Agent {i}-{j} comm quality degraded to {self.communication_qualities_matrix[i, j]:.4f} (depth={depth:.2f}, factor={degradation_factor:.2f})"
                                )

    def update_performance_metrics(self):
        """Calculate and store performance metrics"""
        Jn_new = utils.calculate_Jn(
            self.communication_qualities_matrix,
            self.neighbor_agent_matrix,
            config.PT,
            agent_status=self.agent_status
            if config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING
            else None,
        )
        rn_new = utils.calculate_rn(
            self.distances_matrix,
            self.neighbor_agent_matrix,
            config.PT,
            agent_status=self.agent_status
            if config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING
            else None,
        )

        self.Jn.append(round(Jn_new, 4))
        self.rn.append(round(rn_new, 4))
        self.t_elapsed.append(time.time() - self.start_time)

    def check_convergence(self) -> bool:
        """
        Check if formation has converged based on Jn values
        Exactly matches the original implementation's check
        """
        if len(self.Jn) > 19:
            # Check if the last 20 values are all identical
            return len(set(self.Jn[-20:])) == 1
        return False

    def check_destination_reached(self, threshold=0.5) -> bool:
        """Check if the swarm has reached its destination"""
        # Only consider active agents for destination check
        active_positions = self.swarm_position[self.agent_status]

        if len(active_positions) == 0:
            return False

        swarm_center = np.mean(active_positions, axis=0)
        dist_to_dest = np.linalg.norm(swarm_center - self.swarm_destination)
        # print(
        #     f"\nDEBUG: Swarm center at {swarm_center}, destination at {self.swarm_destination}"
        # )
        # print(
        #     f"DEBUG: Distance to destination: {dist_to_dest:.2f} (threshold: {threshold})"
        # )
        return dist_to_dest < threshold
