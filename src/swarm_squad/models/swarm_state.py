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
        for i in range(self.swarm_size):
            for j in range(self.swarm_size):
                if i != j:
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

    def update_performance_metrics(self):
        """Calculate and store performance metrics"""
        Jn_new = utils.calculate_Jn(
            self.communication_qualities_matrix, self.neighbor_agent_matrix, config.PT
        )
        rn_new = utils.calculate_rn(
            self.distances_matrix, self.neighbor_agent_matrix, config.PT
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

    def check_destination_reached(self, threshold=0.05) -> bool:
        """Check if the swarm has reached its destination"""
        swarm_center = np.mean(self.swarm_position, axis=0)
        dist_to_dest = np.linalg.norm(swarm_center - self.swarm_destination)
        return dist_to_dest < threshold
