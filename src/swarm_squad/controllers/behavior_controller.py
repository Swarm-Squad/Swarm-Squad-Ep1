"""
Behavior-based controller implementing obstacle avoidance, wall following,
and destination control behaviors.
"""

import numpy as np

import swarm_squad.config as config
from swarm_squad.controllers.base_controller import BaseController
from swarm_squad.models.swarm_state import SwarmState


class BehaviorController(BaseController):
    """
    Controller that implements behavior-based control strategies.

    This controller handles behaviors such as:
    - Obstacle avoidance
    - Wall following
    - Destination reaching
    - Return-to-launch (RTL) behavior for jammed agents
    """

    def __init__(self, swarm_state: SwarmState):
        """
        Initialize the behavior controller.

        Args:
            swarm_state: Reference to the swarm state object
        """
        print("BREAKPOINT: BehaviorController initialized")
        super().__init__(swarm_state)

    def compute_control(self) -> np.ndarray:
        """
        Calculate behavior-based control inputs for all agents.

        Returns:
            A numpy array of shape (swarm_size, 2) containing the control
            inputs for each agent in the swarm.
        """
        # print(
        #     f"BREAKPOINT: BehaviorController.compute_control called at iteration {self.swarm_state.iteration}"
        # )
        # Reset control inputs
        control_inputs = np.zeros((self.swarm_state.swarm_size, 2))

        # Apply behavior-based control for each agent
        for i in range(self.swarm_state.swarm_size):
            # For agents affected by high-power jamming, return to launch
            if not self.swarm_state.agent_status[i]:
                self._add_rtl_behavior(control_inputs, i)
                continue

            # Normal behavior for active agents
            has_obstacle_influence = False

            # Only consider hard obstacles if in hard obstacle mode
            if config.OBSTACLE_MODE == config.ObstacleMode.HARD:
                # Check for obstacle collisions and apply avoidance
                for obstacle in self.swarm_state.obstacles:
                    obstacle_pos = np.array([obstacle[0], obstacle[1]])
                    obstacle_radius = obstacle[2]

                    # Calculate distance to obstacle center
                    dist_to_center = np.linalg.norm(
                        self.swarm_state.swarm_position[i] - obstacle_pos
                    )

                    # Define buffer zones
                    buffer_zone = obstacle_radius + 6.0
                    wall_follow_zone = obstacle_radius + 3.0

                    if dist_to_center < buffer_zone:  # If within buffer zone
                        has_obstacle_influence = True
                        if dist_to_center < wall_follow_zone:
                            # Apply strong avoidance when very close
                            self._add_obstacle_avoidance(
                                control_inputs, i, obstacle_pos, obstacle_radius
                            )
                            # Minimal destination control when very close to obstacle
                            self._add_destination_control(control_inputs, i, weight=0.3)
                        else:
                            # Apply wall following when in outer buffer zone
                            wall_normal = (
                                self.swarm_state.swarm_position[i] - obstacle_pos
                            ) / dist_to_center
                            wall_pos = obstacle_pos + wall_normal * obstacle_radius
                            self._add_wall_following(
                                control_inputs, i, wall_pos, wall_normal
                            )
                            # Reduced destination control during wall following
                            self._add_destination_control(control_inputs, i, weight=0.4)

            # If not influenced by any obstacle, apply normal destination control
            if not has_obstacle_influence:
                self._add_destination_control(control_inputs, i, weight=1.0)

        return control_inputs

    def _add_rtl_behavior(self, control_inputs: np.ndarray, agent_index: int):
        """
        Add return-to-launch control input for an agent affected by jamming.

        Args:
            control_inputs: The array of control inputs to modify
            agent_index: Index of the agent to control
        """
        # RTL parameters
        rtl_magnitude = 0.8  # Slightly slower return speed

        # Calculate vector to initial position
        rtl_vector = (
            self.swarm_state.initial_positions[agent_index]
            - self.swarm_state.swarm_position[agent_index]
        )
        dist_to_home = np.linalg.norm(rtl_vector)

        # If very close to home, stop
        if dist_to_home < 0.5:
            control_inputs[agent_index] = np.zeros(2)
            return

        # Calculate direction and apply speed
        if dist_to_home > 0:  # Avoid division by zero
            rtl_direction = rtl_vector / dist_to_home

            # Scale control input based on distance
            control_param = min(rtl_magnitude, dist_to_home * 0.1)

            # Apply to control input
            control_inputs[agent_index] = rtl_direction * control_param

    def _add_destination_control(
        self, control_inputs: np.ndarray, agent_index: int, weight=1.0
    ):
        """
        Add destination-reaching control input for an agent.

        Args:
            control_inputs: The array of control inputs to modify
            agent_index: Index of the agent to control
            weight: Weight factor for the control input (0.0-1.0)
        """
        # Parameters for destination control
        am = config.DESTINATION_ATTRACTION_MAGNITUDE
        bm = config.DESTINATION_DISTANCE_THRESHOLD

        # Calculate vector to destination
        destination_vector = (
            self.swarm_state.swarm_destination
            - self.swarm_state.swarm_position[agent_index]
        )
        dist_to_dest = np.linalg.norm(destination_vector)

        if dist_to_dest > 0:  # Avoid division by zero
            destination_direction = destination_vector / dist_to_dest

            # Scale control input based on distance
            if dist_to_dest > bm:
                control_param = am
            else:
                control_param = am * (dist_to_dest / bm)

            # Apply weight to control input
            control_update = weight * destination_direction * control_param
            control_inputs[agent_index] += control_update

    def _add_obstacle_avoidance(
        self,
        control_inputs: np.ndarray,
        agent_index: int,
        obstacle_position: np.ndarray,
        obstacle_radius: float,
    ):
        """
        Add obstacle avoidance control input for an agent.

        Args:
            control_inputs: The array of control inputs to modify
            agent_index: Index of the agent to control
            obstacle_position: Position of the obstacle
            obstacle_radius: Radius of the obstacle
        """
        # Avoidance parameters
        ao = config.OBSTACLE_AVOIDANCE_MAGNITUDE
        bo = config.OBSTACLE_INFLUENCE_RANGE

        # Calculate vector away from the obstacle
        obstacle_vector = (
            self.swarm_state.swarm_position[agent_index] - obstacle_position
        )
        dist_to_obstacle = np.linalg.norm(obstacle_vector)

        if dist_to_obstacle < (obstacle_radius + bo):
            avoidance_direction = obstacle_vector / dist_to_obstacle

            # Stronger exponential scaling for more aggressive close-range avoidance
            proximity_factor = np.exp(-0.3 * (dist_to_obstacle - obstacle_radius))
            control_param = (
                ao
                * proximity_factor
                * (1 + 1 / (dist_to_obstacle - obstacle_radius + 0.1))
            )

            # Add to existing control input
            control_inputs[agent_index] += avoidance_direction * control_param

    def _add_wall_following(
        self,
        control_inputs: np.ndarray,
        agent_index: int,
        wall_position: np.ndarray,
        wall_normal: np.ndarray,
    ):
        """
        Add wall-following control input for an agent.

        Args:
            control_inputs: The array of control inputs to modify
            agent_index: Index of the agent to control
            wall_position: Position on the wall closest to the agent
            wall_normal: Normal vector perpendicular to the wall
        """
        # Wall following parameters
        af = config.WALL_FOLLOWING_MAGNITUDE
        df = config.WALL_DISTANCE

        # Calculate perpendicular distance to wall
        agent_position = self.swarm_state.swarm_position[agent_index]
        distance_to_wall = np.dot(agent_position - wall_position, wall_normal)

        # Calculate tangent direction (clockwise around obstacle)
        tangent_direction = np.array([-wall_normal[1], wall_normal[0]])

        # Enhanced wall following behavior
        if abs(distance_to_wall) > df:
            # Stronger correction when too close or too far from wall
            correction = -np.sign(distance_to_wall) * wall_normal
            # Increase correction influence
            control = af * (0.4 * tangent_direction + 0.6 * correction)
        else:
            # Stronger wall following when at good distance
            control = 1.2 * af * tangent_direction

        control_inputs[agent_index] += control
