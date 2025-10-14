"""
Path Planning Controller for 3D Formation Control

Integrates path planning algorithms with the formation control system.
Agents follow planned paths while maintaining formation.
"""

import numpy as np
from path_planning_3d import PathPlanner3D


class PathPlanningController3D:
    """
    Controller that uses path planning algorithms to navigate to destination
    while maintaining formation.
    """

    def __init__(
        self,
        swarm_state,
        algorithm="astar",
        voxel_size=3.0,
        replan_threshold=10.0,
        waypoint_threshold=5.0,
        detection_radius=15.0,
    ):
        """
        Initialize path planning controller.

        Args:
            swarm_state: Reference to swarm state (has swarm_position, obstacles, etc.)
            algorithm: Path planning algorithm to use
            voxel_size: Voxel size for grid discretization
            replan_threshold: Distance threshold for replanning (if path becomes invalid)
            waypoint_threshold: Distance to consider waypoint reached
            detection_radius: Distance at which agents can detect obstacles
        """
        print(f"PathPlanningController3D initialized with algorithm: {algorithm}")
        print(f"  Dynamic replanning enabled (detection radius: {detection_radius})")
        self.swarm_state = swarm_state
        self.algorithm = algorithm
        self.voxel_size = voxel_size
        self.replan_threshold = replan_threshold
        self.waypoint_threshold = waypoint_threshold
        self.detection_radius = detection_radius

        # Path planning state
        self.paths = {}  # Agent ID -> path waypoints
        self.current_waypoints = {}  # Agent ID -> current waypoint index
        self.needs_replan = set()  # Set of agent IDs that need replanning

        # Dynamic obstacle discovery
        self.discovered_obstacles = []  # Obstacles that have been detected by any agent
        self.agent_detected_obstacles = {}  # Agent ID -> set of obstacle indices detected

        # Create path planner (will be initialized with bounds from swarm state)
        self.planner = None
        self._initialize_planner()

    def _initialize_planner(self):
        """Initialize path planner with appropriate bounds."""
        # Calculate bounds from swarm and destination
        positions = self.swarm_state.swarm_position
        dest = self.swarm_state.swarm_destination

        # Expand bounds to include current positions and destination
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)

        bounds_min = np.minimum(min_pos, dest) - 50
        bounds_max = np.maximum(max_pos, dest) + 50

        self.planner = PathPlanner3D(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            voxel_size=self.voxel_size,
            algorithm=self.algorithm,
            diagonal_movement=True,
        )

        # Start with NO obstacles - they will be discovered dynamically
        self.planner.update_obstacles([])
        print("  Starting with no known obstacles - dynamic discovery mode")

    def set_algorithm(self, algorithm):
        """Change the path planning algorithm."""
        self.algorithm = algorithm
        if self.planner:
            self.planner.set_algorithm(algorithm)
        # Trigger replanning for all agents
        self.needs_replan = set(range(self.swarm_state.swarm_size))

    def update_obstacles(self):
        """Update planner with current obstacles."""
        if self.planner:
            self.planner.update_obstacles(self.swarm_state.obstacles)
            # Trigger replanning for all agents
            self.needs_replan = set(range(self.swarm_state.swarm_size))

    def plan_path(self, agent_index):
        """
        Plan path for a specific agent from current position to destination.

        Args:
            agent_index: Index of agent to plan for

        Returns:
            bool: True if path found, False otherwise
        """
        start = self.swarm_state.swarm_position[agent_index]
        end = self.swarm_state.swarm_destination

        path, algo, nodes = self.planner.find_path(start, end)

        if path and len(path) > 1:
            self.paths[agent_index] = path
            self.current_waypoints[agent_index] = 0
            print(
                f"Agent {agent_index}: Path planned with {len(path)} waypoints using {algo} ({nodes} nodes)"
            )
            return True
        else:
            print(f"Agent {agent_index}: No path found!")
            self.paths[agent_index] = None
            return False

    def compute_control(self):
        """
        Calculate control inputs for all agents based on planned paths.
        Includes dynamic obstacle detection and replanning.

        Returns:
            A numpy array of shape (swarm_size, 3) containing control inputs
        """
        control_inputs = np.zeros((self.swarm_state.swarm_size, 3))

        # Check for obstacle detection by any agent
        obstacles_detected = self._detect_obstacles()
        if obstacles_detected:
            print(
                f"  🔍 New obstacles detected! Total discovered: {len(self.discovered_obstacles)}"
            )
            # Update planner with discovered obstacles
            self.planner.update_obstacles(self.discovered_obstacles)
            # Trigger replanning for all agents
            self.needs_replan = set(range(self.swarm_state.swarm_size))

        # Track which agents are in emergency mode
        emergency_agents = set()

        # First, apply emergency obstacle avoidance for any agent too close to obstacles
        for i in range(self.swarm_state.swarm_size):
            emergency_control = self._emergency_obstacle_avoidance(i)
            if emergency_control is not None:
                control_inputs[i] = emergency_control
                emergency_agents.add(i)

        for i in range(self.swarm_state.swarm_size):
            # Skip if in emergency mode
            if i in emergency_agents:
                continue

            # Check if agent needs replanning
            if i in self.needs_replan or i not in self.paths:
                self.plan_path(i)
                if i in self.needs_replan:
                    self.needs_replan.remove(i)

            # Get agent's path
            if i not in self.paths or self.paths[i] is None:
                # No path available, use direct control to destination
                control_inputs[i] = self._direct_destination_control(i)
                continue

            path = self.paths[i]
            waypoint_idx = self.current_waypoints[i]

            # Check if reached final destination
            if waypoint_idx >= len(path):
                # Already at destination, no control needed
                continue

            # Get current target waypoint
            target = path[waypoint_idx]
            agent_pos = self.swarm_state.swarm_position[i]

            # Calculate distance to current waypoint
            distance_to_waypoint = np.linalg.norm(target - agent_pos)

            # Check if reached current waypoint
            if distance_to_waypoint < self.waypoint_threshold:
                # Move to next waypoint
                self.current_waypoints[i] += 1
                if self.current_waypoints[i] < len(path):
                    target = path[self.current_waypoints[i]]
                else:
                    # Reached final destination
                    continue

            # Calculate control input toward target waypoint
            direction = target - agent_pos
            distance = np.linalg.norm(direction)

            if distance > 0:
                direction = direction / distance

                # Control magnitude - strong enough to overcome formation control
                control_magnitude = 2.0  # Increased to 2.0 to ensure clear movement

                # Slow down when approaching waypoint
                if distance < self.waypoint_threshold * 2:
                    control_magnitude *= distance / (self.waypoint_threshold * 2)

                control_inputs[i] = direction * control_magnitude

        return control_inputs

    def _emergency_obstacle_avoidance(self, agent_index):
        """
        Check if agent is dangerously close to discovered obstacles and apply emergency avoidance.

        Args:
            agent_index: Index of agent to check

        Returns:
            numpy.ndarray: Emergency control input if needed, None otherwise
        """
        agent_pos = self.swarm_state.swarm_position[agent_index]

        # Check all discovered obstacles
        for obstacle in self.discovered_obstacles:
            obstacle_pos = np.array([obstacle[0], obstacle[1], obstacle[2]])
            obstacle_radius = obstacle[3]

            # Calculate distance to obstacle center
            distance = np.linalg.norm(agent_pos - obstacle_pos)

            # Emergency zone: within 2 units of obstacle surface
            emergency_distance = obstacle_radius + 2.0

            if distance < emergency_distance:
                # Apply strong repulsive force
                avoidance_direction = (agent_pos - obstacle_pos) / distance

                # Exponential force - stronger when closer
                force_magnitude = 3.0 * np.exp(-0.5 * (distance - obstacle_radius))

                return avoidance_direction * force_magnitude

        return None  # No emergency avoidance needed

    def _detect_obstacles(self):
        """
        Check if any agent has detected new obstacles.

        Returns:
            bool: True if new obstacles were detected
        """
        new_obstacles_found = False

        # Iterate through all actual obstacles in the environment
        for obs_idx, obstacle in enumerate(self.swarm_state.obstacles):
            # Skip if already discovered
            if obs_idx < len(self.discovered_obstacles):
                continue

            obstacle_pos = np.array([obstacle[0], obstacle[1], obstacle[2]])
            obstacle_radius = obstacle[3]

            # Check if any agent is within detection radius
            for agent_idx in range(self.swarm_state.swarm_size):
                agent_pos = self.swarm_state.swarm_position[agent_idx]
                distance = np.linalg.norm(agent_pos - obstacle_pos)

                # Detection occurs when agent is within detection_radius of obstacle surface
                if distance <= (obstacle_radius + self.detection_radius):
                    # Initialize agent's detected obstacles set if needed
                    if agent_idx not in self.agent_detected_obstacles:
                        self.agent_detected_obstacles[agent_idx] = set()

                    # Check if this agent hasn't detected this obstacle yet
                    if obs_idx not in self.agent_detected_obstacles[agent_idx]:
                        self.agent_detected_obstacles[agent_idx].add(obs_idx)
                        self.discovered_obstacles.append(obstacle)
                        new_obstacles_found = True
                        print(
                            f"    Agent {agent_idx} detected obstacle {obs_idx + 1} at distance {distance:.1f}"
                        )
                        break  # One detection is enough to add the obstacle

        return new_obstacles_found

    def _direct_destination_control(self, agent_index, weight=1.0):
        """Fallback: Direct control to destination when no path available."""
        am = 0.7
        bm = 1.0

        destination_vector = (
            self.swarm_state.swarm_destination
            - self.swarm_state.swarm_position[agent_index]
        )
        dist_to_dest = np.linalg.norm(destination_vector)

        if dist_to_dest > 0:
            destination_direction = destination_vector / dist_to_dest

            if dist_to_dest > bm:
                control_param = am
            else:
                control_param = am * (dist_to_dest / bm)

            return weight * destination_direction * control_param

        return np.zeros(3)

    def visualize_paths(self, ax):
        """
        Visualize planned paths on a 3D axis.
        Also shows discovered vs undiscovered obstacles.

        Args:
            ax: Matplotlib 3D axis
        """
        # Draw detection radius circles around agents (only for first agent to avoid clutter)
        if len(self.swarm_state.swarm_position) > 0:
            agent_pos = self.swarm_state.swarm_position[0]
            # Draw a sphere representing detection radius
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = agent_pos[0] + self.detection_radius * np.outer(np.cos(u), np.sin(v))
            y = agent_pos[1] + self.detection_radius * np.outer(np.sin(u), np.sin(v))
            z = agent_pos[2] + self.detection_radius * np.outer(
                np.ones(np.size(u)), np.cos(v)
            )
            ax.plot_wireframe(x, y, z, color="cyan", alpha=0.1, linewidth=0.5)

        # Visualize paths
        for agent_idx, path in self.paths.items():
            if path is not None and len(path) > 1:
                path_array = np.array(path)
                ax.plot(
                    path_array[:, 0],
                    path_array[:, 1],
                    path_array[:, 2],
                    linestyle=":",
                    linewidth=2,
                    alpha=0.6,
                    color="cyan",
                    label=f"Agent {agent_idx} path" if agent_idx == 0 else "",
                )

                # Mark current waypoint
                if agent_idx in self.current_waypoints:
                    wp_idx = self.current_waypoints[agent_idx]
                    if wp_idx < len(path):
                        wp = path[wp_idx]
                        ax.scatter(
                            [wp[0]],
                            [wp[1]],
                            [wp[2]],
                            color="yellow",
                            s=100,
                            marker="*",
                            label="Current waypoint" if agent_idx == 0 else "",
                        )


class SwarmStateMock:
    """Mock swarm state for testing."""

    def __init__(self):
        self.swarm_size = 4
        self.swarm_position = np.array(
            [
                [-5, 14, 0],
                [-5, -19, 5],
                [0, 0, -5],
                [35, -4, 0],
            ],
            dtype=float,
        )
        self.swarm_destination = np.array([35, 150, 30], dtype=float)
        self.obstacles = [
            (35, 75, 15, 20),
        ]


def test_path_planning_controller():
    """Test the path planning controller."""
    print("=" * 60)
    print("Testing Path Planning Controller")
    print("=" * 60)

    # Create mock swarm state
    swarm_state = SwarmStateMock()

    # Create controller with A*
    controller = PathPlanningController3D(
        swarm_state, algorithm="astar", voxel_size=2.0, waypoint_threshold=3.0
    )

    # Compute control for first iteration
    print("\nComputing initial control inputs...")
    control = controller.compute_control()

    print(f"\nControl inputs (shape: {control.shape}):")
    for i in range(swarm_state.swarm_size):
        print(f"  Agent {i}: {control[i]}")

    # Test algorithm switching
    print("\n" + "=" * 60)
    print("Testing algorithm switching...")
    print("=" * 60)

    for algo in ["dijkstra", "bfs", "greedy"]:
        print(f"\nSwitching to {algo}...")
        controller.set_algorithm(algo)
        control = controller.compute_control()
        print(f"Control computed successfully with {algo}")


if __name__ == "__main__":
    test_path_planning_controller()
