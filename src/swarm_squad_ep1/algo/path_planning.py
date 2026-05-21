"""
Path planning algorithms for multi-vehicle navigation.

Uses PathPlanner3D from path_planning_3d.py which provides grid-based
3D path planning using the pathfinding3d library.

Available algorithms:
- A* (AStar)
- Theta* (Path Smoothing)
- Dijkstra
- Breadth-First Search (BFS)
- Greedy Best-First Search
- Bidirectional A*
- Minimum Spanning Tree (MSP)
"""

from typing import Any, Optional

import numpy as np

from swarm_squad_ep1.algo.base import JammingZone, ObstacleType
from swarm_squad_ep1.algo.custom_path_algorithms import (
    get_custom_path_algorithm,
    get_custom_path_callable,
    has_custom_path_algorithm,
    list_custom_path_algorithms,
    register_custom_path_algorithm,
    unregister_custom_path_algorithm,
)

# Try to import PathPlanner3D, fallback to simple implementation if not available
try:
    from swarm_squad_ep1.algo.path_planning_3d import PathPlanner3D

    PATHFINDING3D_AVAILABLE = True
except ImportError:
    PATHFINDING3D_AVAILABLE = False
    print("[PathPlanning] pathfinding3d not available, using fallback implementation")


# Available path planning algorithms
PATH_ALGORITHMS = [
    "direct",  # Direct destination with behavior-based obstacle avoidance (default)
    "astar",
    "theta_star",
    "dijkstra",
    "bfs",
    "greedy",
    "bi_astar",
    "msp",
]

# Built-in algorithms that use grid/waypoint planning behavior.
GRID_PATH_ALGORITHMS = [
    "astar",
    "theta_star",
    "dijkstra",
    "bfs",
    "greedy",
    "bi_astar",
    "msp",
]

# Algorithm display names
ALGORITHM_NAMES = {
    "direct": "Direct (Default)",
    "astar": "A* (A-Star)",
    "theta_star": "Theta* (Path Smoothing)",
    "dijkstra": "Dijkstra",
    "bfs": "Breadth-First Search",
    "greedy": "Greedy Best-First",
    "bi_astar": "Bidirectional A*",
    "msp": "Minimum Spanning Tree",
}


def get_available_path_algorithms() -> list[str]:
    """Return built-in algorithms plus registered custom algorithm names."""
    custom_names = [item["name"] for item in list_custom_path_algorithms()]
    extras = [name for name in custom_names if name not in PATH_ALGORITHMS]
    return [*PATH_ALGORITHMS, *extras]


def get_path_algorithm_labels() -> dict[str, str]:
    """Return label mapping for built-in + custom algorithms."""
    labels = dict(ALGORITHM_NAMES)
    for item in list_custom_path_algorithms():
        label = item.get("description") or item["name"].replace("_", " ")
        labels[item["name"]] = label
    return labels


def is_waypoint_path_algorithm(algorithm: str) -> bool:
    """Whether algorithm should follow precomputed waypoints."""
    if algorithm in GRID_PATH_ALGORITHMS:
        return True
    custom = get_custom_path_algorithm(algorithm)
    return bool(custom and custom.mode == "waypoint")


def register_path_algorithm(
    *,
    name: str,
    import_path: str,
    description: str = "",
    mode: str = "waypoint",
    replace: bool = False,
) -> dict[str, str]:
    """Register a custom path algorithm callable by import path."""
    return register_custom_path_algorithm(
        name=name,
        import_path=import_path,
        description=description,
        mode=mode,
        replace=replace,
        reserved_names=PATH_ALGORITHMS,
    )


def list_registered_path_algorithms() -> list[dict[str, str]]:
    """List all registered custom path algorithms."""
    return list_custom_path_algorithms()


def unregister_path_algorithm(name: str) -> Optional[dict[str, str]]:
    """Remove a registered custom path algorithm."""
    return unregister_custom_path_algorithm(name)


class PathPlanner:
    """
    Path planner with jamming zone avoidance.

    Uses PathPlanner3D for grid-based algorithms (A*, Dijkstra, etc.)
    or falls back to potential field method.
    """

    def __init__(
        self,
        algorithm: str = "astar",
        voxel_size: float = 2.0,
        bounds_min: list = None,
        bounds_max: list = None,
        safety_margin: float = 2.0,
    ):
        """
        Initialize path planner.

        Args:
            algorithm: Planning algorithm to use
            voxel_size: Grid cell size for pathfinding3d algorithms
            bounds_min: Minimum bounds [x, y, z] for grid
            bounds_max: Maximum bounds [x, y, z] for grid
            safety_margin: Extra margin around jamming zones
        """
        self.algorithm = algorithm
        self.voxel_size = voxel_size
        self.safety_margin = safety_margin

        # Default bounds
        self.bounds_min = bounds_min or [-50, -50, -10]
        self.bounds_max = bounds_max or [50, 50, 30]

        # PathPlanner3D instance (lazy initialization)
        self._planner3d: Optional[PathPlanner3D] = None
        self._planner_initialized = False

        # Cached paths
        self.paths: dict[str, list[np.ndarray]] = {}
        self.current_waypoints: dict[str, int] = {}

        # Track obstacles for replanning (None = never called, distinct from empty)
        self._current_obstacles: Optional[list] = None
        self._paths_stale: bool = False

    def _ensure_planner_initialized(self):
        """Initialize PathPlanner3D if needed and available."""
        if self._planner_initialized:
            return

        if PATHFINDING3D_AVAILABLE and self.algorithm in GRID_PATH_ALGORITHMS:
            try:
                # Use larger voxel size for slower algorithms
                voxel = self.voxel_size
                if self.algorithm in ["dijkstra", "bfs", "msp"]:
                    voxel = max(voxel, 5.0)  # Coarser grid for slow algorithms

                self._planner3d = PathPlanner3D(
                    bounds_min=self.bounds_min,
                    bounds_max=self.bounds_max,
                    voxel_size=voxel,
                    algorithm=self.algorithm,
                    diagonal_movement=True,
                )
                print(f"[PathPlanner] Initialized PathPlanner3D with {self.algorithm}")
            except Exception as e:
                print(f"[PathPlanner] Failed to init PathPlanner3D: {e}")
                self._planner3d = None

        self._planner_initialized = True

    def update_obstacles_from_jamming(self, jamming_zones: list[JammingZone]):
        """
        Update obstacles from jamming zones with type-based costs.

        Converts JammingZone objects to (x, y, z, radius, type) tuples for PathPlanner3D.

        Obstacle types determine path planning behavior:
        - PHYSICAL: Blocked (infinite cost, must path around)
        - HIGH_JAM: Very high cost (10x normal, strongly prefer to avoid)
        - LOW_JAM: Moderate cost (2x normal, soft preference to avoid)
        """
        obstacles = []
        for zone in jamming_zones:
            if zone.active:
                # Add safety margin to radius
                # Use jamming_radius for jamming types (extended influence area)
                if zone.obstacle_type == ObstacleType.PHYSICAL:
                    effective_radius = zone.radius + self.safety_margin
                else:
                    # For jamming types, use the jamming field radius
                    effective_radius = zone.jamming_radius + self.safety_margin

                obstacles.append(
                    (
                        zone.center[0],
                        zone.center[1],
                        zone.center[2] if len(zone.center) > 2 else 0,
                        effective_radius,
                        zone.obstacle_type.value,  # Include type for cost calculation
                    )
                )

        # Only update if obstacles changed
        if obstacles != self._current_obstacles:
            self._current_obstacles = obstacles

            if self._planner3d is not None:
                self._planner3d.update_obstacles(obstacles)
                # Mark paths as potentially stale but don't clear them
                # They will be replanned when an agent requests a new path
                # This allows visualization to continue showing paths
                self._paths_stale = True

    def plan_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        jamming_zones: list[JammingZone],
        agent_id: str = "default",
    ) -> Optional[list[np.ndarray]]:
        """
        Plan a path from start to goal avoiding jamming zones.

        Args:
            start: Start position [x, y, z]
            goal: Goal position [x, y, z]
            jamming_zones: List of jamming zones to avoid
            agent_id: Agent identifier for caching

        Returns:
            List of waypoints, or None if no path found
        """
        self._ensure_planner_initialized()

        start = np.array(start)
        goal = np.array(goal)

        # Update obstacles
        self.update_obstacles_from_jamming(jamming_zones)

        # Custom plugin algorithms are resolved and executed first.
        if has_custom_path_algorithm(self.algorithm):
            path = self._run_custom_algorithm(start, goal, jamming_zones, agent_id)
            if path:
                self.paths[agent_id] = path
                self.current_waypoints[agent_id] = 0
                return path

        # Use PathPlanner3D for grid-based algorithms
        if self._planner3d is not None and self.algorithm not in [
            "potential_field",
            "direct",
        ]:
            path_result = self._planner3d.find_path(start, goal)
            if path_result[0] is not None:
                path = path_result[0]
                algo_used = path_result[1]
                nodes = path_result[2]
                print(
                    f"[PathPlanner] {agent_id}: Path found with {algo_used}, {len(path)} waypoints, {nodes} nodes"
                )

                self.paths[agent_id] = path
                self.current_waypoints[agent_id] = 0
                return path
            else:
                print(
                    f"[PathPlanner] {agent_id}: No path found with {self.algorithm}, using fallback"
                )

        # Fallback to potential field or direct
        if self.algorithm == "direct":
            path = self._direct_path(start, goal)
        else:
            path = self._potential_field_path(start, goal, jamming_zones)

        if path:
            self.paths[agent_id] = path
            self.current_waypoints[agent_id] = 0

        return path

    def _run_custom_algorithm(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        jamming_zones: list[JammingZone],
        agent_id: str,
    ) -> Optional[list[np.ndarray]]:
        fn = get_custom_path_callable(self.algorithm)
        if fn is None:
            return None

        try:
            raw_path = fn(
                start=np.array(start, dtype=float).copy(),
                goal=np.array(goal, dtype=float).copy(),
                jamming_zones=list(jamming_zones),
                agent_id=agent_id,
                planner=self,
            )
        except TypeError:
            # Backward-compatible fallback for simpler call signatures.
            raw_path = fn(
                np.array(start, dtype=float).copy(),
                np.array(goal, dtype=float).copy(),
                list(jamming_zones),
            )
        except Exception as exc:
            print(
                f"[PathPlanner] {agent_id}: custom algorithm failed ({self.algorithm}): {exc}"
            )
            return None

        path = self._normalize_custom_path(raw_path, start, goal)
        if path is None:
            print(f"[PathPlanner] {agent_id}: custom algorithm returned no valid path")
            return None
        print(
            f"[PathPlanner] {agent_id}: custom algorithm {self.algorithm} generated {len(path)} waypoints"
        )
        return path

    def _normalize_custom_path(
        self,
        raw_path: Any,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Optional[list[np.ndarray]]:
        if raw_path is None:
            return None

        points: list[np.ndarray] = []
        try:
            for point in raw_path:
                arr = np.array(point, dtype=float).reshape(-1)
                if arr.size < 3:
                    continue
                points.append(arr[:3])
        except TypeError:
            return None

        if not points:
            return None

        start_arr = np.array(start, dtype=float)
        goal_arr = np.array(goal, dtype=float)
        if not np.allclose(points[0], start_arr):
            points.insert(0, start_arr.copy())
        if not np.allclose(points[-1], goal_arr):
            points.append(goal_arr.copy())

        if len(points) < 2:
            return None
        return self._smooth_path(points)

    def get_all_paths(self) -> dict[str, list[list[float]]]:
        """
        Get all planned paths for visualization.
        Only returns waypoints AHEAD of the vehicle's current position.

        Returns:
            Dict mapping agent_id to list of [x, y, z] waypoints (from current position forward)
        """
        result = {}
        for agent_id, path in self.paths.items():
            if path:
                # Get current waypoint index (tracks vehicle progress along path)
                start_idx = self.current_waypoints.get(agent_id, 0)
                # Only return waypoints from current position forward (ahead of vehicle)
                remaining_path = path[start_idx:]
                if remaining_path:
                    result[agent_id] = [
                        p.tolist() if hasattr(p, "tolist") else list(p)
                        for p in remaining_path
                    ]
        return result

    def get_next_waypoint(
        self,
        current_pos: np.ndarray,
        agent_id: str,
        threshold: float = 2.0,
    ) -> Optional[np.ndarray]:
        """
        Get next waypoint for agent, advancing if close enough.
        """
        if agent_id not in self.paths or not self.paths[agent_id]:
            return None

        path = self.paths[agent_id]
        idx = self.current_waypoints.get(agent_id, 0)

        if idx >= len(path):
            return None

        waypoint = np.array(path[idx])

        # Check if reached current waypoint
        if np.linalg.norm(np.array(current_pos) - waypoint) < threshold:
            self.current_waypoints[agent_id] = idx + 1
            if idx + 1 < len(path):
                return np.array(path[idx + 1])
            return None

        return waypoint

    def compute_velocity(
        self,
        current_pos: np.ndarray,
        goal: np.ndarray,
        jamming_zones: list[JammingZone],
        max_speed: float = 1.0,
    ) -> np.ndarray:
        """
        Compute velocity vector using potential field method.
        Used as fallback or for real-time obstacle avoidance.
        """
        pos = np.array(current_pos)
        goal = np.array(goal)

        # Attractive force toward goal
        to_goal = goal - pos
        dist_to_goal = np.linalg.norm(to_goal)

        if dist_to_goal < 0.1:
            return np.zeros(3)

        attractive = to_goal / dist_to_goal * min(1.0, dist_to_goal / 10.0)

        # Repulsive force from jamming zones
        repulsive = np.zeros(3)

        for zone in jamming_zones:
            if not zone.active:
                continue

            center = np.array(zone.center)
            to_agent = pos - center
            dist = np.linalg.norm(to_agent)

            influence_radius = zone.radius + self.safety_margin + 5.0

            if dist < influence_radius and dist > 0.1:
                strength = 2.0 * (1.0 - dist / influence_radius) ** 2
                direction = to_agent / dist
                repulsive += direction * strength

        # Combine forces
        velocity = attractive + repulsive

        # Limit speed
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity = velocity / speed * max_speed

        return velocity

    def _potential_field_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        jamming_zones: list[JammingZone],
        max_iterations: int = 500,
        step_size: float = 1.0,
    ) -> Optional[list[np.ndarray]]:
        """Generate path using potential field method."""
        path = [start.copy()]
        pos = start.copy()

        for _ in range(max_iterations):
            if np.linalg.norm(pos - goal) < step_size * 2:
                path.append(goal.copy())
                break

            vel = self.compute_velocity(pos, goal, jamming_zones, step_size)

            if np.linalg.norm(vel) < 0.01:
                vel = np.random.randn(3) * step_size * 0.5

            pos = pos + vel
            path.append(pos.copy())

        return self._smooth_path(path)

    def _direct_path(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        """Direct straight-line path."""
        return [start.copy(), goal.copy()]

    def _smooth_path(
        self, path: list[np.ndarray], tolerance: float = 0.1
    ) -> list[np.ndarray]:
        """Remove redundant waypoints on straight lines."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]

        for i in range(1, len(path) - 1):
            prev = np.array(smoothed[-1])
            curr = np.array(path[i])
            next_pt = np.array(path[i + 1])

            v1 = curr - prev
            v2 = next_pt - curr

            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)

            if len1 > 0 and len2 > 0:
                v1_norm = v1 / len1
                v2_norm = v2 / len2

                if np.dot(v1_norm, v2_norm) < 0.95:
                    smoothed.append(path[i])

        smoothed.append(path[-1])
        return smoothed

    def clear_path(self, agent_id: str):
        """Clear cached path for an agent."""
        if agent_id in self.paths:
            del self.paths[agent_id]
        if agent_id in self.current_waypoints:
            del self.current_waypoints[agent_id]

    def clear_all_paths(self):
        """Clear all cached paths."""
        self.paths.clear()
        self.current_waypoints.clear()

    def set_algorithm(self, algorithm: str):
        """Change path planning algorithm."""
        available = get_available_path_algorithms()
        if algorithm not in available:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")

        self.algorithm = algorithm

        # Reinitialize planner with new algorithm
        self._planner_initialized = False
        self._planner3d = None
        self.clear_all_paths()

        print(
            f"[PathPlanner] Algorithm changed to: {ALGORITHM_NAMES.get(algorithm, algorithm)}"
        )

    def set_bounds(self, bounds_min: list, bounds_max: list):
        """Update planning bounds."""
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max

        # Force reinitialization
        self._planner_initialized = False
        self._planner3d = None
