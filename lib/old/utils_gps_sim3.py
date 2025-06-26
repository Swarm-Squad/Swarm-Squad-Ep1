"""
Utilities for GPS-Denied Multi-Vehicle RTK Simulation with Line-of-Sight Obstacles
Contains BaseStation, Rover, Transceiver, EKF, A* pathfinding, and line-of-sight blocking
"""

import heapq
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

# Satellite constellation for GPS simulation
SATELLITE_POSITIONS = []


def generate_satellite_constellation(n_satellites=12, orbit_radius=200):
    """Generate a constellation of GPS satellites"""
    global SATELLITE_POSITIONS
    SATELLITE_POSITIONS = []
    
    for i in range(n_satellites):
        angle = 2 * math.pi * i / n_satellites
        x = 50 + orbit_radius * math.cos(angle)  # Center around map
        y = 50 + orbit_radius * math.sin(angle)
        z = 150 + random.uniform(-20, 20)  # Height variation
        SATELLITE_POSITIONS.append([x, y, z])
    
    return SATELLITE_POSITIONS


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def simulate_distance_measurement(a_pos: Tuple[float, float], b_pos: Tuple[float, float], noise_std: float = 0.1) -> float:
    """Simulate distance measurement with noise"""
    true_dist = euclidean_distance(a_pos, b_pos)
    return max(0.1, true_dist + random.gauss(0, noise_std))


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Calculate the shortest distance from a point to a line segment"""
    # Vector from line_start to line_end
    line_vec = line_end - line_start
    # Vector from line_start to point
    point_vec = point - line_start
    
    # Handle degenerate case where line_start == line_end
    line_length_sq = np.dot(line_vec, line_vec)
    if line_length_sq < 1e-10:
        return np.linalg.norm(point_vec)
    
    # Project point onto line
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_length_sq))
    projection = line_start + t * line_vec
    
    return np.linalg.norm(point - projection)


def line_circle_intersection(line_start: np.ndarray, line_end: np.ndarray, 
                           circle_center: np.ndarray, circle_radius: float) -> bool:
    """Check if a line segment intersects with a circle (obstacle)"""
    # Calculate distance from circle center to line segment
    distance_to_line = point_to_line_distance(circle_center, line_start, line_end)
    
    # If distance is less than radius, there's an intersection
    return distance_to_line <= circle_radius


def has_line_of_sight(start_pos: np.ndarray, end_pos: np.ndarray, 
                     obstacles: List[Tuple[float, float, float]]) -> bool:
    """Check if there's a clear line of sight between two points, considering obstacles"""
    for obstacle_x, obstacle_y, obstacle_radius in obstacles:
        obstacle_center = np.array([obstacle_x, obstacle_y])
        
        # Check if line intersects with this obstacle
        if line_circle_intersection(start_pos, end_pos, obstacle_center, obstacle_radius):
            return False
    
    return True


class Transceiver:
    """Communication transceiver with range, packet loss, and line-of-sight simulation"""
    
    def __init__(self, comm_range: float, packet_loss: float = 0.1, obstacles: List[Tuple[float, float, float]] = None):
        self.comm_range = comm_range
        self.packet_loss = packet_loss
        self.obstacles = obstacles or []
    
    def update_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """Update the list of obstacles for line-of-sight calculations"""
        self.obstacles = obstacles
    
    def in_range(self, other_pos: np.ndarray, my_pos: np.ndarray) -> bool:
        """Check if other position is within communication range"""
        return np.linalg.norm(other_pos - my_pos) <= self.comm_range
    
    def has_line_of_sight(self, other_pos: np.ndarray, my_pos: np.ndarray) -> bool:
        """Check if there's line of sight to other position (not blocked by obstacles)"""
        return has_line_of_sight(my_pos, other_pos, self.obstacles)
    
    def can_communicate(self, other_pos: np.ndarray, my_pos: np.ndarray) -> bool:
        """Check if communication is possible (in range AND has line of sight)"""
        return self.in_range(other_pos, my_pos) and self.has_line_of_sight(other_pos, my_pos)
    
    def send(self, message: Any, target: 'Vehicle', sender_pos: np.ndarray) -> bool:
        """Send message to target with packet loss and line-of-sight simulation"""
        if random.random() < self.packet_loss:
            return False  # Packet lost
        
        if self.can_communicate(target.true_pos, sender_pos):
            target.receive(message)
            return True
        return False


class BaseStation(QObject):
    """RTK base station that provides correction data with line-of-sight considerations"""
    
    correctionsReady = pyqtSignal(dict, object)  # corrections, source
    
    def __init__(self, true_pos: Tuple[float, float], comm_range: float = 30, 
                 noise_std: float = 0.5, packet_loss: float = 0.1,
                 obstacles: List[Tuple[float, float, float]] = None):
        super().__init__()
        self.true_pos = np.array(true_pos, dtype=float)
        self.comm_range = comm_range
        self.noise_std = noise_std
        self.packet_loss = packet_loss
        self.obstacles = obstacles or []
        self.transceiver = Transceiver(comm_range, packet_loss, obstacles)
    
    def update_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """Update obstacles for line-of-sight calculations"""
        self.obstacles = obstacles
        self.transceiver.update_obstacles(obstacles)
    
    def measure_and_broadcast(self, satellite_positions: List[List[float]], rovers: List['Rover']):
        """Measure pseudoranges and broadcast RTK corrections with line-of-sight checking"""
        corrections = {}
        
        # Compute corrections for each satellite
        for i, sat_pos in enumerate(satellite_positions):
            # True range
            sat_3d = np.array([sat_pos[0], sat_pos[1], sat_pos[2]])
            base_3d = np.array([self.true_pos[0], self.true_pos[1], 0])
            true_range = np.linalg.norm(sat_3d - base_3d)
            
            # Measured range with noise
            measured_range = true_range + random.gauss(0, self.noise_std)
            
            # Correction is the difference
            corrections[i] = true_range - measured_range
        
        # Broadcast to rovers with line-of-sight checking
        for rover in rovers:
            if self.transceiver.can_communicate(rover.true_pos, self.true_pos):
                if random.random() > self.packet_loss:  # Packet not lost
                    self.correctionsReady.emit(corrections, self)


class ExtendedKalmanFilter:
    """Extended Kalman Filter for rover state estimation"""
    
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray):
        self.state = initial_state.copy()  # [x, y, vx, vy, bias_x, bias_y]
        self.covariance = initial_covariance.copy()
        
        # Process noise
        self.Q = np.diag([0.1, 0.1, 0.05, 0.05, 0.01, 0.01])  # Process noise covariance
        
        # Measurement noise for GPS
        self.R_gps = np.diag([1.0, 1.0])  # GPS measurement noise
        
        # Measurement noise for RTK
        self.R_rtk = np.diag([0.1, 0.1])  # RTK measurement noise (much lower)
    
    def predict(self, dt: float, control_input: np.ndarray = None):
        """Prediction step with IMU and control input"""
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Control input matrix (if acceleration is applied)
        if control_input is not None:
            B = np.array([
                [0.5 * dt**2, 0],
                [0, 0.5 * dt**2],
                [dt, 0],
                [0, dt],
                [0, 0],
                [0, 0]
            ])
            self.state = F @ self.state + B @ control_input
        else:
            self.state = F @ self.state
        
        # Add IMU bias drift
        self.state[4] += random.gauss(0, 0.01)  # x bias drift
        self.state[5] += random.gauss(0, 0.01)  # y bias drift
        
        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q
    
    def update_gps(self, measurement: np.ndarray):
        """Update with GPS measurement"""
        self._update_measurement(measurement, self.R_gps, is_rtk=False)
    
    def update_rtk(self, measurement: np.ndarray):
        """Update with RTK-corrected measurement"""
        self._update_measurement(measurement, self.R_rtk, is_rtk=True)
    
    def _update_measurement(self, measurement: np.ndarray, R: np.ndarray, is_rtk: bool = False):
        """Generic measurement update"""
        # Measurement matrix (observe position only)
        H = np.array([
            [1, 0, 0, 0, 1, 0],  # x position + x bias
            [0, 1, 0, 0, 0, 1]   # y position + y bias
        ])
        
        # Innovation
        predicted_measurement = H @ self.state
        innovation = measurement - predicted_measurement
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ innovation
        Identity = np.eye(self.covariance.shape[0])
        self.covariance = (Identity - K @ H) @ self.covariance


class PurePursuitController:
    """Pure pursuit controller for waypoint following"""
    
    def __init__(self, lookahead_distance: float = 3.0, max_speed: float = 2.0):
        self.lookahead_distance = lookahead_distance
        self.max_speed = max_speed
    
    def compute_control(self, current_pos: np.ndarray, current_heading: float, 
                       target_waypoint: np.ndarray) -> Tuple[float, float]:
        """Compute control commands (speed, steering_angle) for pure pursuit"""
        # Vector to target
        to_target = target_waypoint - current_pos
        distance_to_target = np.linalg.norm(to_target)
        
        if distance_to_target < 0.5:  # Close enough to waypoint
            return 0.0, 0.0
        
        # Desired heading
        desired_heading = math.atan2(to_target[1], to_target[0])
        
        # Heading error
        heading_error = desired_heading - current_heading
        
        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Speed control (slow down near target)
        speed = min(self.max_speed, distance_to_target)
        
        # Steering angle (proportional to heading error)
        steering_angle = heading_error * 0.5  # Gain factor
        
        return speed, steering_angle


class Vehicle:
    """Base vehicle class"""
    
    def __init__(self, agent_id: str, initial_position: Tuple[float, float], color: str = "blue"):
        self.id = agent_id
        self.true_pos = np.array(initial_position, dtype=float)
        self.color = color
        self.heading = 0.0
        self.velocity = np.array([0.0, 0.0])
        self.trajectory = [self.true_pos.copy()]
    
    def move(self, dt: float, control: Tuple[float, float]):
        """Update vehicle position based on control input"""
        speed, steering_angle = control
        
        # Update heading
        self.heading += steering_angle * dt
        
        # Update velocity
        self.velocity[0] = speed * math.cos(self.heading)
        self.velocity[1] = speed * math.sin(self.heading)
        
        # Update position
        self.true_pos += self.velocity * dt
        
        # Store trajectory
        self.trajectory.append(self.true_pos.copy())
        if len(self.trajectory) > 500:  # Limit memory
            self.trajectory.pop(0)
    
    def receive(self, message: Any):
        """Receive a message (to be overridden)"""
        pass


class Rover(Vehicle):
    """Rover with GPS, RTK, and navigation capabilities with line-of-sight awareness"""
    
    def __init__(self, agent_id: str, initial_position: Tuple[float, float], 
                 waypoints: List[Tuple[float, float]], color: str = "blue",
                 comm_range: float = 20, packet_loss: float = 0.05,
                 obstacles: List[Tuple[float, float, float]] = None):
        super().__init__(agent_id, initial_position, color)
        
        # Navigation
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self.original_waypoints = self.waypoints.copy()
        self.current_waypoint_index = 0
        self.waypoint_threshold = 2.0
        self.is_replanned = False
        
        # Communication with line-of-sight support
        self.obstacles = obstacles or []
        self.transceiver = Transceiver(comm_range, packet_loss, obstacles)
        self.queued_corrections = None
        self.last_correction_time = 0.0
        
        # State estimation
        initial_state = np.array([
            initial_position[0], initial_position[1],  # position
            0.0, 0.0,  # velocity
            0.0, 0.0   # IMU biases
        ])
        initial_covariance = np.diag([1.0, 1.0, 0.5, 0.5, 0.1, 0.1])
        self.ekf = ExtendedKalmanFilter(initial_state, initial_covariance)
        self.estimate = self.true_pos.copy()
        self.covariance = initial_covariance[:2, :2]  # Position covariance only
        
        # Control
        self.controller = PurePursuitController()
        
        # GPS status
        self.has_gps = True
    
    def update_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """Update obstacles for line-of-sight calculations"""
        self.obstacles = obstacles
        self.transceiver.update_obstacles(obstacles)
    
    def predict_with_imu(self, dt: float):
        """IMU-based prediction step"""
        # Add IMU noise
        imu_noise = np.array([random.gauss(0, 0.1), random.gauss(0, 0.1)])
        control_input = imu_noise  # Simulated IMU acceleration
        
        self.ekf.predict(dt, control_input)
        
        # Update estimate from EKF
        self.estimate = self.ekf.state[:2].copy()
        self.covariance = self.ekf.covariance[:2, :2].copy()
    
    def update_with_pseudoranges_and_corrections(self, satellite_positions: List[List[float]]):
        """Update with pseudorange measurements and RTK corrections"""
        if not satellite_positions:
            return
        
        # Simulate pseudorange measurements
        measurements = []
        for sat_pos in satellite_positions[:6]:  # Use first 6 satellites
            sat_2d = np.array([sat_pos[0], sat_pos[1]])
            range_measurement = np.linalg.norm(self.true_pos - sat_2d)
            range_measurement += random.gauss(0, 2.0)  # GPS noise
            measurements.append(range_measurement)
        
        # If we have RTK corrections, apply them
        if self.queued_corrections:
            # Simulate RTK-corrected position measurement
            rtk_position = self.true_pos + np.array([random.gauss(0, 0.1), random.gauss(0, 0.1)])
            self.ekf.update_rtk(rtk_position)
            self.last_correction_time = 0.0  # Reset timer
            self.queued_corrections = None
        elif self.has_gps:
            # Regular GPS update
            gps_position = self.true_pos + np.array([random.gauss(0, 1.0), random.gauss(0, 1.0)])
            self.ekf.update_gps(gps_position)
        
        # Update estimate
        self.estimate = self.ekf.state[:2].copy()
        self.covariance = self.ekf.covariance[:2, :2].copy()
    
    def compute_control_to_next_waypoint(self) -> Tuple[float, float]:
        """Compute control commands to reach next waypoint"""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return 0.0, 0.0
        
        current_waypoint = self.waypoints[self.current_waypoint_index]
        
        # Check if we've reached the current waypoint
        distance_to_waypoint = np.linalg.norm(self.estimate - current_waypoint)
        if distance_to_waypoint < self.waypoint_threshold:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                # If this was a replanned path, restore original waypoints
                if self.is_replanned:
                    self.waypoints = self.original_waypoints.copy()
                    self.current_waypoint_index = 0
                    self.is_replanned = False
                else:
                    return 0.0, 0.0  # Mission complete
        
        if self.current_waypoint_index < len(self.waypoints):
            target = self.waypoints[self.current_waypoint_index]
            return self.controller.compute_control(self.estimate, self.heading, target)
        
        return 0.0, 0.0
    
    def has_recent_corrections(self, threshold_steps: int) -> bool:
        """Check if rover has received recent RTK corrections"""
        return self.last_correction_time < threshold_steps
    
    def set_waypoints(self, new_waypoints: List[np.ndarray], is_replan: bool = False):
        """Set new waypoints for the rover"""
        self.waypoints = new_waypoints.copy()
        self.current_waypoint_index = 0
        self.is_replanned = is_replan
    
    def on_corrections_ready(self, corrections: Dict[int, float], source: BaseStation):
        """Slot to receive RTK corrections from base stations"""
        self.queued_corrections = corrections
        self.last_correction_time = 0.0
    
    def receive(self, message: Any):
        """Receive communication message"""
        if isinstance(message, tuple) and len(message) == 2:
            tag, payload = message
            if tag == "corrections":
                self.queued_corrections = payload
                self.last_correction_time = 0.0
    
    def reset(self, initial_position: Tuple[float, float], waypoints: List[Tuple[float, float]]):
        """Reset rover to initial state"""
        self.true_pos = np.array(initial_position, dtype=float)
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self.original_waypoints = self.waypoints.copy()
        self.current_waypoint_index = 0
        self.is_replanned = False
        self.trajectory = [self.true_pos.copy()]
        self.heading = 0.0
        self.velocity = np.array([0.0, 0.0])
        self.has_gps = True
        self.queued_corrections = None
        self.last_correction_time = 0.0
        
        # Reset EKF
        initial_state = np.array([
            initial_position[0], initial_position[1],
            0.0, 0.0, 0.0, 0.0
        ])
        initial_covariance = np.diag([1.0, 1.0, 0.5, 0.5, 0.1, 0.1])
        self.ekf = ExtendedKalmanFilter(initial_state, initial_covariance)
        self.estimate = self.true_pos.copy()
        self.covariance = initial_covariance[:2, :2]


@dataclass
class GridNode:
    """Node for A* pathfinding"""
    x: int
    y: int
    g_cost: float = float('inf')  # Cost from start
    h_cost: float = 0.0           # Heuristic cost to goal
    f_cost: float = float('inf')  # Total cost
    parent: Optional['GridNode'] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class Grid:
    """Grid for A* pathfinding with obstacles"""
    
    def __init__(self, width: int, height: int, cell_size: float = 1.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.obstacles = set()  # Set of (x, y) grid coordinates that are obstacles
    
    def add_circular_obstacle(self, center: Tuple[float, float], radius: float):
        """Add a circular obstacle to the grid"""
        center_grid = (int(center[0] / self.cell_size), int(center[1] / self.cell_size))
        radius_grid = int(radius / self.cell_size) + 1
        
        for dx in range(-radius_grid, radius_grid + 1):
            for dy in range(-radius_grid, radius_grid + 1):
                if dx*dx + dy*dy <= radius_grid*radius_grid:
                    x, y = center_grid[0] + dx, center_grid[1] + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        self.obstacles.add((x, y))
    
    def is_obstacle(self, x: int, y: int) -> bool:
        """Check if grid cell is an obstacle"""
        return (x, y) in self.obstacles
    
    def is_valid(self, x: int, y: int) -> bool:
        """Check if grid cell is valid (within bounds and not obstacle)"""
        return (0 <= x < self.width and 0 <= y < self.height and 
                not self.is_obstacle(x, y))
    
    def get_neighbors(self, node: GridNode) -> List[GridNode]:
        """Get valid neighboring nodes"""
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dx, dy in directions:
            x, y = node.x + dx, node.y + dy
            if self.is_valid(x, y):
                neighbors.append(GridNode(x, y))
        
        return neighbors
    
    def clear_obstacles(self):
        """Clear all obstacles from the grid"""
        self.obstacles.clear()


def heuristic(node1: GridNode, node2: GridNode) -> float:
    """Heuristic function for A* (Euclidean distance)"""
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


def astar_search(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* pathfinding algorithm"""
    if not grid.is_valid(start[0], start[1]) or not grid.is_valid(goal[0], goal[1]):
        return None
    
    start_node = GridNode(start[0], start[1], g_cost=0.0)
    goal_node = GridNode(goal[0], goal[1])
    
    start_node.h_cost = heuristic(start_node, goal_node)
    start_node.f_cost = start_node.g_cost + start_node.h_cost
    
    open_set = [start_node]
    closed_set = set()
    
    # Keep track of nodes for efficient lookup
    all_nodes = {(start[0], start[1]): start_node}
    
    while open_set:
        current = heapq.heappop(open_set)
        
        if (current.x, current.y) in closed_set:
            continue
        
        closed_set.add((current.x, current.y))
        
        # Check if we reached the goal
        if current.x == goal[0] and current.y == goal[1]:
            # Reconstruct path
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]  # Reverse to get path from start to goal
        
        # Examine neighbors
        for neighbor in grid.get_neighbors(current):
            if (neighbor.x, neighbor.y) in closed_set:
                continue
            
            # Calculate movement cost (diagonal moves cost more)
            dx, dy = abs(neighbor.x - current.x), abs(neighbor.y - current.y)
            movement_cost = 1.414 if dx == 1 and dy == 1 else 1.0  # sqrt(2) for diagonal
            
            tentative_g_cost = current.g_cost + movement_cost
            
            # Check if we found a better path to this neighbor
            node_key = (neighbor.x, neighbor.y)
            if node_key not in all_nodes:
                all_nodes[node_key] = neighbor
                neighbor.g_cost = tentative_g_cost
                neighbor.h_cost = heuristic(neighbor, goal_node)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                neighbor.parent = current
                heapq.heappush(open_set, neighbor)
            elif tentative_g_cost < all_nodes[node_key].g_cost:
                existing_node = all_nodes[node_key]
                existing_node.g_cost = tentative_g_cost
                existing_node.f_cost = existing_node.g_cost + existing_node.h_cost
                existing_node.parent = current
                heapq.heappush(open_set, existing_node)
    
    return None  # No path found


# Additional utility functions from the original utils_gps_sim.py

def trilaterate_2d(anchor1: Tuple[float, float], dist1: float,
                   anchor2: Tuple[float, float], dist2: float,
                   anchor3: Tuple[float, float], dist3: float) -> Tuple[float, float]:
    """Trilateration in 2D using three known positions and their distances to unknown point"""
    x1, y1 = anchor1
    x2, y2 = anchor2
    x3, y3 = anchor3

    # System of equations for trilateration
    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = dist1**2 - dist2**2 - x1**2 + x2**2 - y1**2 + y2**2
    D = 2 * (x3 - x2)
    E = 2 * (y3 - y2)
    F = dist2**2 - dist3**2 - x2**2 + x3**2 - y2**2 + y3**2

    denominator = A * E - B * D
    if abs(denominator) < 1e-10:
        raise ValueError("Trilateration failed: degenerate geometry (anchors are collinear)")

    x = (C * E - B * F) / denominator
    y = (A * F - C * D) / denominator

    return (x, y)


def multilaterate_2d(anchors: List[Tuple[float, float]], distances: List[float]) -> Tuple[float, float]:
    """Enhanced trilateration using multiple anchors (more than 3) with least squares"""
    if len(anchors) < 3:
        raise ValueError("Need at least 3 anchors for trilateration")

    if len(anchors) == 3:
        return trilaterate_2d(anchors[0], distances[0], anchors[1], distances[1], anchors[2], distances[2])

    # Use least squares for overdetermined system (more than 3 anchors)
    n = len(anchors)
    A = np.zeros((n - 1, 2))
    b = np.zeros(n - 1)

    x0, y0 = anchors[0]
    r0 = distances[0]

    for i in range(1, n):
        xi, yi = anchors[i]
        ri = distances[i]

        A[i - 1, 0] = 2 * (xi - x0)
        A[i - 1, 1] = 2 * (yi - y0)
        b[i - 1] = ri**2 - r0**2 - xi**2 + x0**2 - yi**2 + y0**2

    try:
        # Solve using least squares
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        return (float(solution[0]), float(solution[1]))
    except np.linalg.LinAlgError:
        # Fallback to simple trilateration with first 3 anchors
        return trilaterate_2d(anchors[0], distances[0], anchors[1], distances[1], anchors[2], distances[2])


def calculate_position_error(true_pos: Tuple[float, float], estimated_pos: Tuple[float, float]) -> float:
    """Calculate the error between true and estimated positions"""
    return euclidean_distance(true_pos, estimated_pos)


def bound_position(pos: Tuple[float, float], bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Keep position within specified bounds (min_x, max_x, min_y, max_y)"""
    min_x, max_x, min_y, max_y = bounds
    x = max(min_x, min(max_x, pos[0]))
    y = max(min_y, min(max_y, pos[1]))
    return (x, y) 