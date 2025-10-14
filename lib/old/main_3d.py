import matplotlib

matplotlib.use("Qt5Agg")
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import utils_3d as utils
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from path_planning_controller_3d import PathPlanningController3D
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FormationControl3DGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Formation Control Simulation")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)

        # Create main figure with mixed 3D and 2D subplots using GridSpec for custom sizing
        self.fig = plt.figure(figsize=(14, 10))

        # Use GridSpec to create custom layout: bigger top plots, smaller bottom plots
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(
            3, 2, figure=self.fig, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3
        )

        # Create 3D subplots for formation scene and trajectories (larger, top 2 rows)
        ax1 = self.fig.add_subplot(gs[0:2, 0], projection="3d")
        ax2 = self.fig.add_subplot(gs[0:2, 1], projection="3d")

        # Create 2D subplots for performance indicators (smaller, bottom row)
        ax3 = self.fig.add_subplot(gs[2, 0])
        ax4 = self.fig.add_subplot(gs[2, 1])

        # Store axes in array format for compatibility
        self.axs = np.array([[ax1, ax2], [ax3, ax4]], dtype=object)

        # Create canvas for all plots
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # Manually define obstacles (x, y, z, radius)
        # Add obstacles between start and destination to test avoidance
        self.obstacles = [
            (35, 75, 15, 10),
            (20, 50, 10, 15),
            (50, 100, 20, 12),
        ]

        # Initialize simulation parameters
        self.max_iter = 1000
        self.alpha = 10 ** (-5)
        self.delta = 2
        self.beta = self.alpha * (2**self.delta - 1)
        self.v = 3
        self.r0 = 5
        self.PT = 0.94

        # Initialize swarm positions in 3D space
        self.swarm_position = np.array(
            [
                [-5, 14, 0],
                [-5, -19, 5],
                [0, 0, -5],
                [35, -4, 0],
                [68, 0, 5],
                [72, 13, -5],
                [72, -18, 0],
            ],
            dtype=float,
        )
        self.swarm_destination = np.array([35, 150, 30], dtype=float)
        self.swarm_size = self.swarm_position.shape[0]
        self.swarm_control_ui = np.zeros((self.swarm_size, 3))  # 3D control

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
        self.swarm_paths = []

        # Colors
        self.node_colors = [
            [108 / 255, 155 / 255, 207 / 255],  # Light Blue
            [247 / 255, 147 / 255, 39 / 255],  # Orange
            [242 / 255, 102 / 255, 171 / 255],  # Light Pink
            [255 / 255, 217 / 255, 90 / 255],  # Light Gold
            [122 / 255, 168 / 255, 116 / 255],  # Green
            [147 / 255, 132 / 255, 209 / 255],  # Purple
            [245 / 255, 80 / 255, 80 / 255],  # Red
        ]
        self.line_colors = np.random.rand(
            self.swarm_position.shape[0], self.swarm_position.shape[0], 3
        )

        # Simulation control variables
        self.running = False
        self.paused = False
        self.iteration = 0
        self.Jn_converged = False

        # Control mode: 'behavior' or path planning algorithm
        self.control_mode = "behavior"
        self.path_planning_controller = None

        # Create swarm state mock for path planning
        self.swarm_state_wrapper = self._create_swarm_state_wrapper()

        # Create control buttons and mode selector
        self.create_plot_controls(main_layout)

        # Create timer for simulation
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)

        # Initial plot
        self.update_plot()

        # Auto-start the simulation
        self.running = True
        self.timer.start(50)  # 50ms interval

    def create_plot_controls(self, main_layout):
        # Create control widget and layout
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Algorithm selector row
        algo_widget = QWidget()
        algo_layout = QHBoxLayout(algo_widget)

        algo_label = QLabel("Control Mode:")
        algo_label.setStyleSheet("font-weight: bold; font-size: 12px;")

        self.algo_selector = QComboBox()
        self.algo_selector.addItem("Behavior-based (Obstacle Avoidance)", "behavior")
        self.algo_selector.addItem("A* (A-Star) Path Planning", "astar")
        self.algo_selector.addItem("Theta* (Path Smoothing)", "theta_star")
        self.algo_selector.addItem("Dijkstra Path Planning", "dijkstra")
        self.algo_selector.addItem("Breadth-First Search (BFS)", "bfs")
        self.algo_selector.addItem("Greedy Best-First Search", "greedy")
        self.algo_selector.addItem("Bidirectional A*", "bi_astar")
        self.algo_selector.addItem("Minimum Spanning Tree (MSP)", "msp")
        self.algo_selector.setStyleSheet("font-size: 11px; min-width: 250px;")
        self.algo_selector.currentIndexChanged.connect(self.on_algorithm_changed)

        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.algo_selector)
        algo_layout.addStretch()

        control_layout.addWidget(algo_widget)

        # Buttons row
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)

        # Create buttons with styling
        button_style = """
            QPushButton {
                min-width: 80px;
                min-height: 40px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
                border: 2px solid #333;
                color: #000000;
            }
        """

        self.pause_button = QPushButton("Pause")
        self.pause_button.setStyleSheet(
            button_style + "QPushButton { background-color: #fdf2ca; color: #8b4513; }"
        )
        self.pause_button.clicked.connect(self.pause_simulation)

        self.continue_button = QPushButton("Continue")
        self.continue_button.setStyleSheet(
            button_style + "QPushButton { background-color: #e3f0d8; color: #2d5016; }"
        )
        self.continue_button.clicked.connect(self.continue_simulation)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet(
            button_style + "QPushButton { background-color: #d8e3f0; color: #1e3a8a; }"
        )
        self.reset_button.clicked.connect(self.reset_simulation)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(
            button_style + "QPushButton { background-color: #f9aeae; color: #7f1d1d; }"
        )
        self.stop_button.clicked.connect(self.stop_simulation)

        # Add buttons to layout
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.continue_button)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.stop_button)

        control_layout.addWidget(buttons_widget)

        # Add control widget to main layout
        main_layout.addWidget(control_widget)

    def _create_swarm_state_wrapper(self):
        """Create a simple wrapper object for path planning controller."""

        class SwarmStateWrapper:
            def __init__(self, gui):
                self.gui = gui

            @property
            def swarm_size(self):
                return self.gui.swarm_size

            @property
            def swarm_position(self):
                return self.gui.swarm_position

            @property
            def swarm_destination(self):
                return self.gui.swarm_destination

            @property
            def obstacles(self):
                return self.gui.obstacles

        return SwarmStateWrapper(self)

    def on_algorithm_changed(self, index):
        """Handle control mode/algorithm change."""
        self.control_mode = self.algo_selector.itemData(index)

        if self.control_mode == "behavior":
            print("Switched to Behavior-based control")
            self.path_planning_controller = None
        else:
            # Initialize path planning controller with selected algorithm
            print(f"Switched to {self.control_mode.upper()} path planning")

            # Use larger voxel size for exhaustive algorithms to reduce computational cost
            if self.control_mode in ["dijkstra", "bfs", "msp"]:
                voxel_size = 10.0  # Coarser grid for slow algorithms
                print(
                    f"  Using coarse grid (voxel={voxel_size}) for {self.control_mode}"
                )
            else:
                voxel_size = 2.0  # Fine grid for fast algorithms

            self.path_planning_controller = PathPlanningController3D(
                self.swarm_state_wrapper,
                algorithm=self.control_mode,
                voxel_size=voxel_size,
                waypoint_threshold=3.0,
                detection_radius=5.0,  # Agents can detect obstacles within 5 units
            )

    def formation_control_step(self):
        # Reset control inputs at start of step
        self.swarm_control_ui = np.zeros((self.swarm_size, 3))  # 3D control

        # Formation control
        for i in range(self.swarm_size):
            for j in [x for x in range(self.swarm_size) if x != i]:
                rij = utils.calculate_distance(
                    self.swarm_position[i], self.swarm_position[j]
                )
                aij = utils.calculate_aij(self.alpha, self.delta, rij, self.r0, self.v)
                gij = utils.calculate_gij(rij, self.r0)

                if aij >= self.PT:
                    rho_ij = utils.calculate_rho_ij(self.beta, self.v, rij, self.r0)
                else:
                    rho_ij = 0

                qi = self.swarm_position[i, :]
                qj = self.swarm_position[j, :]
                eij = (qi - qj) / np.sqrt(rij)

                # Record matrices
                phi_rij = gij * aij
                self.communication_qualities_matrix[i, j] = phi_rij
                self.communication_qualities_matrix[j, i] = phi_rij
                self.distances_matrix[i, j] = rij
                self.distances_matrix[j, i] = rij
                self.neighbor_agent_matrix[i, j] = aij
                self.neighbor_agent_matrix[j, i] = aij

                # Formation control input
                # Before convergence: always maintain formation (weight = 1.0)
                # After convergence:
                #   - Path planning mode: disable formation (weight = 0.0) to allow free movement
                #   - Behavior mode: keep formation (weight = 1.0)
                if self.Jn_converged and self.control_mode != "behavior":
                    formation_weight = (
                        0.0  # Disable formation, let path planning take over
                    )
                else:
                    formation_weight = 1.0  # Maintain formation

                self.swarm_control_ui[i] += formation_weight * rho_ij * eij

            # Update position with formation control
            self.swarm_position[i] += self.swarm_control_ui[i]

        # Add destination-reaching control only after formation convergence
        if self.Jn_converged:
            # Choose control strategy based on mode
            if self.control_mode == "behavior":
                # Behavior-based obstacle avoidance
                for i in range(self.swarm_size):
                    has_obstacle_influence = False
                    # Check for obstacle collisions and apply avoidance
                    for obstacle in self.obstacles:
                        obstacle_pos = np.array([obstacle[0], obstacle[1], obstacle[2]])
                        obstacle_radius = obstacle[3]

                        # Calculate distance to obstacle center
                        dist_to_center = np.linalg.norm(
                            self.swarm_position[i] - obstacle_pos
                        )

                        # Buffer zones
                        buffer_zone = obstacle_radius + 8.0
                        wall_follow_zone = obstacle_radius + 4.0

                        if dist_to_center < buffer_zone:  # If within buffer zone
                            has_obstacle_influence = True
                            if dist_to_center < wall_follow_zone:
                                # Strong avoidance when very close
                                self.add_obstacle_avoidance_3d(
                                    i, obstacle_pos, obstacle_radius
                                )
                                # Minimal destination control when very close to obstacle
                                self.add_destination_control_3d(i, weight=0.2)
                            else:
                                # Wall following when in outer buffer zone
                                wall_normal = (
                                    self.swarm_position[i] - obstacle_pos
                                ) / dist_to_center
                                wall_pos = obstacle_pos + wall_normal * obstacle_radius
                                self.add_wall_following_3d(i, wall_pos, wall_normal)
                                # Reduced destination control during wall following
                                self.add_destination_control_3d(i, weight=0.3)

                    # If not influenced by any obstacle, apply normal destination control
                    if not has_obstacle_influence:
                        self.add_destination_control_3d(i, weight=1.0)

                    # Apply behavior control
                    self.swarm_position[i] += self.swarm_control_ui[i]
                    # Reset for next agent
                    self.swarm_control_ui[i] = np.zeros(3)
            else:
                # Path planning mode - compute and apply path planning control
                if self.path_planning_controller:
                    path_control = self.path_planning_controller.compute_control()
                    # Apply path planning control directly (formation already applied)
                    for i in range(self.swarm_size):
                        self.swarm_position[i] += path_control[i]

                    # Debug: Print control magnitudes occasionally
                    if self.iteration % 50 == 0 and np.any(path_control):
                        avg_control = np.mean(np.linalg.norm(path_control, axis=1))
                        print(
                            f"  [Debug] Average path control magnitude: {avg_control:.4f}"
                        )

        # Calculate performance indicators
        Jn_new = utils.calculate_Jn(
            self.communication_qualities_matrix, self.neighbor_agent_matrix, self.PT
        )
        rn_new = utils.calculate_rn(
            self.distances_matrix, self.neighbor_agent_matrix, self.PT
        )

        self.Jn.append(round(Jn_new, 4))
        self.rn.append(round(rn_new, 4))

        self.t_elapsed.append(time.time() - self.start_time)

    def add_destination_control_3d(self, agent_index, weight=1.0):
        """Add destination-reaching control input for an agent in 3D"""
        # Parameters for destination control
        am = 0.7  # Attraction magnitude
        bm = 1.0  # Distance threshold

        # Calculate vector to destination
        destination_vector = self.swarm_destination - self.swarm_position[agent_index]
        dist_to_dest = np.linalg.norm(destination_vector)

        if dist_to_dest > 0:  # Avoid division by zero
            destination_direction = destination_vector / dist_to_dest

            # Scale control input based on distance
            if dist_to_dest > bm:
                control_param = am
            else:
                control_param = am * (dist_to_dest / bm)

            # Apply weight to control input
            self.swarm_control_ui[agent_index] += (
                weight * destination_direction * control_param
            )

    def add_obstacle_avoidance_3d(
        self, agent_index, obstacle_position, obstacle_radius
    ):
        """Add obstacle avoidance control input for an agent in 3D"""
        # Avoidance parameters
        ao = 3.5  # Avoidance magnitude
        bo = 8.0  # Influence range

        # Calculate vector away from the obstacle
        obstacle_vector = self.swarm_position[agent_index] - obstacle_position
        dist_to_obstacle = np.linalg.norm(obstacle_vector)

        if dist_to_obstacle < (obstacle_radius + bo):
            avoidance_direction = obstacle_vector / dist_to_obstacle

            # Exponential scaling for aggressive close-range avoidance
            proximity_factor = np.exp(-0.3 * (dist_to_obstacle - obstacle_radius))
            control_param = (
                ao
                * proximity_factor
                * (1 + 1 / (dist_to_obstacle - obstacle_radius + 0.1))
            )

            # Add to existing control input
            self.swarm_control_ui[agent_index] += avoidance_direction * control_param

    def add_wall_following_3d(self, agent_index, wall_position, wall_normal):
        """Add wall-following control input for an agent in 3D"""
        # Wall following parameters
        af = 2.0  # Wall following force
        df = 12.0  # Desired distance from wall

        # Calculate perpendicular distance to wall
        agent_position = self.swarm_position[agent_index]
        distance_to_wall = np.dot(agent_position - wall_position, wall_normal)

        # Calculate two tangent directions perpendicular to wall_normal
        # Use a reference vector that's not parallel to wall_normal
        if abs(wall_normal[2]) < 0.9:
            reference = np.array([0, 0, 1])
        else:
            reference = np.array([1, 0, 0])

        # Create tangent direction using cross product
        tangent1 = np.cross(wall_normal, reference)
        tangent1 = tangent1 / np.linalg.norm(tangent1)

        # Get direction to destination
        to_dest = self.swarm_destination - agent_position
        to_dest_norm = np.linalg.norm(to_dest)
        if to_dest_norm > 0:
            to_dest = to_dest / to_dest_norm

        # Project destination direction onto tangent plane
        tangent_dest = to_dest - np.dot(to_dest, wall_normal) * wall_normal
        tangent_dest_norm = np.linalg.norm(tangent_dest)
        if tangent_dest_norm > 0:
            tangent_direction = tangent_dest / tangent_dest_norm
        else:
            tangent_direction = tangent1

        # Wall following behavior
        if abs(distance_to_wall) > df:
            # Correction when too close or too far from wall
            correction = -np.sign(distance_to_wall) * wall_normal
            control = af * (0.4 * tangent_direction + 0.6 * correction)
        else:
            # Wall following when at good distance
            control = 1.2 * af * tangent_direction

        self.swarm_control_ui[agent_index] += control

    def update_plot(self):
        # Determine which obstacles to show based on control mode
        if self.control_mode != "behavior" and self.path_planning_controller:
            # In path planning mode, only show discovered obstacles
            visible_obstacles = self.path_planning_controller.discovered_obstacles
            # Show undiscovered obstacles in gray/invisible
            undiscovered_obstacles = self.obstacles
        else:
            # In behavior mode, show all obstacles
            visible_obstacles = self.obstacles
            undiscovered_obstacles = []

        utils.plot_figures_3d(
            self.axs,
            self.t_elapsed,
            self.Jn,
            self.rn,
            self.swarm_position,
            self.PT,
            self.communication_qualities_matrix,
            self.swarm_size,
            self.swarm_paths,
            self.node_colors,
            self.line_colors,
            visible_obstacles,  # Known obstacles (red)
            self.swarm_destination,
            undiscovered_obstacles,  # Unknown obstacles (gray)
        )

        # Visualize planned paths if using path planning mode
        if self.control_mode != "behavior" and self.path_planning_controller:
            self.path_planning_controller.visualize_paths(self.axs[0, 0])
            self.path_planning_controller.visualize_paths(self.axs[0, 1])

        self.canvas.draw()

    def simulation_step(self):
        if self.running and not self.paused and self.iteration < self.max_iter:
            self.formation_control_step()
            self.update_plot()

            # Check convergence
            if len(self.Jn) > 19 and len(set(self.Jn[-20:])) == 1:
                if not self.Jn_converged:
                    print(
                        f"Formation completed: Jn values has converged in {round(self.t_elapsed[-1], 2)} seconds {self.iteration - 20} iterations."
                    )
                    print("Now moving toward destination...")
                    self.Jn_converged = True
                    # Don't stop - continue to destination phase

            # Check if swarm center is close to destination
            swarm_center = np.mean(self.swarm_position, axis=0)
            dist_to_dest = np.linalg.norm(swarm_center - self.swarm_destination)

            if dist_to_dest < 0.05:  # Threshold of 0.05 units
                print(
                    f"Swarm has reached the destination in {round(self.t_elapsed[-1], 2)} seconds {self.iteration} iterations!"
                )
                self.running = False
                self.timer.stop()
            else:
                self.iteration += 1

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.timer.start(50)

    def pause_simulation(self):
        self.paused = True
        self.running = False
        self.timer.stop()
        print("Simulation paused.")

    def continue_simulation(self):
        if not self.running:
            self.running = True
            self.paused = False
            if self.Jn_converged:
                print("Simulation resumed.\nSwarm start reaching to the destination...")
            else:
                print("Simulation resumed.")
            self.timer.start(50)

    def reset_simulation(self):
        # Reset all simulation parameters
        self.running = False
        self.paused = False
        self.iteration = 0
        self.Jn_converged = False
        self.timer.stop()

        # Reset performance indicators
        self.Jn = []
        self.rn = []
        self.t_elapsed = []
        self.start_time = time.time()

        # Reset swarm positions to initial state
        self.swarm_position = np.array(
            [
                [-5, 14, 0],
                [-5, -19, 5],
                [0, 0, -5],
                [35, -4, 0],
                [68, 0, 5],
                [72, 13, -5],
                [72, -18, 0],
            ],
            dtype=float,
        )
        self.swarm_control_ui = np.zeros((self.swarm_size, 3))

        # Reset matrices
        self.communication_qualities_matrix = np.zeros(
            (self.swarm_size, self.swarm_size)
        )
        self.distances_matrix = np.zeros((self.swarm_size, self.swarm_size))
        self.neighbor_agent_matrix = np.zeros((self.swarm_size, self.swarm_size))
        self.swarm_paths = []

        # Update the plot
        self.update_plot()
        print("Simulation reset.")

    def stop_simulation(self):
        self.running = False
        self.timer.stop()
        print("Stopping application...")
        QApplication.quit()


# Create and run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FormationControl3DGUI()
    window.show()
    print("\n=== 3D Formation Control Simulation ===")
    print("Instructions:")
    print("1. Obstacles are pre-defined in the code (edit self.obstacles in __init__)")
    print("2. You can rotate the 3D plots by clicking and dragging")
    print("3. Use control buttons to Pause/Continue/Reset simulation")
    print("4. Watch the swarm avoid obstacles while maintaining formation\n")
    print(f"Current obstacles: {len(window.obstacles)} defined")
    for i, obs in enumerate(window.obstacles):
        print(
            f"  Obstacle {i + 1}: center=({obs[0]:.1f}, {obs[1]:.1f}, {obs[2]:.1f}), radius={obs[3]:.1f}"
        )
    print()
    sys.exit(app.exec_())
