import matplotlib

matplotlib.use("Qt5Agg")
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import utils_old_qt5 as utils
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FormationControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Formation Control Simulation")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QVBoxLayout(central_widget)

        # Create main figure with subplots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 10))

        # Create canvas for all plots
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # Connect mouse events to the canvas
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Initialize obstacle list
        self.obstacles = []  # Will store (x, y, radius) for each obstacle

        # Initialize simulation parameters (keeping your original parameters)
        self.max_iter = 500
        self.alpha = 10 ** (-5)
        self.delta = 2
        self.beta = self.alpha * (2**self.delta - 1)
        self.v = 3
        self.r0 = 5
        self.PT = 0.94

        # Initialize swarm positions and other parameters (your original initialization)
        self.swarm_position = np.array(
            [[-5, 14], [-5, -19], [0, 0], [35, -4], [68, 0], [72, 13], [72, -18]],
            dtype=float,
        )
        self.swarm_destination = np.array([35, 150], dtype=float)
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
        self.swarm_paths = []

        # Colors (your original color settings)
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

        # Add drawing state variables
        self.drawing_obstacle = False
        self.obstacle_start = None
        self.temp_circle = None  # Store temporary circle while drawing

        # Create control buttons
        self.create_plot_controls(main_layout)

        # Create timer for simulation
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)

        # Auto-start the simulation
        self.running = True
        self.timer.start(50)  # 50ms interval

    def create_plot_controls(self, main_layout):
        # Create control widget and layout
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)

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

        self.undo_button = QPushButton("Undo")
        self.undo_button.setStyleSheet(
            button_style + "QPushButton { background-color: #e6e6e6; color: #374151; }"
        )
        self.undo_button.clicked.connect(self.undo_last_obstacle)

        # Add buttons to layout
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.continue_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.undo_button)

        # Add control widget to main layout
        main_layout.addWidget(control_widget)

    def formation_control_step(self):
        # Reset control inputs at start of step
        self.swarm_control_ui = np.zeros((self.swarm_size, 2))

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
                self.swarm_control_ui[i] += rho_ij * eij

            # Add destination-reaching control only after formation convergence
            if self.Jn_converged:
                has_obstacle_influence = False
                # First check for obstacle collisions and apply avoidance
                for obstacle in self.obstacles:
                    obstacle_pos = np.array([obstacle[0], obstacle[1]])
                    obstacle_radius = obstacle[2]

                    # Calculate distance to obstacle center
                    dist_to_center = np.linalg.norm(
                        self.swarm_position[i] - obstacle_pos
                    )

                    # Increase buffer zones
                    buffer_zone = obstacle_radius + 6.0
                    wall_follow_zone = obstacle_radius + 3.0

                    if dist_to_center < buffer_zone:  # If within buffer zone
                        has_obstacle_influence = True
                        if dist_to_center < wall_follow_zone:
                            # Much stronger avoidance when very close
                            self.add_obstacle_avoidance(
                                i, obstacle_pos, obstacle_radius
                            )
                            # Minimal destination control when very close to obstacle
                            self.add_destination_control(i, weight=0.1)
                        else:
                            # Enhanced wall following when in outer buffer zone
                            wall_normal = (
                                self.swarm_position[i] - obstacle_pos
                            ) / dist_to_center
                            wall_pos = obstacle_pos + wall_normal * obstacle_radius
                            self.add_wall_following(i, wall_pos, wall_normal)
                            # Reduced destination control during wall following
                            self.add_destination_control(i, weight=0.2)

                # If not influenced by any obstacle, apply normal destination control
                if not has_obstacle_influence:
                    self.add_destination_control(i, weight=1.0)

            # Update position
            self.swarm_position[i] += self.swarm_control_ui[i]

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

    def add_destination_control(self, agent_index, weight=1.0):
        """Add destination-reaching control input for an agent"""
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

    def add_obstacle_avoidance(self, agent_index, obstacle_position, obstacle_radius):
        """Add obstacle avoidance control input for an agent"""
        # Much stronger avoidance parameters
        ao = 3.0  # Significantly increased avoidance magnitude
        bo = 6.0  # Even larger influence range

        # Calculate vector away from the obstacle
        obstacle_vector = self.swarm_position[agent_index] - obstacle_position
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
            self.swarm_control_ui[agent_index] += avoidance_direction * control_param

    def add_wall_following(self, agent_index, wall_position, wall_normal):
        """Add wall-following control input for an agent"""
        # Stronger wall following parameters
        af = 2.0  # Much stronger wall following force
        df = 10.0  # Larger desired distance from wall

        # Calculate perpendicular distance to wall
        agent_position = self.swarm_position[agent_index]
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

        self.swarm_control_ui[agent_index] += control

    def update_plot(self):
        utils.plot_figures_task1(
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
            self.obstacles,
            self.swarm_destination,
        )
        self.canvas.draw()

    def simulation_step(self):
        if self.running and not self.paused and self.iteration < self.max_iter:
            self.formation_control_step()
            self.update_plot()

            # Check convergence but don't pause
            if len(self.Jn) > 19 and len(set(self.Jn[-20:])) == 1:
                if not self.Jn_converged:
                    print(
                        f"Formation completed: Jn values has converged in {round(self.t_elapsed[-1], 2)} seconds {self.iteration - 20} iterations.\nSimulation paused."
                    )
                    self.Jn_converged = True
                    self.running = False
                    self.timer.stop()

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

    def continue_simulation(self):
        if not self.running:
            self.running = True
            self.paused = False
            if self.Jn_converged:
                print("Simulation resumed.\nSwarm start reaching to the destination...")
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
            [[-5, 14], [-5, -19], [0, 0], [35, -4], [68, 0], [72, 13], [72, -18]],
            dtype=float,
        )
        self.swarm_control_ui = np.zeros((self.swarm_size, 2))

        # Reset matrices
        self.communication_qualities_matrix = np.zeros(
            (self.swarm_size, self.swarm_size)
        )
        self.distances_matrix = np.zeros((self.swarm_size, self.swarm_size))
        self.neighbor_agent_matrix = np.zeros((self.swarm_size, self.swarm_size))
        self.swarm_paths = []

        # Update the plot
        self.update_plot()

    def stop_simulation(self):
        self.running = False
        self.timer.stop()
        QApplication.quit()

    def on_click(self, event):
        if event.inaxes == self.axs[0, 0]:  # Only allow drawing in formation scene
            # Pause simulation when starting to draw
            self.paused = True
            self.timer.stop()
            self.drawing_obstacle = True
            self.obstacle_start = (event.xdata, event.ydata)

    def on_drag(self, event):
        if self.drawing_obstacle and event.inaxes:
            # Calculate radius from drag distance
            radius = np.sqrt(
                (event.xdata - self.obstacle_start[0]) ** 2
                + (event.ydata - self.obstacle_start[1]) ** 2
            )

            # Remove previous temporary circle if it exists
            if self.temp_circle is not None:
                self.temp_circle.remove()

            # Draw new temporary circle
            self.temp_circle = plt.Circle(
                self.obstacle_start, radius, color="red", alpha=0.3
            )
            self.axs[0, 0].add_artist(self.temp_circle)
            self.canvas.draw()

    def on_release(self, event):
        if self.drawing_obstacle and event.inaxes:
            radius = np.sqrt(
                (event.xdata - self.obstacle_start[0]) ** 2
                + (event.ydata - self.obstacle_start[1]) ** 2
            )

            # Add permanent obstacle
            self.obstacles.append(
                (self.obstacle_start[0], self.obstacle_start[1], radius)
            )

            # Clean up
            self.drawing_obstacle = False
            self.obstacle_start = None
            if self.temp_circle is not None:
                self.temp_circle.remove()
                self.temp_circle = None

            # Update plot with new obstacle
            self.update_plot()

            # Resume simulation properly
            self.paused = False
            self.running = True
            self.timer.start(50)

    def undo_last_obstacle(self):
        """Remove the most recently added obstacle"""
        if self.obstacles:  # Check if there are any obstacles
            self.obstacles.pop()  # Remove the last obstacle
            self.update_plot()  # Update the visualization


# Create and run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FormationControlGUI()
    window.show()
    sys.exit(app.exec_())
