"""
GUI for the formation control simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import swarm_squad.config as config
import swarm_squad.visualization as visualization
from swarm_squad.controllers.controller_factory import ControllerFactory, ControllerType
from swarm_squad.models.swarm_state import SwarmState

# UI Constants
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_FONT = QFont("Arial", 12)
BUTTON_SPACING = 10
STATUS_SPACING = 30

# Common Styles
COMMON_BUTTON_STYLE = """
    font-family: Arial;
    font-size: 14px;
    color: black;
    border-radius: 5px;
    border: 1px solid #888888;
    padding: 5px;
    min-width: 100px;
"""

COMMON_LABEL_STYLE = """
    font-family: 'Arial';
    font-size: 14px;
    color: black;
    border-radius: 5px;
    border: 1px solid #888888;
    padding: 8px 15px;
"""

# Color Constants
COLORS = {
    "pause": "#fdf2ca",
    "continue": "#e3f0d8",
    "reset": "#d8e3f0",
    "stop": "#f9aeae",
    "undo": "#c0c0c0",
    "hard": "#c0c0c0",
    "low_power": "#fdf2ca",
    "high_power": "#f9aeae",
}


class FormationControlGUI(QMainWindow):
    """
    GUI for the formation control simulation.

    This class handles the visualization and user interaction for the
    formation control simulation.
    """

    def __init__(self, parent=None):
        """
        Initialize the GUI.

        Args:
            parent: The parent widget (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("Formation Control Simulation")
        self.setup_main_window()
        self.initialize_state()
        self.create_plot_controls()
        self.setup_simulation()

    def setup_main_window(self):
        """Set up the main window and matplotlib components."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create main figure with subplots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 10))
        self.fig.tight_layout(pad=3.0)  # Add padding between subplots

        # Create canvas for all plots
        self.canvas = FigureCanvas(self.fig)
        self.main_layout.addWidget(self.canvas)

        # Add matplotlib toolbar for additional navigation
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)

        # Only bind mouse events to the formation scene subplot
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Set window size
        self.resize(800, 800)

    def initialize_state(self):
        """Initialize simulation state and variables."""
        self.swarm_state = SwarmState()
        self.controller_factory = ControllerFactory(self.swarm_state)
        self.controller_factory.set_active_controller(ControllerType.COMBINED)

        self.running = False
        self.paused = False
        self.max_iter = config.MAX_ITER
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.simulation_step)
        self.timer.setInterval(50)  # 50ms interval, similar to the Tkinter version

        # Add drawing state variables
        self.drawing_obstacle = False
        self.obstacle_start = None
        self.temp_circle = None  # Store temporary circle while drawing

    def setup_simulation(self):
        """Set up simulation timer and start simulation."""
        self.running = True
        self.timer.start()

    def create_plot_controls(self):
        """Create control buttons and layout for the simulation."""
        controls_container = QWidget()
        controls_vertical_layout = QVBoxLayout(controls_container)
        controls_vertical_layout.setContentsMargins(10, 5, 10, 10)
        self.main_layout.addWidget(controls_container)

        # Create frames
        main_controls_frame = self.create_main_controls()
        obstacle_controls_frame = self.create_obstacle_controls()
        status_frame = self.create_status_bar()

        # Add frames to layout with spacing
        controls_vertical_layout.addWidget(main_controls_frame)
        controls_vertical_layout.addWidget(obstacle_controls_frame)
        controls_vertical_layout.addSpacing(STATUS_SPACING)
        controls_vertical_layout.addWidget(status_frame)

    def create_main_controls(self):
        """Create main control buttons."""
        frame = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 5)
        layout.setAlignment(Qt.AlignCenter)

        # Define button configurations
        buttons = [
            ("Pause", self.pause_simulation, COLORS["pause"]),
            ("Continue", self.continue_simulation, COLORS["continue"]),
            ("Reset", self.reset_simulation, COLORS["reset"]),
            ("Stop", self.stop_simulation, COLORS["stop"]),
            ("Undo", self.undo_last_obstacle, COLORS["undo"]),
        ]

        # Create buttons
        for text, callback, color in buttons:
            button = self.create_button(text, callback, color)
            layout.addWidget(button)
            layout.addSpacing(BUTTON_SPACING)
            if text == "Pause":
                self.pause_button = button
            elif text == "Continue":
                self.continue_button = button

        return frame

    def create_button(self, text, callback, color):
        """Create a styled button with given parameters."""
        button = QPushButton(text)
        button.clicked.connect(callback)
        button.setStyleSheet(f"{COMMON_BUTTON_STYLE} background-color: {color};")
        button.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        button.setFont(BUTTON_FONT)
        return button

    def create_obstacle_controls(self):
        """Create obstacle mode control buttons."""
        frame = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 5)
        layout.setAlignment(Qt.AlignCenter)

        self.mode_buttons = {}

        # Define obstacle modes
        modes = [
            (config.ObstacleMode.HARD, "Physical", COLORS["hard"]),
            (config.ObstacleMode.LOW_POWER_JAMMING, "Low Power", COLORS["low_power"]),
            (
                config.ObstacleMode.HIGH_POWER_JAMMING,
                "High Power",
                COLORS["high_power"],
            ),
        ]

        # Create mode buttons
        for mode, text, color in modes:
            button = self.create_mode_button(mode, text, color)
            layout.addWidget(button)
            layout.addSpacing(BUTTON_SPACING)

        # Set initial mode
        self.mode_buttons[config.OBSTACLE_MODE].setChecked(True)
        return frame

    def create_mode_button(self, mode, text, color):
        """Create a mode selection button."""
        button = QPushButton(text)
        button.setCheckable(True)
        button.setStyleSheet(f"{COMMON_BUTTON_STYLE} background-color: {color};")
        button.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        button.setFont(BUTTON_FONT)
        button.clicked.connect(lambda: self.on_mode_button_clicked(mode))
        self.mode_buttons[mode] = button
        return button

    def create_status_bar(self):
        """Create status bar with labels."""
        frame = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        # Create status labels
        self.simulation_status_label = QLabel("Simulation Status: Running")
        self.simulation_status_label.setFont(BUTTON_FONT)
        self.simulation_status_label.setStyleSheet(
            f"{COMMON_LABEL_STYLE} background-color: {COLORS['continue']};"
        )

        self.spacer_label = QLabel("   ")

        self.obstacle_mode_label = QLabel("Obstacle Mode: Physical")
        self.obstacle_mode_label.setFont(BUTTON_FONT)
        self.obstacle_mode_label.setStyleSheet(
            f"{COMMON_LABEL_STYLE} background-color: {COLORS['hard']};"
        )

        # Add labels to layout
        layout.addWidget(self.simulation_status_label)
        layout.addWidget(self.spacer_label)
        layout.addWidget(self.obstacle_mode_label)

        # Set initial status
        self.update_status_bar("Running", config.OBSTACLE_MODE.value)
        return frame

    def on_mode_button_clicked(self, mode):
        """Handle mode button click"""
        # Update the checked state of all buttons
        for button_mode, button in self.mode_buttons.items():
            button.setChecked(button_mode == mode)

        # Update the config
        config.OBSTACLE_MODE = mode

        # Update the status bar
        if self.running:
            status = "Running"
        elif self.paused:
            status = "Paused"
        else:
            status = "Ready"
        self.update_status_bar(status, mode.value)

        # Update the plot to reflect changes
        self.update_plot()

        print(f"DEBUG: Obstacle mode changed to {mode.value}")

    def update_plot(self):
        """Update the plot with the current swarm state"""
        visualization.plot_all_figures(
            self.axs,
            self.swarm_state.t_elapsed,
            self.swarm_state.Jn,
            self.swarm_state.rn,
            self.swarm_state.swarm_position,
            config.PT,
            self.swarm_state.communication_qualities_matrix,
            self.swarm_state.swarm_size,
            self.swarm_state.swarm_paths,
            config.NODE_COLORS,
            self.swarm_state.line_colors,
            self.swarm_state.obstacles,
            self.swarm_state.swarm_destination,
            self.swarm_state.agent_status,
            self.swarm_state.jamming_affected,
        )
        self.canvas.draw_idle()  # Use draw_idle for better performance

    def simulation_step(self):
        """Perform a single step of the simulation"""
        if (
            self.running
            and not self.paused
            and self.swarm_state.iteration < self.max_iter
        ):
            # Perform the control step
            self.controller_factory.step()

            # Update the plot
            self.update_plot()

            # Direct check for convergence (only if not already converged)
            if (
                not self.swarm_state.Jn_converged
                and self.swarm_state.check_convergence()
            ):
                print(
                    f"Formation completed: Jn values has converged in {round(self.swarm_state.t_elapsed[-1], 2)} seconds {self.swarm_state.iteration - 20} iterations.\nSimulation paused."
                )
                self.swarm_state.Jn_converged = True
                self.running = False
                self.timer.stop()
                self.update_status_bar(
                    "Formation Converged", config.OBSTACLE_MODE.value
                )
                self.update_plot()
                return

            # Check if swarm center is close to destination
            if self.swarm_state.check_destination_reached():
                print("DEBUG: Destination reached check passed!")
                self.running = False
                self.timer.stop()
                self.update_status_bar(
                    "Destination Reached", config.OBSTACLE_MODE.value
                )
                print(
                    f"\n=== Mission Accomplished! ===\n"
                    f"Swarm has successfully reached the destination in:\n"
                    f"- Time: {round(self.swarm_state.t_elapsed[-1], 2)} seconds\n"
                    f"- Iterations: {self.swarm_state.iteration} steps\n"
                    f"- Final Jn value: {round(self.swarm_state.Jn[-1], 4)}\n"
                    f"==========================="
                )
                self.update_plot()  # Final update to show end state

    def pause_simulation(self):
        """Pause the simulation"""
        self.paused = True
        self.running = False  # Stop the simulation loop
        self.timer.stop()
        self.update_status_bar("Paused", config.OBSTACLE_MODE.value)

    def continue_simulation(self):
        """Continue the simulation after pause"""
        if not self.running:  # Only restart if not already running
            self.running = True
            self.paused = False
            self.update_status_bar("Running", config.OBSTACLE_MODE.value)

            # Debug the controller status
            print(
                f"DEBUG: continue_simulation - active controller: {self.controller_factory.active_controller_type}"
            )

            if (
                self.swarm_state.Jn_converged
            ):  # Check if this is after formation convergence
                print("Simulation resumed.\nSwarm start reaching to the destination...")

                # Make sure we're using the combined controller
                self.controller_factory.set_active_controller(ControllerType.COMBINED)
                print(
                    f"DEBUG: Set active controller to {self.controller_factory.active_controller_type}"
                )

            self.timer.start()

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        # Reset the simulation
        self.running = False
        self.paused = False
        self.timer.stop()

        # Reset the swarm state
        self.swarm_state.reset()

        # Update the plot
        self.update_plot()
        self.update_status_bar("Reset", config.OBSTACLE_MODE.value)

    def stop_simulation(self):
        """Stop the simulation and close the application"""
        self.running = False
        self.timer.stop()
        self.close()  # Close the window

    def on_click(self, event):
        """Handle mouse click events for drawing obstacles"""
        if event.inaxes == self.axs[0, 0]:  # Only allow drawing in formation scene
            # Pause simulation when starting to draw
            self.paused = True
            self.timer.stop()
            self.drawing_obstacle = True
            self.obstacle_start = (event.xdata, event.ydata)

            # Select color based on current obstacle mode
            obstacle_color = "gray"  # Default for hard obstacles

            if config.OBSTACLE_MODE == config.ObstacleMode.LOW_POWER_JAMMING:
                obstacle_color = "yellow"
            elif config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING:
                obstacle_color = "red"

            # Create initial circle with 0 radius
            self.temp_circle = plt.Circle(
                self.obstacle_start, 0, color=obstacle_color, alpha=0.3
            )
            self.axs[0, 0].add_artist(self.temp_circle)
            self.canvas.draw_idle()

    def on_drag(self, event):
        """Handle mouse drag events for drawing obstacles"""
        if self.drawing_obstacle and event.inaxes:
            # Calculate radius from drag distance
            radius = np.sqrt(
                (event.xdata - self.obstacle_start[0]) ** 2
                + (event.ydata - self.obstacle_start[1]) ** 2
            )

            # Update circle radius
            if self.temp_circle is not None:
                self.temp_circle.set_radius(radius)
                self.canvas.draw_idle()  # Use draw_idle for better performance during drag

    def on_release(self, event):
        """Handle mouse release events for placing obstacles"""
        if self.drawing_obstacle and event.inaxes:
            radius = np.sqrt(
                (event.xdata - self.obstacle_start[0]) ** 2
                + (event.ydata - self.obstacle_start[1]) ** 2
            )

            # Add permanent obstacle
            self.swarm_state.add_obstacle(
                self.obstacle_start[0], self.obstacle_start[1], radius
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
            self.timer.start()

            # Update the status bar to show "Running"
            self.update_status_bar("Running", config.OBSTACLE_MODE.value)

    def undo_last_obstacle(self):
        """Remove the most recently added obstacle"""
        self.swarm_state.remove_last_obstacle()
        self.update_plot()  # Update the visualization

    def update_status_bar(self, simulation_status, obstacle_mode):
        """Update the status bar with current simulation status and obstacle mode"""
        # Format obstacle mode text
        obstacle_mode_text = obstacle_mode.replace("_", " ").title()

        # Set simulation status with appropriate color
        self.simulation_status_label.setText(f"Simulation Status: {simulation_status}")

        # Set obstacle mode with appropriate color
        self.obstacle_mode_label.setText(f"Obstacle Mode: {obstacle_mode_text}")

        # Set color based on simulation status
        if simulation_status == "Running":
            self.simulation_status_label.setStyleSheet(
                f"{COMMON_LABEL_STYLE} background-color: {COLORS['continue']};"
            )
        elif simulation_status == "Paused":
            self.simulation_status_label.setStyleSheet(
                f"{COMMON_LABEL_STYLE} background-color: {COLORS['pause']};"
            )
        elif simulation_status == "Reset":
            self.simulation_status_label.setStyleSheet(
                f"{COMMON_LABEL_STYLE} background-color: {COLORS['reset']};"
            )
        else:
            # For other statuses like "Formation Converged" or "Destination Reached"
            self.simulation_status_label.setStyleSheet(
                f"{COMMON_LABEL_STYLE} background-color: {COLORS['stop']};"
            )

        # Set obstacle mode with appropriate color
        if obstacle_mode == config.ObstacleMode.HARD.value:
            self.obstacle_mode_label.setStyleSheet(
                f"{COMMON_LABEL_STYLE} background-color: {COLORS['hard']};"
            )
        elif obstacle_mode == config.ObstacleMode.LOW_POWER_JAMMING.value:
            self.obstacle_mode_label.setStyleSheet(
                f"{COMMON_LABEL_STYLE} background-color: {COLORS['low_power']};"
            )
        elif obstacle_mode == config.ObstacleMode.HIGH_POWER_JAMMING.value:
            self.obstacle_mode_label.setStyleSheet(
                f"{COMMON_LABEL_STYLE} background-color: {COLORS['high_power']};"
            )
