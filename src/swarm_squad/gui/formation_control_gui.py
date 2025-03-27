"""
GUI for the formation control simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QPushButton, QVBoxLayout, QWidget

import swarm_squad.config as config
import swarm_squad.visualization as visualization
from swarm_squad.controllers.controller_factory import ControllerFactory, ControllerType
from swarm_squad.models.swarm_state import SwarmState


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

        # Create central widget
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

        # Initialize state and controllers
        self.swarm_state = SwarmState()
        self.controller_factory = ControllerFactory(self.swarm_state)

        # Set the combined controller as the active controller
        self.controller_factory.set_active_controller(ControllerType.COMBINED)

        # Initialize simulation control variables
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

        # Create control buttons
        self.create_plot_controls()

        # Set window size
        self.resize(800, 800)

        # Auto-start the simulation
        self.running = True
        self.timer.start()

    def create_plot_controls(self):
        """Create control buttons for the simulation"""
        # Create control frame (horizontal layout)
        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(10, 5, 10, 10)  # Add margins
        self.main_layout.addWidget(control_frame)

        # Button styles and colors
        button_width = 120
        button_height = 40

        # Create a common button style
        button_style = """
        QPushButton {
            font-family: 'Arial';
            font-size: 14px;
            font-weight: bold;
            border-radius: 5px;
            border: 1px solid #888888;
            padding: 5px;
            min-width: 100px;
        }
        QPushButton:hover {
            border: 2px solid #555555;
        }
        """

        # Create a common font
        button_font = QFont("Arial", 12, QFont.Bold)

        # Create buttons
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_simulation)
        self.pause_button.setStyleSheet(
            button_style + "background-color: #fdf2ca;"
        )  # Yellow
        self.pause_button.setFixedSize(button_width, button_height)
        self.pause_button.setFont(button_font)

        self.continue_button = QPushButton("Continue")
        self.continue_button.clicked.connect(self.continue_simulation)
        self.continue_button.setStyleSheet(
            button_style + "background-color: #e3f0d8;"
        )  # Green
        self.continue_button.setFixedSize(button_width, button_height)
        self.continue_button.setFont(button_font)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        self.reset_button.setStyleSheet(
            button_style + "background-color: #d8e3f0;"
        )  # Blue
        self.reset_button.setFixedSize(button_width, button_height)
        self.reset_button.setFont(button_font)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setStyleSheet(
            button_style + "background-color: #f9aeae;"
        )  # Red
        self.stop_button.setFixedSize(button_width, button_height)
        self.stop_button.setFont(button_font)

        # Add Undo button
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_last_obstacle)
        self.undo_button.setStyleSheet(
            button_style + "background-color: #e6e6e6;"
        )  # Light Gray
        self.undo_button.setFixedSize(button_width, button_height)
        self.undo_button.setFont(button_font)

        # Pack buttons horizontally with spacing
        control_layout.addWidget(self.pause_button)
        control_layout.addSpacing(10)  # Add spacing between buttons
        control_layout.addWidget(self.continue_button)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.reset_button)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.stop_button)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.undo_button)
        control_layout.addStretch()  # Add stretch to keep buttons left-aligned

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
                self.update_plot()
                return

            # Check if swarm center is close to destination
            if self.swarm_state.check_destination_reached():
                print("DEBUG: Destination reached check passed!")
                self.running = False
                self.timer.stop()
                print(
                    f"\n=== Mission Accomplished! ===\n"
                    f"Swarm has successfully reached the destination in:\n"
                    f"- Time: {round(self.swarm_state.t_elapsed[-1], 2)} seconds\n"
                    f"- Iterations: {self.swarm_state.iteration} steps\n"
                    f"- Final Jn value: {round(self.swarm_state.Jn[-1], 4)}\n"
                    f"==========================="
                )
                self.update_plot()  # Final update to show end state
            else:
                print("DEBUG: Destination not yet reached")

    def pause_simulation(self):
        """Pause the simulation"""
        self.paused = True
        self.running = False  # Stop the simulation loop
        self.timer.stop()

    def continue_simulation(self):
        """Continue the simulation after pause"""
        if not self.running:  # Only restart if not already running
            self.running = True
            self.paused = False

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

            # Create initial circle with 0 radius
            self.temp_circle = plt.Circle(
                self.obstacle_start, 0, color="red", alpha=0.3
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

    def undo_last_obstacle(self):
        """Remove the most recently added obstacle"""
        self.swarm_state.remove_last_obstacle()
        self.update_plot()  # Update the visualization
