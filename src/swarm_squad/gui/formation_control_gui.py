"""
GUI for the formation control simulation.
"""

import re
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
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

        # Using a horizontal layout to place plots on left, controls on right
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create left panel for plots
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Create main figure with subplots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.tight_layout(pad=3.0)  # Add padding between subplots

        # Create canvas for all plots
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        # Add matplotlib toolbar for additional navigation
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)

        # Only bind mouse events to the formation scene subplot
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Add plot panel to main layout
        self.main_layout.addWidget(plot_panel, 3)  # Give plots 3/4 of the width

        # Set window size - wider to accommodate side panel
        self.resize(1400, 800)

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

        # Add LLM feedback display timer - check more frequently (250ms)
        self.llm_feedback_timer = QTimer(self)
        self.llm_feedback_timer.timeout.connect(self.update_llm_feedback)
        self.llm_feedback_timer.setInterval(
            250
        )  # Check for new feedback more frequently

    def setup_simulation(self):
        """Set up simulation timer and start simulation."""
        self.running = True
        self.timer.start()

        # Start LLM feedback timer
        self.llm_feedback_timer.start()

    def create_plot_controls(self):
        """Create control buttons and layout for the simulation."""
        # Create right side panel for controls
        controls_container = QWidget()
        controls_vertical_layout = QVBoxLayout(controls_container)
        controls_vertical_layout.setContentsMargins(10, 5, 10, 10)

        # Add controls container to main layout (right side)
        self.main_layout.addWidget(controls_container, 1)  # 1/4 of the width

        # Add spacer at the top to push content down for vertical centering
        controls_vertical_layout.addStretch(1)

        # Create frames
        main_controls_frame = self.create_main_controls()
        obstacle_controls_frame = self.create_obstacle_controls()
        status_frame = self.create_status_bar()

        # Add frames to layout with spacing
        controls_vertical_layout.addWidget(main_controls_frame)
        controls_vertical_layout.addWidget(obstacle_controls_frame)
        controls_vertical_layout.addSpacing(STATUS_SPACING)
        controls_vertical_layout.addWidget(status_frame)

        # Create and add feedback panel
        feedback_frame = self.create_llm_feedback_panel()
        controls_vertical_layout.addWidget(
            feedback_frame, 2
        )  # Give it more vertical space (increased from 1 to 2)

        # Add spacer at the bottom to push content up for vertical centering
        controls_vertical_layout.addStretch(1)

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
        layout = QVBoxLayout(frame)  # Changed to vertical layout
        layout.setContentsMargins(0, 0, 0, 0)

        # Create horizontal layout for status labels
        status_layout = QHBoxLayout()
        status_layout.setAlignment(Qt.AlignCenter)

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
        status_layout.addWidget(self.simulation_status_label)
        status_layout.addWidget(self.spacer_label)
        status_layout.addWidget(self.obstacle_mode_label)

        # Add status layout to main layout
        layout.addLayout(status_layout)

        # Set initial status
        self.update_status_bar("Running", config.OBSTACLE_MODE.value)
        return frame

    def create_llm_feedback_panel(self):
        """Create the LLM feedback panel as a fixed widget in the layout"""
        # Create feedback panel container
        feedback_frame = QWidget()
        feedback_layout = QVBoxLayout(feedback_frame)
        feedback_layout.setContentsMargins(10, 10, 10, 10)

        # Make panel background visible
        feedback_frame.setStyleSheet(
            "background-color: rgba(220, 220, 255, 0.9); border-radius: 10px; border: 2px solid #3333aa;"
        )

        # Add title
        title_label = QLabel("LLM TACTICAL FEEDBACK")
        title_label.setStyleSheet("font-weight: bold; color: #333399; font-size: 16px;")
        title_label.setAlignment(Qt.AlignCenter)
        feedback_layout.addWidget(title_label)

        # Add current feedback label
        self.llm_feedback_label = QLabel("Waiting for tactical advice...")
        self.llm_feedback_label.setWordWrap(True)
        self.llm_feedback_label.setStyleSheet(
            "color: #333366; font-size: 14px; font-weight: bold; padding: 5px;"
        )
        self.llm_feedback_label.setAlignment(Qt.AlignCenter)
        self.llm_feedback_label.setMinimumHeight(50)
        feedback_layout.addWidget(self.llm_feedback_label)

        # Add timestamp for feedback (now centered)
        self.feedback_timestamp = QLabel("")
        self.feedback_timestamp.setStyleSheet("color: #4d4d4d; font-size: 11px;")
        self.feedback_timestamp.setAlignment(Qt.AlignCenter)  # Center alignment
        feedback_layout.addWidget(self.feedback_timestamp)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #8888cc;")
        feedback_layout.addWidget(separator)

        # Add previous feedback section
        self.prev_feedback_title = QLabel("PREVIOUS ADVICE:")
        self.prev_feedback_title.setStyleSheet(
            "color: #333399; font-size: 14px; font-weight: bold;"
        )
        self.prev_feedback_title.setAlignment(Qt.AlignCenter)  # Center alignment
        feedback_layout.addWidget(self.prev_feedback_title)

        self.prev_feedback_label = QLabel("No previous advice available")
        self.prev_feedback_label.setWordWrap(True)
        self.prev_feedback_label.setStyleSheet(
            "color: #4d4d4d; font-size: 13px; font-style: italic; padding: 5px;"
        )
        self.prev_feedback_label.setAlignment(Qt.AlignCenter)  # Center alignment
        self.prev_feedback_label.setMinimumHeight(40)
        feedback_layout.addWidget(self.prev_feedback_label)

        # Add timestamp for previous feedback
        self.prev_feedback_timestamp = QLabel("")
        self.prev_feedback_timestamp.setStyleSheet("color: #4d4d4d; font-size: 11px;")
        self.prev_feedback_timestamp.setAlignment(Qt.AlignCenter)  # Center alignment
        feedback_layout.addWidget(self.prev_feedback_timestamp)

        # Add second separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setStyleSheet("background-color: #8888cc;")
        feedback_layout.addWidget(separator2)

        # Add state perception section
        perceived_state_title = QLabel("PERCEIVED STATE:")
        perceived_state_title.setStyleSheet(
            "color: #333399; font-size: 14px; font-weight: bold;"
        )
        perceived_state_title.setAlignment(Qt.AlignCenter)
        feedback_layout.addWidget(perceived_state_title)

        # Create a container for the state label with border
        perceived_state_container = QWidget()
        perceived_state_container.setStyleSheet(
            "background-color: rgba(240, 240, 255, 0.7); border: 1px solid #8888cc; border-radius: 5px;"
        )
        perceived_state_layout = QVBoxLayout(perceived_state_container)
        perceived_state_layout.setContentsMargins(5, 5, 5, 5)

        # State content
        self.perceived_state_label = QLabel("Waiting for state information...")
        self.perceived_state_label.setWordWrap(True)
        self.perceived_state_label.setStyleSheet(
            "color: #333366; font-size: 12px; font-family: 'Courier New', monospace; padding: 5px; background-color: transparent;"
        )
        self.perceived_state_label.setAlignment(Qt.AlignLeft)
        self.perceived_state_label.setMinimumHeight(150)  # Minimum height
        self.perceived_state_label.setTextFormat(Qt.RichText)  # Use rich text format
        self.perceived_state_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )  # Make text selectable

        # Allow the label to expand as needed
        perceived_state_layout.addWidget(self.perceived_state_label)

        # Create a scroll area for the perceived state
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(perceived_state_container)
        scroll_area.setMinimumHeight(250)  # Increased minimum height
        scroll_area.setMaximumHeight(350)  # Set maximum height
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOn
        )  # Always show scrollbar
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(
            "border: none; background-color: transparent;"
        )  # Remove scroll area border since we have one on the container

        feedback_layout.addWidget(scroll_area, 2)  # Give it more stretch priority

        return feedback_frame

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

        # print(f"DEBUG: Obstacle mode changed to {mode.value}")

    def update_plot(self):
        """Update the plot with the current swarm state"""
        # Get LLM controller if enabled
        llm_controller = None
        if config.LLM_ENABLED:
            llm_controller = self.controller_factory.get_controller(ControllerType.LLM)

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
            llm_controller,
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
                # print("DEBUG: Destination reached check passed!")
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
        self.llm_feedback_timer.stop()  # Also stop LLM feedback updates
        self.update_status_bar("Paused", config.OBSTACLE_MODE.value)

    def continue_simulation(self):
        """Continue the simulation after pause"""
        if not self.running:  # Only restart if not already running
            self.running = True
            self.paused = False
            self.update_status_bar("Running", config.OBSTACLE_MODE.value)

            # Debug the controller status
            # print(
            #     f"DEBUG: continue_simulation - active controller: {self.controller_factory.active_controller_type}"
            # )

            if (
                self.swarm_state.Jn_converged
            ):  # Check if this is after formation convergence
                print("Simulation resumed.\nSwarm start reaching to the destination...")

                # Make sure we're using the combined controller
                self.controller_factory.set_active_controller(ControllerType.COMBINED)
                # print(
                #     f"DEBUG: Set active controller to {self.controller_factory.active_controller_type}"
                # )

            self.timer.start()
            self.llm_feedback_timer.start()  # Restart LLM feedback updates

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

    def update_llm_feedback(self):
        """Update the LLM feedback display with latest information"""
        if not config.LLM_ENABLED:
            return

        # Get LLM controller
        llm_controller = self.controller_factory.get_controller(ControllerType.LLM)

        # Get latest feedback
        current_feedback = llm_controller.get_last_feedback()

        if current_feedback and current_feedback != self.llm_feedback_label.text():
            # Store previous feedback before updating
            prev_feedback = self.llm_feedback_label.text()
            if (
                prev_feedback
                and prev_feedback != "Waiting for tactical advice..."
                and prev_feedback
                != "No feedback received from LLM. Please check Ollama is running correctly."
            ):
                self.prev_feedback_label.setText(prev_feedback)

                # Update previous feedback timestamp
                prev_time = self.feedback_timestamp.text().replace("Updated at ", "")
                self.prev_feedback_timestamp.setText(prev_time)

            # Update current feedback
            self.llm_feedback_label.setText(current_feedback)

            # Update timestamp
            current_time = time.strftime("%H:%M:%S", time.localtime())
            self.feedback_timestamp.setText(
                f"Updated at {current_time} (Iteration: {self.swarm_state.iteration})"
            )

            # Update perceived state information
            if (
                hasattr(llm_controller, "last_state_description")
                and llm_controller.last_state_description
            ):
                # Format the state description for better display
                formatted_state = self._format_state_for_display(
                    llm_controller.last_state_description
                )
                self.perceived_state_label.setText(formatted_state)
            else:
                # Generate simplified state description as fallback
                state_desc = []
                state_desc.append(
                    f"Destination: [{self.swarm_state.swarm_destination[0]:.1f}, {self.swarm_state.swarm_destination[1]:.1f}]"
                )

                for i in range(self.swarm_state.swarm_size):
                    pos = self.swarm_state.swarm_position[i]
                    conn_count = sum(
                        self.swarm_state.neighbor_agent_matrix[i, :] > config.PT
                    )
                    state_desc.append(
                        f"Agent-{i} at [{pos[0]:.1f}, {pos[1]:.1f}] with {conn_count} connections"
                    )

                self.perceived_state_label.setText("\n".join(state_desc))

    def _format_state_for_display(self, state_description):
        """Format the state description for better display in the UI"""
        lines = state_description.split("\n")
        formatted_lines = []

        # Convert NODE_COLORS to hex for HTML use
        node_colors_hex = []
        for color in config.NODE_COLORS:
            r, g, b = color
            hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            node_colors_hex.append(hex_color)

        # Track current agent for coloring
        current_agent_idx = -1

        for line in lines:
            # Skip empty lines
            if not line.strip():
                formatted_lines.append("<br>")
                continue

            # Highlight destination info
            if "destination" in line.lower():
                formatted_lines.append(
                    f"<span style='color:#006699; font-weight:bold; background-color: rgba(0, 102, 153, 0.2);'>{line}</span>"
                )
            # Highlight obstacles
            elif "obstacle" in line.lower():
                formatted_lines.append(
                    f"<span style='color:#994400; font-weight:bold; background-color: rgba(153, 68, 0, 0.2);'>{line}</span>"
                )
            # Format agent information
            elif any(f"Agent-{i}" in line for i in range(10)):
                # Identify which agent this is to use the right color
                for i in range(10):
                    if f"Agent-{i}" in line:
                        current_agent_idx = i
                        break

                # Get agent color (with fallback)
                agent_color = "#333366"  # Default color
                if 0 <= current_agent_idx < len(node_colors_hex):
                    agent_color = node_colors_hex[current_agent_idx]

                # Format agent name in color and bold
                agent_name_match = re.search(r"(Agent-\d+)", line)
                if agent_name_match:
                    agent_name = agent_name_match.group(1)
                    colored_line = line.replace(
                        agent_name,
                        f"<span style='color:{agent_color}; font-weight:bold;'>{agent_name}</span>",
                    )
                    formatted_lines.append(colored_line)
                else:
                    formatted_lines.append(line)
            # Format bullet points for agent connections
            elif line.strip().startswith("  - Agent-"):
                # Extract the agent number from the line
                other_agent_match = re.search(r"Agent-(\d+)", line)
                if other_agent_match:
                    other_agent_idx = int(other_agent_match.group(1))
                    other_agent = f"Agent-{other_agent_idx}"

                    # Get color for the other agent
                    other_color = "#333366"  # Default
                    if 0 <= other_agent_idx < len(node_colors_hex):
                        other_color = node_colors_hex[other_agent_idx]

                    # Color the agent name in the line
                    colored_line = line.replace(
                        other_agent,
                        f"<span style='color:{other_color}; font-weight:bold;'>{other_agent}</span>",
                    )

                    # Color the connection quality info
                    if "poor quality" in colored_line:
                        # Highlight poor quality in red
                        colored_line = colored_line.replace(
                            "poor quality",
                            "<span style='color:#cc0000; font-weight:bold;'>poor quality</span>",
                        )
                    elif "good quality" in colored_line:
                        # Highlight good quality in green
                        colored_line = colored_line.replace(
                            "good quality",
                            "<span style='color:#006600; font-weight:bold;'>good quality</span>",
                        )

                    # Highlight connection status
                    if "connected" in colored_line:
                        colored_line = colored_line.replace(
                            "connected",
                            "<span style='color:#006600; font-weight:bold;'>connected</span>",
                        )
                    elif "disconnected" in colored_line:
                        colored_line = colored_line.replace(
                            "disconnected",
                            "<span style='color:#cc0000; font-weight:bold;'>disconnected</span>",
                        )

                    # Indent and format as a bullet point
                    formatted_lines.append(f"&nbsp;&nbsp;• {colored_line[4:]}")
                else:
                    formatted_lines.append(f"&nbsp;&nbsp;• {line[4:]}")
            # Handle mission info and other lines
            elif "mission" in line.lower():
                formatted_lines.append(
                    f"<span style='color:#663399; font-weight:bold;'>{line}</span>"
                )
            else:
                formatted_lines.append(line)

        return "<br>".join(formatted_lines)

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
