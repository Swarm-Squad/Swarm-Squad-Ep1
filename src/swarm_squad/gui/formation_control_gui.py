"""
GUI for the formation control simulation.
"""

import os
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
BUTTON_WIDTH = 150
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

    def __init__(self, parent=None, llm_model=None, llm_feedback_interval=None):
        """
        Initialize the GUI.

        Args:
            parent: The parent widget (optional)
            llm_model: Custom LLM model to use (overrides config)
            llm_feedback_interval: Custom LLM feedback interval (overrides config)
        """
        super().__init__(parent)

        # Store custom LLM settings
        self.llm_model = llm_model if llm_model is not None else config.LLM_MODEL
        self.llm_feedback_interval = (
            llm_feedback_interval
            if llm_feedback_interval is not None
            else config.LLM_FEEDBACK_INTERVAL
        )

        self.setWindowTitle("Formation Control Simulation")

        # Initialize class variables
        self.running = False
        self.paused = False
        self.max_iter = config.MAX_ITER
        self.drawing_obstacle = False
        self.obstacle_start = None
        self.temp_circle = None
        self.mode_buttons = {}

        # Setup the UI and simulation components
        self._setup_main_window()
        self._initialize_state()
        self._create_plot_controls()
        self._setup_simulation()

    def _setup_main_window(self):
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

        # Connect mouse events to the formation scene subplot
        self._connect_mouse_events()

        # Add plot panel to main layout
        self.main_layout.addWidget(plot_panel, 3)  # Give plots 3/4 of the width

        # Set window size - wider to accommodate side panel
        self.resize(1400, 800)

    def _connect_mouse_events(self):
        """Connect mouse events to the canvas for obstacle drawing."""
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def _initialize_state(self):
        """Initialize simulation state and variables."""
        self.swarm_state = SwarmState()
        self.controller_factory = ControllerFactory(
            self.swarm_state,
            llm_model=self.llm_model,
            llm_feedback_interval=self.llm_feedback_interval,
        )
        self.controller_factory.set_active_controller(ControllerType.COMBINED)

        # Set up timers
        self._initialize_timers()

    def _initialize_timers(self):
        """Initialize simulation and LLM feedback timers."""
        # Simulation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.simulation_step)
        self.timer.setInterval(50)  # 50ms interval

        # LLM feedback display timer
        self.llm_feedback_timer = QTimer(self)
        self.llm_feedback_timer.timeout.connect(self.update_llm_feedback)
        self.llm_feedback_timer.setInterval(250)  # 250ms interval

    def _setup_simulation(self):
        """Set up simulation timer and start simulation."""
        self.running = True
        self.timer.start()
        self.llm_feedback_timer.start()

    def _create_plot_controls(self):
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
        main_controls_frame = self._create_main_controls()
        obstacle_controls_frame = self._create_obstacle_controls()
        status_frame = self._create_status_bar()

        # Add frames to layout with spacing
        controls_vertical_layout.addWidget(main_controls_frame)
        controls_vertical_layout.addWidget(obstacle_controls_frame)
        controls_vertical_layout.addSpacing(STATUS_SPACING)
        controls_vertical_layout.addWidget(status_frame)

        # Create and add feedback panel
        feedback_frame = self._create_llm_feedback_panel()
        controls_vertical_layout.addWidget(
            feedback_frame, 2
        )  # Give it more vertical space

        # Add spacer at the bottom to push content up for vertical centering
        controls_vertical_layout.addStretch(1)

    def _create_main_controls(self):
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
            button = self._create_button(text, callback, color)
            layout.addWidget(button)
            layout.addSpacing(BUTTON_SPACING)
            if text == "Pause":
                self.pause_button = button
            elif text == "Continue":
                self.continue_button = button

        return frame

    def _create_button(self, text, callback, color):
        """Create a styled button with given parameters."""
        button = QPushButton(text)
        button.clicked.connect(callback)
        button.setStyleSheet(f"{COMMON_BUTTON_STYLE} background-color: {color};")
        button.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        button.setFont(BUTTON_FONT)
        return button

    def _create_obstacle_controls(self):
        """Create obstacle mode control buttons."""
        frame = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 5)
        layout.setAlignment(Qt.AlignCenter)

        # Define obstacle modes
        modes = [
            (config.ObstacleMode.HARD, "Physical Obstacle", COLORS["hard"]),
            (
                config.ObstacleMode.LOW_POWER_JAMMING,
                "Low Power Jamming",
                COLORS["low_power"],
            ),
            (
                config.ObstacleMode.HIGH_POWER_JAMMING,
                "High Power Jamming",
                COLORS["high_power"],
            ),
        ]

        # Create mode buttons
        for mode, text, color in modes:
            button = self._create_mode_button(mode, text, color)
            layout.addWidget(button)
            layout.addSpacing(BUTTON_SPACING)

        # Set initial mode
        self.mode_buttons[config.OBSTACLE_MODE].setChecked(True)
        return frame

    def _create_mode_button(self, mode, text, color):
        """Create a mode selection button."""
        button = QPushButton(text)
        button.setCheckable(True)
        button.setStyleSheet(f"{COMMON_BUTTON_STYLE} background-color: {color};")
        button.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
        button.setFont(BUTTON_FONT)
        button.clicked.connect(lambda: self.on_mode_button_clicked(mode))
        self.mode_buttons[mode] = button
        return button

    def _create_status_bar(self):
        """Create status bar with labels."""
        frame = QWidget()
        layout = QVBoxLayout(frame)  # Vertical layout
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

    def _create_llm_feedback_panel(self):
        """Create the LLM feedback panel as a fixed widget in the layout"""
        # Create feedback panel container
        feedback_frame = QWidget()
        feedback_layout = QVBoxLayout(feedback_frame)
        feedback_layout.setContentsMargins(10, 10, 10, 10)

        # Make panel background visible
        feedback_frame.setStyleSheet(
            "background-color: rgba(220, 220, 255, 0.9); border-radius: 10px; border: 2px solid #3333aa;"
        )

        # Create feedback panel sections
        self._create_feedback_title(feedback_layout)
        self._create_current_feedback_section(feedback_layout)
        self._add_separator(feedback_layout)
        self._create_previous_feedback_section(feedback_layout)
        self._add_separator(feedback_layout)
        self._create_perceived_state_section(feedback_layout)

        # Add log file display section if LLM is enabled
        if config.LLM_ENABLED:
            self._add_separator(feedback_layout)
            self._create_log_file_section(feedback_layout)

        return feedback_frame

    def _create_feedback_title(self, layout):
        """Add title to the feedback panel."""
        title_label = QLabel("LLM TACTICAL FEEDBACK")
        title_label.setStyleSheet("font-weight: bold; color: #333399; font-size: 16px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

    def _create_current_feedback_section(self, layout):
        """Create the current feedback section."""
        # Add current feedback label
        self.llm_feedback_label = QLabel("Waiting for tactical advice...")
        self.llm_feedback_label.setWordWrap(True)
        self.llm_feedback_label.setStyleSheet(
            "color: #333366; font-size: 14px; font-weight: bold; padding: 5px;"
        )
        self.llm_feedback_label.setAlignment(Qt.AlignCenter)
        self.llm_feedback_label.setMinimumHeight(50)
        layout.addWidget(self.llm_feedback_label)

        # Add timestamp for feedback
        self.feedback_timestamp = QLabel("")
        self.feedback_timestamp.setStyleSheet("color: #4d4d4d; font-size: 11px;")
        self.feedback_timestamp.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.feedback_timestamp)

    def _add_separator(self, layout):
        """Add a separator line to the layout."""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #8888cc;")
        layout.addWidget(separator)

    def _create_previous_feedback_section(self, layout):
        """Create the previous feedback section."""
        # Add previous feedback section title
        self.prev_feedback_title = QLabel("PREVIOUS ADVICE:")
        self.prev_feedback_title.setStyleSheet(
            "color: #333399; font-size: 14px; font-weight: bold;"
        )
        self.prev_feedback_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prev_feedback_title)

        # Add previous feedback content
        self.prev_feedback_label = QLabel("No previous advice available")
        self.prev_feedback_label.setWordWrap(True)
        self.prev_feedback_label.setStyleSheet(
            "color: #4d4d4d; font-size: 13px; font-style: italic; padding: 5px;"
        )
        self.prev_feedback_label.setAlignment(Qt.AlignCenter)
        self.prev_feedback_label.setMinimumHeight(40)
        layout.addWidget(self.prev_feedback_label)

        # Add timestamp for previous feedback
        self.prev_feedback_timestamp = QLabel("")
        self.prev_feedback_timestamp.setStyleSheet("color: #4d4d4d; font-size: 11px;")
        self.prev_feedback_timestamp.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prev_feedback_timestamp)

    def _create_perceived_state_section(self, layout):
        """Create the perceived state section."""
        # Add state perception section title
        perceived_state_title = QLabel("PERCEIVED STATE:")
        perceived_state_title.setStyleSheet(
            "color: #333399; font-size: 14px; font-weight: bold;"
        )
        perceived_state_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(perceived_state_title)

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
        self.perceived_state_label.setMinimumHeight(150)
        self.perceived_state_label.setTextFormat(Qt.RichText)
        self.perceived_state_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Allow the label to expand as needed
        perceived_state_layout.addWidget(self.perceived_state_label)

        # Create a scroll area for the perceived state
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(perceived_state_container)
        scroll_area.setMinimumHeight(250)
        scroll_area.setMaximumHeight(350)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("border: none; background-color: transparent;")

        layout.addWidget(scroll_area, 2)  # Give it more stretch priority

    def _create_log_file_section(self, layout):
        """Create a section to show the log file path and provide a button to open it."""
        # Create container for log file info
        log_container = QWidget()
        log_layout = QHBoxLayout(log_container)
        log_layout.setContentsMargins(5, 5, 5, 5)

        # Create label for log file path
        self.log_file_label = QLabel("LLM Log: Not started yet")
        self.log_file_label.setWordWrap(True)
        self.log_file_label.setStyleSheet("color: #333399; font-size: 11px;")

        # Create button to open log file
        self.open_log_button = QPushButton("Open Log")
        self.open_log_button.setStyleSheet(
            "background-color: #6666cc; color: white; border-radius: 5px; padding: 5px;"
        )
        self.open_log_button.setEnabled(False)
        self.open_log_button.clicked.connect(self._open_log_file)

        # Add widgets to layout
        log_layout.addWidget(self.log_file_label, 3)  # 3:1 ratio
        log_layout.addWidget(self.open_log_button, 1)

        # Add container to main layout
        layout.addWidget(log_container)

    def _open_log_file(self):
        """Open the log file with the system's default text editor."""
        llm_controller = self._get_llm_controller()
        if llm_controller and hasattr(llm_controller, "log_file_path"):
            log_path = llm_controller.log_file_path
            if os.path.exists(log_path):
                import platform
                import subprocess

                # Open the file based on the operating system
                if platform.system() == "Windows":
                    os.startfile(log_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.call(("open", log_path))
                else:  # Linux and other Unix-like systems
                    subprocess.call(("xdg-open", log_path))

    def on_mode_button_clicked(self, mode):
        """Handle mode button click"""
        self._update_mode_buttons(mode)
        self._update_config_and_status(mode)
        self.update_plot()

    def _update_mode_buttons(self, mode):
        """Update the checked state of all mode buttons."""
        for button_mode, button in self.mode_buttons.items():
            button.setChecked(button_mode == mode)

    def _update_config_and_status(self, mode):
        """Update the configuration and status bar with the selected mode."""
        # Update the config
        config.OBSTACLE_MODE = mode

        # Update the swarm_state's obstacle_mode attribute
        self.swarm_state.obstacle_mode = mode

        # Update the status bar
        status = self._get_current_status()
        self.update_status_bar(status, mode.value)

    def _get_current_status(self):
        """Get the current simulation status string."""
        if self.running:
            return "Running"
        elif self.paused:
            return "Paused"
        else:
            return "Ready"

    def update_plot(self):
        """Update the plot with the current swarm state"""
        # Get LLM controller if enabled
        llm_controller = self._get_llm_controller()

        # Update all figures
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

    def _get_llm_controller(self):
        """Get the LLM controller if enabled."""
        if config.LLM_ENABLED:
            return self.controller_factory.get_controller(ControllerType.LLM)
        return None

    def simulation_step(self):
        """Perform a single step of the simulation"""
        if not self._should_run_simulation():
            return

        # Perform the control step
        self.controller_factory.step()

        # Update the plot
        self.update_plot()

        # Check for convergence or destination reached
        if (
            self._check_and_handle_convergence()
            or self._check_and_handle_destination_reached()
        ):
            return

    def _should_run_simulation(self):
        """Check if the simulation should continue running."""
        return (
            self.running
            and not self.paused
            and self.swarm_state.iteration < self.max_iter
        )

    def _check_and_handle_convergence(self):
        """Check if the formation has converged and handle it."""
        if not self.swarm_state.Jn_converged and self.swarm_state.check_convergence():
            self._handle_convergence()
            return True
        return False

    def _handle_convergence(self):
        """Handle the formation convergence event."""
        print(
            f"Formation completed: Jn values has converged in {round(self.swarm_state.t_elapsed[-1], 2)} seconds {self.swarm_state.iteration - 20} iterations.\nSimulation paused."
        )
        self.swarm_state.Jn_converged = True
        self.running = False
        self.timer.stop()
        self.update_status_bar("Formation Converged", config.OBSTACLE_MODE.value)
        self.update_plot()

    def _check_and_handle_destination_reached(self):
        """Check if the swarm has reached its destination and handle it."""
        if self.swarm_state.check_destination_reached():
            self._handle_destination_reached()
            return True
        return False

    def _handle_destination_reached(self):
        """Handle the destination reached event."""
        self.running = False
        self.timer.stop()
        self.update_status_bar("Destination Reached", config.OBSTACLE_MODE.value)
        self._print_mission_accomplished()
        self.update_plot()  # Final update to show end state

    def _print_mission_accomplished(self):
        """Print mission accomplished message with statistics."""
        print(
            f"\n=== Mission Accomplished! ===\n"
            f"Swarm has successfully reached the destination in:\n"
            f"- Time: {round(self.swarm_state.t_elapsed[-1], 2)} seconds\n"
            f"- Iterations: {self.swarm_state.iteration} steps\n"
            f"- Final Jn value: {round(self.swarm_state.Jn[-1], 4)}\n"
            f"==========================="
        )

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

            # Check if this is after formation convergence
            if self.swarm_state.Jn_converged:
                print("Simulation resumed.\nSwarm start reaching to the destination...")
                self.controller_factory.set_active_controller(ControllerType.COMBINED)

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
        if event.inaxes != self.axs[0, 0]:
            return  # Only allow drawing in formation scene

        # Pause simulation when starting to draw
        self._pause_for_drawing()

        # Initialize obstacle drawing
        self.drawing_obstacle = True
        self.obstacle_start = (event.xdata, event.ydata)

        # Create initial obstacle circle
        self._create_temporary_obstacle()

    def _pause_for_drawing(self):
        """Pause the simulation while drawing an obstacle."""
        self.paused = True
        self.timer.stop()

    def _create_temporary_obstacle(self):
        """Create a temporary obstacle circle for visualization during drawing."""
        # Select color based on current obstacle mode
        obstacle_color = self._get_obstacle_color()

        # Create initial circle with 0 radius
        self.temp_circle = plt.Circle(
            self.obstacle_start, 0, color=obstacle_color, alpha=0.3
        )
        self.axs[0, 0].add_artist(self.temp_circle)
        self.canvas.draw_idle()

    def _get_obstacle_color(self):
        """Get the appropriate color for the current obstacle mode."""
        if config.OBSTACLE_MODE == config.ObstacleMode.LOW_POWER_JAMMING:
            return "yellow"
        elif config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING:
            return "red"
        return "gray"  # Default for hard obstacles

    def on_drag(self, event):
        """Handle mouse drag events for drawing obstacles"""
        if not self.drawing_obstacle or not event.inaxes:
            return

        # Calculate radius from drag distance
        radius = self._calculate_drag_radius(event)

        # Update circle radius
        if self.temp_circle is not None:
            self.temp_circle.set_radius(radius)
            self.canvas.draw_idle()  # Use draw_idle for better performance during drag

    def _calculate_drag_radius(self, event):
        """Calculate the radius based on drag distance from start point."""
        return np.sqrt(
            (event.xdata - self.obstacle_start[0]) ** 2
            + (event.ydata - self.obstacle_start[1]) ** 2
        )

    def on_release(self, event):
        """Handle mouse release events for placing obstacles"""
        if not self.drawing_obstacle or not event.inaxes:
            return

        # Calculate final radius
        radius = self._calculate_drag_radius(event)

        # Add the permanent obstacle
        self._add_permanent_obstacle(radius)

        # Resume the simulation
        self._resume_after_drawing()

    def _add_permanent_obstacle(self, radius):
        """Add a permanent obstacle to the swarm state."""
        # Add obstacle to the swarm state
        self.swarm_state.add_obstacle(
            self.obstacle_start[0], self.obstacle_start[1], radius
        )

        # Clean up temporary drawing
        self._cleanup_temporary_obstacle()

        # Update plot with new obstacle
        self.update_plot()

    def _cleanup_temporary_obstacle(self):
        """Clean up the temporary obstacle circle used for drawing."""
        self.drawing_obstacle = False
        self.obstacle_start = None
        if self.temp_circle is not None:
            self.temp_circle.remove()
            self.temp_circle = None

    def _resume_after_drawing(self):
        """Resume the simulation after drawing an obstacle."""
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

        # Update status bar colors
        self._update_status_bar_colors(simulation_status, obstacle_mode)

    def _update_status_bar_colors(self, simulation_status, obstacle_mode):
        """Update the status bar colors based on the current status and mode."""
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
        llm_controller = self._get_llm_controller()
        if not llm_controller:
            return

        # Get latest feedback
        current_feedback = llm_controller.get_last_feedback()

        # Update log file path display if available
        if hasattr(llm_controller, "log_file_path"):
            log_path = llm_controller.log_file_path
            if log_path and os.path.exists(log_path):
                # Show truncated path to fit in UI
                truncated_path = log_path
                if len(log_path) > 40:
                    truncated_path = f"...{log_path[-40:]}"
                self.log_file_label.setText(f"LLM Log: {truncated_path}")
                self.open_log_button.setEnabled(True)

        # Check if there's new feedback to display
        if not current_feedback or current_feedback == self.llm_feedback_label.text():
            return

        # Handle the new feedback
        self._update_feedback_display(current_feedback, llm_controller)

    def _update_feedback_display(self, current_feedback, llm_controller):
        """Update the feedback display with new feedback."""
        # Store previous feedback before updating
        self._save_previous_feedback()

        # Update current feedback
        self.llm_feedback_label.setText(current_feedback)

        # Update timestamp
        self._update_feedback_timestamp()

        # Update perceived state information
        self._update_perceived_state(llm_controller)

    def _save_previous_feedback(self):
        """Save the current feedback as previous feedback."""
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

    def _update_feedback_timestamp(self):
        """Update the feedback timestamp."""
        current_time = time.strftime("%H:%M:%S", time.localtime())
        self.feedback_timestamp.setText(
            f"Updated at {current_time} (Iteration: {self.swarm_state.iteration})"
        )

    def _update_perceived_state(self, llm_controller):
        """Update the perceived state display."""
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
            self._generate_fallback_state_description()

    def _generate_fallback_state_description(self):
        """Generate a simplified state description as fallback."""
        state_desc = []
        state_desc.append(
            f"Destination: [{self.swarm_state.swarm_destination[0]:.1f}, {self.swarm_state.swarm_destination[1]:.1f}]"
        )

        for i in range(self.swarm_state.swarm_size):
            pos = self.swarm_state.swarm_position[i]
            conn_count = sum(self.swarm_state.neighbor_agent_matrix[i, :] > config.PT)
            state_desc.append(
                f"Agent-{i} at [{pos[0]:.1f}, {pos[1]:.1f}] with {conn_count} connections"
            )

        self.perceived_state_label.setText("\n".join(state_desc))

    def _format_state_for_display(self, state_description):
        """Format the state description for better display in the UI"""
        lines = state_description.split("\n")
        formatted_lines = []

        # Convert NODE_COLORS to hex for HTML use
        node_colors_hex = self._convert_node_colors_to_hex()

        # Track current agent for coloring
        current_agent_idx = -1

        for line in lines:
            # Skip empty lines
            if not line.strip():
                formatted_lines.append("<br>")
                continue

            # Process line based on content type
            if "destination" in line.lower():
                formatted_lines.append(self._format_destination_line(line))
            elif "obstacle" in line.lower():
                formatted_lines.append(self._format_obstacle_line(line))
            elif any(f"Agent-{i}" in line for i in range(10)):
                formatted_line, current_agent_idx = self._format_agent_line(
                    line, node_colors_hex
                )
                formatted_lines.append(formatted_line)
            elif line.strip().startswith("  - Agent-"):
                formatted_lines.append(
                    self._format_connection_line(line, node_colors_hex)
                )
            elif "mission" in line.lower():
                formatted_lines.append(self._format_mission_line(line))
            else:
                formatted_lines.append(line)

        return "<br>".join(formatted_lines)

    def _convert_node_colors_to_hex(self):
        """Convert NODE_COLORS to hex format for HTML use."""
        node_colors_hex = []
        for color in config.NODE_COLORS:
            r, g, b = color
            hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            node_colors_hex.append(hex_color)
        return node_colors_hex

    def _format_destination_line(self, line):
        """Format a destination information line."""
        return f"<span style='color:#006699; font-weight:bold; background-color: rgba(0, 102, 153, 0.2);'>{line}</span>"

    def _format_obstacle_line(self, line):
        """Format an obstacle information line."""
        return f"<span style='color:#994400; font-weight:bold; background-color: rgba(153, 68, 0, 0.2);'>{line}</span>"

    def _format_agent_line(self, line, node_colors_hex):
        """Format an agent information line."""
        # Identify which agent this is to use the right color
        current_agent_idx = -1
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
            return colored_line, current_agent_idx

        return line, current_agent_idx

    def _format_connection_line(self, line, node_colors_hex):
        """Format an agent connection line."""
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
            if "poor communication quality" in colored_line:
                # Highlight poor communication quality in red
                colored_line = colored_line.replace(
                    "poor communication quality",
                    "<span style='color:#cc0000; font-weight:bold;'>poor communication quality</span>",
                )
            elif "good communication quality" in colored_line:
                # Highlight good communication quality in green
                colored_line = colored_line.replace(
                    "good communication quality",
                    "<span style='color:#006600; font-weight:bold;'>good communication quality</span>",
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
            return f"&nbsp;&nbsp;• {colored_line[4:]}"

        return f"&nbsp;&nbsp;• {line[4:]}"

    def _format_mission_line(self, line):
        """Format a mission information line."""
        return f"<span style='color:#663399; font-weight:bold;'>{line}</span>"

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
