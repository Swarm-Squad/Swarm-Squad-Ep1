import matplotlib

matplotlib.use("Qt5Agg")
import math
import random
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import utils_gps_sim as utils
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class Vehicle:
    """Represents a vehicle/agent in the simulation."""

    def __init__(self, agent_id: str, position: Tuple[float, float], color: str):
        self.id = agent_id
        self.position = position
        self.previous_position = position
        self.color = color
        self.has_gps = True
        self.estimated_position = None
        self.position_error = 0.0
        self.trajectory = [position]
        self.velocity = (0.0, 0.0)

        # Movement parameters
        self.speed = 1.0
        self.trajectory_angle = random.uniform(0, 2 * math.pi)
        self.trajectory_center = (50, 50)
        self.trajectory_radius = 30

    def update_position(self, bounds: Tuple[float, float, float, float]):
        """Update vehicle position with circular movement pattern."""
        self.previous_position = self.position

        # Generate circular trajectory with some randomness
        new_pos, self.trajectory_angle = utils.generate_circular_trajectory(
            self.trajectory_center,
            self.trajectory_radius,
            self.trajectory_angle,
            angular_speed=0.02 + random.uniform(-0.01, 0.01),
        )

        # Add some random noise to make movement more realistic
        noise_x = random.gauss(0, 0.3)
        noise_y = random.gauss(0, 0.3)
        new_pos = (new_pos[0] + noise_x, new_pos[1] + noise_y)

        # Keep within bounds
        self.position = utils.bound_position(new_pos, bounds)

        # Calculate velocity
        dt = 0.1  # Simulation time step
        self.velocity = utils.estimate_velocity(
            self.previous_position, self.position, dt
        )

        # Store trajectory
        self.trajectory.append(self.position)
        if len(self.trajectory) > 500:  # Limit trajectory length
            self.trajectory.pop(0)

    def lose_gps(self):
        """Simulate losing GPS signal."""
        self.has_gps = False
        self.estimated_position = None
        self.position_error = float("inf")

    def regain_gps(self):
        """Simulate regaining GPS signal."""
        self.has_gps = True
        self.estimated_position = None
        self.position_error = 0.0

    def estimate_position_via_trilateration(self, anchor_vehicles: List["Vehicle"]):
        """Estimate position using trilateration from anchor vehicles."""
        if len(anchor_vehicles) < 3:
            self.estimated_position = None
            self.position_error = float("inf")
            return

        # Get positions and distances from anchors
        anchors = []
        distances = []

        for anchor in anchor_vehicles:
            if anchor.has_gps and anchor.id != self.id:
                anchors.append(anchor.position)
                # Simulate distance measurement with noise
                measured_distance = utils.simulate_distance_measurement(
                    self.position, anchor.position, noise_std=0.2
                )
                distances.append(measured_distance)

        if len(anchors) >= 3:
            try:
                # Use multilateration for robust estimation
                self.estimated_position = utils.multilaterate_2d(anchors, distances)
                self.position_error = utils.calculate_position_error(
                    self.position, self.estimated_position
                )
            except (ValueError, np.linalg.LinAlgError):
                self.estimated_position = None
                self.position_error = float("inf")
        else:
            self.estimated_position = None
            self.position_error = float("inf")


class GPSLocalizationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPS Localization Simulation - RTK Trilateration")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)

        # Initialize simulation parameters first
        self.simulation_bounds = (0, 100, 0, 100)
        self.gps_denied_areas = []  # List of (center_x, center_y, radius)

        # Drawing state variables
        self.drawing_area = False
        self.area_start = None
        self.temp_circle = None

        # Initialize vehicles BEFORE creating info panel
        self.vehicles = [
            Vehicle("Vehicle-1", (20, 30), "red"),
            Vehicle("Vehicle-2", (70, 20), "blue"),
            Vehicle("Vehicle-3", (50, 70), "green"),
            Vehicle("Vehicle-4", (30, 80), "orange"),
        ]

        # Simulation control
        self.running = False
        self.paused = False
        self.simulation_time = 0.0

        # Create plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        # Connect mouse events
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Create control buttons
        self.create_plot_controls(plot_layout)

        # Add plot widget to main layout
        main_layout.addWidget(plot_widget)

        # Create info panel (now vehicles are defined)
        self.create_info_panel(main_layout)

        # Create timer for simulation
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)

        # Initialize plot
        self.update_plot()

    def create_plot_controls(self, layout):
        """Create control buttons for the simulation."""
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)

        button_style = """
            QPushButton {
                min-width: 80px;
                min-height: 35px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 5px;
                border: 2px solid #333;
                color: #000000;
            }
        """

        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet(
            button_style + "QPushButton { background-color: #e3f0d8; color: #2d5016; }"
        )
        self.start_button.clicked.connect(self.start_simulation)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setStyleSheet(
            button_style + "QPushButton { background-color: #fdf2ca; color: #8b4513; }"
        )
        self.pause_button.clicked.connect(self.pause_simulation)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet(
            button_style + "QPushButton { background-color: #d8e3f0; color: #1e3a8a; }"
        )
        self.reset_button.clicked.connect(self.reset_simulation)

        self.clear_areas_button = QPushButton("Clear Areas")
        self.clear_areas_button.setStyleSheet(
            button_style + "QPushButton { background-color: #f9aeae; color: #7f1d1d; }"
        )
        self.clear_areas_button.clicked.connect(self.clear_gps_denied_areas)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.clear_areas_button)

        layout.addWidget(control_widget)

    def create_info_panel(self, main_layout):
        """Create information panel showing vehicle status."""
        info_widget = QWidget()
        info_widget.setFixedWidth(350)
        info_layout = QVBoxLayout(info_widget)

        # Title
        title_label = QLabel("Vehicle Status")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        info_layout.addWidget(title_label)

        # Instructions
        instructions = QLabel("""
        Instructions:
        • Click and drag to draw GPS-denied areas
        • Vehicles lose GPS when entering red areas
        • Green circles show trilateration estimates
        • Start simulation to see real-time tracking
        """)
        instructions.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #666;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
                margin-bottom: 15px;
            }
        """)
        instructions.setWordWrap(True)
        info_layout.addWidget(instructions)

        # Vehicle info labels
        self.vehicle_info_labels = {}
        for vehicle in self.vehicles:
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box)
            frame.setLineWidth(1)
            frame_layout = QVBoxLayout(frame)

            # Vehicle header
            header = QLabel(f"{vehicle.id}")
            header.setStyleSheet(f"""
                QLabel {{
                    font-size: 14px;
                    font-weight: bold;
                    color: {vehicle.color};
                    padding: 5px;
                }}
            """)
            frame_layout.addWidget(header)

            # Status labels
            gps_label = QLabel("GPS: Available")
            pos_label = QLabel("Position: (0.0, 0.0)")
            est_label = QLabel("Estimated: N/A")
            error_label = QLabel("Error: 0.0m")

            for label in [gps_label, pos_label, est_label, error_label]:
                label.setStyleSheet("""
                    QLabel {
                        font-size: 10px;
                        padding: 2px;
                        color: #444;
                    }
                """)
                frame_layout.addWidget(label)

            self.vehicle_info_labels[vehicle.id] = {
                "gps": gps_label,
                "position": pos_label,
                "estimated": est_label,
                "error": error_label,
            }

            info_layout.addWidget(frame)

        # Simulation status
        self.sim_status_label = QLabel("Simulation: Stopped")
        self.sim_status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #666;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
                margin-top: 15px;
            }
        """)
        info_layout.addWidget(self.sim_status_label)

        info_layout.addStretch()
        main_layout.addWidget(info_widget)

    def update_info_panel(self):
        """Update the information panel with current vehicle status."""
        for vehicle in self.vehicles:
            labels = self.vehicle_info_labels[vehicle.id]

            # GPS status
            gps_status = "Available" if vehicle.has_gps else "DENIED"
            gps_color = "green" if vehicle.has_gps else "red"
            labels["gps"].setText(f"GPS: {gps_status}")
            labels["gps"].setStyleSheet(f"""
                QLabel {{
                    font-size: 10px;
                    padding: 2px;
                    color: {gps_color};
                    font-weight: bold;
                }}
            """)

            # Position
            pos_text = utils.format_coordinates(vehicle.position)
            labels["position"].setText(f"True Pos: {pos_text}")

            # Estimated position
            if vehicle.estimated_position:
                est_text = utils.format_coordinates(vehicle.estimated_position)
                labels["estimated"].setText(f"Estimated: {est_text}")
            else:
                labels["estimated"].setText("Estimated: N/A")

            # Error
            if vehicle.position_error != float("inf"):
                labels["error"].setText(f"Error: {vehicle.position_error:.2f}m")
                error_color = (
                    "red"
                    if vehicle.position_error > 2.0
                    else "orange"
                    if vehicle.position_error > 1.0
                    else "green"
                )
                labels["error"].setStyleSheet(f"""
                    QLabel {{
                        font-size: 10px;
                        padding: 2px;
                        color: {error_color};
                        font-weight: bold;
                    }}
                """)
            else:
                labels["error"].setText("Error: N/A")
                labels["error"].setStyleSheet("""
                    QLabel {
                        font-size: 10px;
                        padding: 2px;
                        color: #666;
                    }
                """)

        # Simulation status
        if self.running:
            status = f"Running (t={self.simulation_time:.1f}s)"
            color = "green"
        elif self.paused:
            status = "Paused"
            color = "orange"
        else:
            status = "Stopped"
            color = "red"

        self.sim_status_label.setText(f"Simulation: {status}")
        self.sim_status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 12px;
                font-weight: bold;
                color: {color};
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
                margin-top: 15px;
            }}
        """)

    def simulation_step(self):
        """Single step of the simulation."""
        if not self.running or self.paused:
            return

        # Update simulation time
        self.simulation_time += 0.1

        # Update vehicle positions
        for vehicle in self.vehicles:
            vehicle.update_position(self.simulation_bounds)

            # Check if vehicle is in GPS-denied area
            in_denied_area = False
            for area_center_x, area_center_y, radius in self.gps_denied_areas:
                if utils.is_point_in_circle(
                    vehicle.position, (area_center_x, area_center_y), radius
                ):
                    in_denied_area = True
                    break

            # Update GPS status
            if in_denied_area and vehicle.has_gps:
                vehicle.lose_gps()
            elif not in_denied_area and not vehicle.has_gps:
                vehicle.regain_gps()

            # If vehicle doesn't have GPS, try trilateration
            if not vehicle.has_gps:
                anchor_vehicles = [
                    v for v in self.vehicles if v.has_gps and v.id != vehicle.id
                ]
                vehicle.estimate_position_via_trilateration(anchor_vehicles)

        # Update display
        self.update_plot()
        self.update_info_panel()

    def update_plot(self):
        """Update the matplotlib plot."""
        self.ax.clear()

        # Set up plot
        self.ax.set_xlim(self.simulation_bounds[0], self.simulation_bounds[1])
        self.ax.set_ylim(self.simulation_bounds[2], self.simulation_bounds[3])
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.set_title("GPS Localization Simulation with Trilateration")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")

        # Draw GPS-denied areas
        for area_center_x, area_center_y, radius in self.gps_denied_areas:
            circle = plt.Circle(
                (area_center_x, area_center_y),
                radius,
                color="red",
                alpha=0.3,
                label="GPS-Denied Area",
            )
            self.ax.add_artist(circle)

        # Draw vehicles
        for vehicle in self.vehicles:
            # Draw trajectory
            if len(vehicle.trajectory) > 1:
                trajectory_array = np.array(vehicle.trajectory)
                self.ax.plot(
                    trajectory_array[:, 0],
                    trajectory_array[:, 1],
                    color=vehicle.color,
                    alpha=0.3,
                    linewidth=1,
                )

            # Draw vehicle current position
            marker_style = "o" if vehicle.has_gps else "s"
            marker_size = 8 if vehicle.has_gps else 10
            self.ax.scatter(
                *vehicle.position,
                color=vehicle.color,
                marker=marker_style,
                s=marker_size**2,
                edgecolor="black",
                linewidth=2,
                zorder=5,
            )

            # Draw estimated position if available
            if vehicle.estimated_position and not vehicle.has_gps:
                self.ax.scatter(
                    *vehicle.estimated_position,
                    color="lime",
                    marker="*",
                    s=100,
                    edgecolor="darkgreen",
                    linewidth=2,
                    zorder=4,
                    label="Trilateration Estimate",
                )

                # Draw error circle
                error_circle = plt.Circle(
                    vehicle.estimated_position,
                    vehicle.position_error,
                    color="lime",
                    alpha=0.2,
                    fill=True,
                )
                self.ax.add_artist(error_circle)

                # Draw line between true and estimated position
                self.ax.plot(
                    [vehicle.position[0], vehicle.estimated_position[0]],
                    [vehicle.position[1], vehicle.estimated_position[1]],
                    "r--",
                    alpha=0.7,
                    linewidth=1,
                )

            # Add vehicle label
            self.ax.annotate(
                vehicle.id,
                vehicle.position,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=vehicle.color,
                fontweight="bold",
            )

        # Add legend
        legend_elements = []
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="GPS Available",
                markeredgecolor="black",
            )
        )
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="GPS Denied",
                markeredgecolor="black",
            )
        )
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="lime",
                markersize=10,
                label="Trilateration Est.",
                markeredgecolor="darkgreen",
            )
        )

        self.ax.legend(handles=legend_elements, loc="upper right")

        # Draw canvas
        self.canvas.draw()

    def on_click(self, event):
        """Handle mouse click to start drawing GPS-denied area."""
        if event.inaxes != self.ax:
            return

        self.drawing_area = True
        self.area_start = (event.xdata, event.ydata)

        # Pause simulation while drawing
        if self.running:
            self.pause_simulation()

    def on_drag(self, event):
        """Handle mouse drag to show GPS-denied area preview."""
        if not self.drawing_area or event.inaxes != self.ax:
            return

        # Calculate radius
        radius = utils.euclidean_distance(self.area_start, (event.xdata, event.ydata))

        # Remove previous temporary circle
        if self.temp_circle is not None:
            self.temp_circle.remove()

        # Draw new temporary circle
        self.temp_circle = plt.Circle(self.area_start, radius, color="red", alpha=0.3)
        self.ax.add_artist(self.temp_circle)
        self.canvas.draw()

    def on_release(self, event):
        """Handle mouse release to finalize GPS-denied area."""
        if not self.drawing_area or event.inaxes != self.ax:
            return

        # Calculate final radius
        radius = utils.euclidean_distance(self.area_start, (event.xdata, event.ydata))

        # Add GPS-denied area (minimum radius of 2)
        if radius >= 2.0:
            self.gps_denied_areas.append(
                (self.area_start[0], self.area_start[1], radius)
            )

        # Clean up
        self.drawing_area = False
        self.area_start = None
        if self.temp_circle is not None:
            self.temp_circle.remove()
            self.temp_circle = None

        # Update plot
        self.update_plot()

    def start_simulation(self):
        """Start the simulation."""
        self.running = True
        self.paused = False
        self.timer.start(100)  # 100ms interval (10 FPS)

    def pause_simulation(self):
        """Pause the simulation."""
        self.paused = True
        self.timer.stop()

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        self.running = False
        self.paused = False
        self.simulation_time = 0.0
        self.timer.stop()

        # Reset vehicle positions and states
        initial_positions = [(20, 30), (70, 20), (50, 70), (30, 80)]
        for i, vehicle in enumerate(self.vehicles):
            vehicle.position = initial_positions[i]
            vehicle.previous_position = initial_positions[i]
            vehicle.trajectory = [initial_positions[i]]
            vehicle.has_gps = True
            vehicle.estimated_position = None
            vehicle.position_error = 0.0
            vehicle.trajectory_angle = random.uniform(0, 2 * math.pi)

        self.update_plot()
        self.update_info_panel()

    def clear_gps_denied_areas(self):
        """Clear all GPS-denied areas."""
        self.gps_denied_areas.clear()
        self.update_plot()


# Create and run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GPSLocalizationGUI()
    window.show()
    sys.exit(app.exec_())
