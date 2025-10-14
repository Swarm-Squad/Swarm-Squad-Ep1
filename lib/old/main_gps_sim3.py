"""
Enhanced GPS-Denied Multi-Vehicle RTK Simulation with Line-of-Sight Obstacles
Implements waypoint-based trajectory following, RTK corrections, A* path planning, and physical obstacles
"""

import matplotlib

matplotlib.use("Qt5Agg")

import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from utils_gps_sim3 import (
    BaseStation,
    Grid,
    Rover,
    astar_search,
    generate_satellite_constellation,
)


def show_demo_info():
    """Show information about the demo"""
    from PyQt5.QtWidgets import QMessageBox

    msg = QMessageBox()
    msg.setWindowTitle("GPS-Denied RTK Simulation with Physical Obstacles")
    msg.setText("Enhanced Multi-Vehicle RTK Simulation with Line-of-Sight Blocking")
    msg.setInformativeText("""
🚀 NEW FEATURES IN GPS SIM 3:

🔗 LINE-OF-SIGHT COMMUNICATION:
   • Physical obstacles block radio communication
   • Grey circles represent solid obstacles
   • Communication links cut when obstructed
   • Realistic signal propagation simulation

🎯 ENHANCED NAVIGATION:
   • Same waypoint navigation and A* pathfinding
   • Obstacles affect both communication AND movement
   • Dynamic replanning around obstacles

📡 IMPROVED RTK CORRECTIONS:
   • Base stations can't transmit through obstacles
   • Rovers lose RTK accuracy when blocked
   • Position degradation shown in real-time

🎮 DUAL DRAWING MODES:
   • RED AREAS: GPS-denied zones (satellite blocking)
   • GREY OBSTACLES: Physical structures (radio blocking)
   • Toggle between modes using radio buttons

🔍 VISUAL ENHANCEMENTS:
   • Blocked communication links shown as red dashed lines
   • Active links shown as green solid lines
   • Enhanced rover status with communication quality

Instructions:
1. Select drawing mode: GPS Areas or Physical Obstacles
2. Click and drag to create areas/obstacles
3. Watch communication links get blocked by obstacles
4. Observe rovers navigate around both types of obstacles
5. Monitor RTK accuracy degradation when isolated

Key Differences from GPS Sim 2:
• Physical obstacles are GREY (not red)
• Line-of-sight blocking affects communication
• Dual obstacle types with different effects
    """)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.setDefaultButton(QMessageBox.Ok)
    return msg.exec_()


class GPSRTKSimulation(QMainWindow):
    """Main simulation window with enhanced obstacles and line-of-sight capabilities"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "GPS-Denied Multi-Vehicle RTK Simulation with Line-of-Sight Obstacles"
        )
        self.setGeometry(100, 100, 1760, 800)

        # Simulation parameters
        self.simulation_bounds = (0, 100, 0, 100)
        self.grid_size = 1.0
        self.running = False
        self.paused = False
        self.simulation_time = 0.0
        self.dt = 0.1

        # Create grid for A* pathfinding
        self.grid = Grid(
            int(self.simulation_bounds[1] / self.grid_size),
            int(self.simulation_bounds[3] / self.grid_size),
            self.grid_size,
        )

        # GPS-denied areas (RED) and physical obstacles (GREY)
        self.gps_denied_areas = []
        self.physical_obstacles = []  # NEW: Physical obstacles that block communication
        self.gps_denied_areas_history = []
        self.physical_obstacles_history = []  # NEW: History for undo

        # Drawing state
        self.drawing_mode = "gps"  # "gps" or "obstacle"
        self.drawing_state = {"drawing": False, "start": None, "temp_circle": None}

        # Initialize satellite constellation
        self.satellite_positions = generate_satellite_constellation()

        # Initialize base stations with obstacle awareness
        self.base_stations = [
            BaseStation(
                (10, 10), comm_range=25, noise_std=0.5, packet_loss=0.1, obstacles=[]
            ),
            BaseStation(
                (90, 90), comm_range=25, noise_std=0.5, packet_loss=0.1, obstacles=[]
            ),
            BaseStation(
                (10, 90), comm_range=25, noise_std=0.5, packet_loss=0.1, obstacles=[]
            ),
        ]

        # Initialize rovers with different mission waypoints and obstacle awareness
        initial_positions = [(15, 15), (85, 15), (15, 85), (85, 85)]
        waypoint_sets = [
            [(25, 25), (50, 30), (75, 50), (80, 80)],  # Rover 1
            [(75, 25), (50, 40), (25, 60), (20, 80)],  # Rover 2
            [(25, 75), (40, 50), (65, 30), (80, 20)],  # Rover 3
            [(75, 75), (60, 60), (35, 40), (20, 20)],  # Rover 4
        ]

        self.rovers = []
        for i, (pos, waypoints) in enumerate(zip(initial_positions, waypoint_sets)):
            rover = Rover(
                agent_id=f"Rover-{i + 1}",
                initial_position=pos,
                waypoints=waypoints,
                color=["red", "blue", "green", "orange"][i],
                comm_range=20,
                packet_loss=0.05,
                obstacles=[],
            )
            self.rovers.append(rover)

            # Connect base station signals to rovers
            for base_station in self.base_stations:
                base_station.correctionsReady.connect(rover.on_corrections_ready)

        # Performance tracking
        self.performance_data = {
            "time": [],
            "rover_errors": [[] for _ in self.rovers],
            "communication_status": [[] for _ in self.rovers],
            "waypoint_progress": [[] for _ in self.rovers],
        }

        # Setup UI
        self.setup_ui()

        # Timer for simulation
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)

        # Initial plot
        self.update_plot()

        # Show demo info on startup
        QTimer.singleShot(500, show_demo_info)

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        # Create matplotlib figure
        self.fig, (self.ax_main, self.ax_error) = plt.subplots(
            2, 1, figsize=(12, 7), height_ratios=[3, 1]
        )
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        # Connect mouse events for drawing
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Control buttons
        self.create_control_buttons(plot_layout)

        main_layout.addWidget(plot_widget, 3)

        # Control panel
        self.create_control_panel(main_layout)

    def create_control_buttons(self, layout):
        """Create simulation control buttons"""
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)

        button_style = """
            QPushButton {
                min-width: 90px;
                min-height: 35px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 5px;
                border: 2px solid #333;
                margin: 1px;
                padding: 2px 6px;
            }
        """

        # Start button
        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet(
            button_style + "QPushButton { background-color: #4CAF50; color: white; }"
        )
        self.start_button.clicked.connect(self.start_simulation)

        # Pause button
        self.pause_button = QPushButton("Pause")
        self.pause_button.setStyleSheet(
            button_style + "QPushButton { background-color: #FF9800; color: white; }"
        )
        self.pause_button.clicked.connect(self.pause_simulation)

        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(
            button_style + "QPushButton { background-color: #f44336; color: white; }"
        )
        self.stop_button.clicked.connect(self.stop_application)

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet(
            button_style + "QPushButton { background-color: #795548; color: white; }"
        )
        self.reset_button.clicked.connect(self.reset_simulation)

        # Undo button
        self.undo_button = QPushButton("Undo")
        self.undo_button.setStyleSheet(
            button_style + "QPushButton { background-color: #9C27B0; color: white; }"
        )
        self.undo_button.clicked.connect(self.undo_last_area)

        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.setStyleSheet(
            button_style + "QPushButton { background-color: #607D8B; color: white; }"
        )
        self.clear_button.clicked.connect(self.clear_areas)

        # Help button
        self.demo_button = QPushButton("Help")
        self.demo_button.setStyleSheet(
            button_style + "QPushButton { background-color: #2196F3; color: white; }"
        )
        self.demo_button.clicked.connect(show_demo_info)

        control_layout.addStretch()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.undo_button)
        control_layout.addWidget(self.clear_button)
        control_layout.addWidget(self.demo_button)
        control_layout.addStretch()

        layout.addWidget(control_widget)

    def create_control_panel(self, main_layout):
        """Create the control panel with parameters and status"""
        control_panel = QWidget()
        control_panel.setFixedWidth(480)
        control_panel.setStyleSheet("""
            QWidget { 
                background-color: #2b2b2b; 
                color: #ffffff;
            }
        """)
        panel_layout = QVBoxLayout(control_panel)

        # Drawing mode selection (NEW)
        mode_group = QGroupBox("Drawing Mode")
        mode_group.setStyleSheet("""
            QGroupBox { 
                font-size: 13px; 
                font-weight: bold; 
                padding: 8px;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 6px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        mode_layout = QVBoxLayout(mode_group)

        self.mode_button_group = QButtonGroup()
        self.gps_mode_radio = QRadioButton("GPS-Denied Areas (Red)")
        self.obstacle_mode_radio = QRadioButton("Physical Obstacles (Grey)")
        self.gps_mode_radio.setChecked(True)  # Default to GPS mode

        self.gps_mode_radio.setStyleSheet(
            "color: #ff6b6b; font-weight: bold; font-size: 11px;"
        )
        self.obstacle_mode_radio.setStyleSheet(
            "color: #95a5a6; font-weight: bold; font-size: 11px;"
        )

        self.mode_button_group.addButton(self.gps_mode_radio, 0)
        self.mode_button_group.addButton(self.obstacle_mode_radio, 1)
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)

        mode_layout.addWidget(self.gps_mode_radio)
        mode_layout.addWidget(self.obstacle_mode_radio)

        panel_layout.addWidget(mode_group)

        # Simulation parameters
        param_group = QGroupBox("Simulation Parameters")
        param_group.setStyleSheet("""
            QGroupBox { 
                font-size: 13px; 
                font-weight: bold; 
                padding: 8px;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 6px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        param_layout = QGridLayout(param_group)
        param_layout.setSpacing(8)

        # Base station range
        base_range_label = QLabel("Base Station Range:")
        base_range_label.setStyleSheet(
            "font-size: 11px; font-weight: 500; color: #cccccc;"
        )
        param_layout.addWidget(base_range_label, 0, 0)
        self.base_range_slider = QSlider(Qt.Horizontal)
        self.base_range_slider.setRange(10, 100)
        self.base_range_slider.setValue(25)
        self.base_range_slider.valueChanged.connect(self.update_base_station_range)
        param_layout.addWidget(self.base_range_slider, 0, 1)
        self.base_range_label = QLabel("25m")
        self.base_range_label.setStyleSheet(
            "font-size: 11px; font-weight: bold; min-width: 40px; color: #4CAF50;"
        )
        param_layout.addWidget(self.base_range_label, 0, 2)

        # Noise level
        noise_label = QLabel("Measurement Noise:")
        noise_label.setStyleSheet("font-size: 11px; font-weight: 500; color: #cccccc;")
        param_layout.addWidget(noise_label, 1, 0)
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(1, 20)
        self.noise_slider.setValue(5)
        self.noise_slider.valueChanged.connect(self.update_noise_level)
        param_layout.addWidget(self.noise_slider, 1, 1)
        self.noise_label = QLabel("0.5m")
        self.noise_label.setStyleSheet(
            "font-size: 11px; font-weight: bold; min-width: 40px; color: #FF9800;"
        )
        param_layout.addWidget(self.noise_label, 1, 2)

        # Packet loss
        packet_loss_label = QLabel("Packet Loss:")
        packet_loss_label.setStyleSheet(
            "font-size: 11px; font-weight: 500; color: #cccccc;"
        )
        param_layout.addWidget(packet_loss_label, 2, 0)
        self.packet_loss_slider = QSlider(Qt.Horizontal)
        self.packet_loss_slider.setRange(0, 50)
        self.packet_loss_slider.setValue(10)
        self.packet_loss_slider.valueChanged.connect(self.update_packet_loss)
        param_layout.addWidget(self.packet_loss_slider, 2, 1)
        self.packet_loss_label = QLabel("10%")
        self.packet_loss_label.setStyleSheet(
            "font-size: 11px; font-weight: bold; min-width: 40px; color: #f44336;"
        )
        param_layout.addWidget(self.packet_loss_label, 2, 2)

        # Replan threshold
        replan_label = QLabel("Replan Threshold:")
        replan_label.setStyleSheet("font-size: 11px; font-weight: 500; color: #cccccc;")
        param_layout.addWidget(replan_label, 3, 0)
        self.replan_spin = QSpinBox()
        self.replan_spin.setRange(1, 20)
        self.replan_spin.setValue(5)
        self.replan_spin.setStyleSheet("""
            QSpinBox { 
                font-size: 11px; 
                background-color: #404040; 
                color: #ffffff; 
                border: 1px solid #666666; 
                border-radius: 3px; 
                padding: 2px;
            }
        """)
        param_layout.addWidget(self.replan_spin, 3, 1)
        steps_label = QLabel("steps")
        steps_label.setStyleSheet(
            "font-size: 11px; font-weight: 500; min-width: 40px; color: #cccccc;"
        )
        param_layout.addWidget(steps_label, 3, 2)

        panel_layout.addWidget(param_group)

        # Rover status (2x2 grid)
        status_group = QGroupBox("Rover Status")
        status_group.setStyleSheet("""
            QGroupBox { 
                font-size: 13px; 
                font-weight: bold; 
                padding: 8px;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 6px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        status_layout = QGridLayout(status_group)
        status_layout.setSpacing(2)

        self.rover_status_labels = []
        for i, rover in enumerate(self.rovers):
            rover_widget = QWidget()
            rover_layout = QVBoxLayout(rover_widget)
            rover_layout.setContentsMargins(2, 2, 2, 2)
            rover_layout.setSpacing(1)

            # Header
            header = QLabel(f"{rover.id}")
            header.setStyleSheet(
                f"font-weight: bold; color: {rover.color}; font-size: 11px; padding: 2px; background-color: #404040; border-radius: 3px;"
            )
            rover_layout.addWidget(header)

            # Status info
            status_info = {
                "position": QLabel("Pos: (0, 0)"),
                "waypoint": QLabel("WP: 0/0"),
                "gps_status": QLabel("GPS: OK"),
                "estimate": QLabel("Est: N/A"),
                "error": QLabel("Err: 0.0m"),
                "corrections": QLabel("RTK: None"),
                "communication": QLabel("Comm: 0/3"),
                "line_of_sight": QLabel("LoS: 0/3"),  # NEW: Line of sight status
            }

            for label in status_info.values():
                label.setStyleSheet(
                    "font-size: 9px; margin: 1px; padding: 1px; color: #cccccc;"
                )
                rover_layout.addWidget(label)

            self.rover_status_labels.append(status_info)

            rover_widget.setStyleSheet(f"""
                QWidget {{ 
                    border: 2px solid {rover.color}; 
                    border-radius: 4px; 
                    background-color: #353535;
                    margin: 2px;
                    padding: 3px;
                }}
            """)

            row = i // 2
            col = i % 2
            status_layout.addWidget(rover_widget, row, col)

        panel_layout.addWidget(status_group)

        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox { 
                font-size: 13px; 
                font-weight: bold; 
                padding: 8px;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 6px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
        """)
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setSpacing(2)

        self.metrics_labels = {
            "avg_error": QLabel("Average Error: 0.0m"),
            "comm_quality": QLabel("Communication Quality: 100%"),
            "line_of_sight_quality": QLabel("Line-of-Sight Quality: 100%"),  # NEW
            "waypoints_completed": QLabel("Waypoints Completed: 0/16"),
            "simulation_time": QLabel("Simulation Time: 0.0s"),
        }

        for label in self.metrics_labels.values():
            label.setStyleSheet(
                "font-size: 11px; margin: 2px; padding: 1px; font-weight: 500; color: #e3f2fd;"
            )
            metrics_layout.addWidget(label)

        panel_layout.addWidget(metrics_group)

        panel_layout.addStretch()
        main_layout.addWidget(control_panel, 1)

    def on_mode_changed(self):
        """Handle drawing mode change"""
        if self.gps_mode_radio.isChecked():
            self.drawing_mode = "gps"
        else:
            self.drawing_mode = "obstacle"

    def update_obstacles_in_entities(self):
        """Update obstacle lists in all entities for line-of-sight calculations"""
        # Update base stations
        for base_station in self.base_stations:
            base_station.update_obstacles(self.physical_obstacles)

        # Update rovers
        for rover in self.rovers:
            rover.update_obstacles(self.physical_obstacles)

    def update_base_station_range(self, value):
        """Update base station communication range"""
        for base_station in self.base_stations:
            base_station.comm_range = value
            base_station.transceiver.comm_range = value
        self.base_range_label.setText(f"{value}m")

    def update_noise_level(self, value):
        """Update measurement noise level"""
        noise_std = value / 10.0
        for base_station in self.base_stations:
            base_station.noise_std = noise_std
        self.noise_label.setText(f"{noise_std:.1f}m")

    def update_packet_loss(self, value):
        """Update packet loss probability"""
        packet_loss = value / 100.0
        for base_station in self.base_stations:
            base_station.packet_loss = packet_loss
        for rover in self.rovers:
            rover.transceiver.packet_loss = packet_loss
        self.packet_loss_label.setText(f"{value}%")

    def simulation_step(self):
        """Single simulation step"""
        if not self.running or self.paused:
            return

        self.simulation_time += self.dt

        # Update satellite positions (they move slowly)
        for sat_pos in self.satellite_positions:
            sat_pos[0] += random.gauss(0, 0.01)
            sat_pos[1] += random.gauss(0, 0.01)

        # Update obstacles in all entities (NEW)
        self.update_obstacles_in_entities()

        # Base stations measure and broadcast corrections
        for base_station in self.base_stations:
            base_station.measure_and_broadcast(self.satellite_positions, self.rovers)

        # Update each rover
        for rover in self.rovers:
            # Check GPS availability (not in denied areas)
            rover.has_gps = not self.is_in_gps_denied_area(rover.true_pos)

            # IMU prediction step
            rover.predict_with_imu(self.dt)

            # GPS/RTK update if available
            if rover.has_gps or rover.queued_corrections:
                rover.update_with_pseudoranges_and_corrections(self.satellite_positions)

            # Check if rover needs replanning (hasn't had corrections for a while)
            if not rover.has_recent_corrections(self.replan_spin.value()):
                self.trigger_replanning(rover)

            # Pure pursuit control
            control = rover.compute_control_to_next_waypoint()

            # Update physical position
            rover.move(self.dt, control)

            # Update rover's last correction time
            rover.last_correction_time += 1

            # Update grid obstacles for A*
            self.update_grid_obstacles()

        # Store performance data
        self.store_performance_data()

        # Update displays
        self.update_rover_status()
        self.update_performance_metrics()
        self.update_plot()

    def is_in_gps_denied_area(self, position):
        """Check if position is in a GPS-denied area"""
        for center_x, center_y, radius in self.gps_denied_areas:
            dist = np.linalg.norm([position[0] - center_x, position[1] - center_y])
            if dist <= radius:
                return True
        return False

    def trigger_replanning(self, rover):
        """Trigger A* path replanning for a rover"""
        # Find nearest base station with line-of-sight
        min_dist = float("inf")
        nearest_base = None

        for base_station in self.base_stations:
            # Check both distance and line-of-sight
            if base_station.transceiver.can_communicate(
                rover.true_pos, base_station.true_pos
            ):
                dist = np.linalg.norm(rover.estimate - base_station.true_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_base = base_station

        if nearest_base:
            # Convert positions to grid coordinates
            start = self.world_to_grid(rover.estimate)
            goal = self.world_to_grid(nearest_base.true_pos)

            # Plan path using A*
            path = astar_search(self.grid, start, goal)

            if path:
                # Convert path back to world coordinates and set as waypoints
                world_path = [self.grid_to_world(grid_pos) for grid_pos in path[1:]]
                rover.set_waypoints(world_path, is_replan=True)

    def world_to_grid(self, world_pos):
        """Convert world position to grid coordinates"""
        return (int(world_pos[0] / self.grid_size), int(world_pos[1] / self.grid_size))

    def grid_to_world(self, grid_pos):
        """Convert grid position to world coordinates"""
        return np.array([grid_pos[0] * self.grid_size, grid_pos[1] * self.grid_size])

    def update_grid_obstacles(self):
        """Update grid with current obstacles and GPS-denied areas"""
        self.grid.clear_obstacles()

        # Add both GPS-denied areas and physical obstacles to the grid
        all_obstacles = self.gps_denied_areas + self.physical_obstacles
        for center_x, center_y, radius in all_obstacles:
            self.grid.add_circular_obstacle((center_x, center_y), radius)

    def store_performance_data(self):
        """Store current performance data for plotting"""
        self.performance_data["time"].append(self.simulation_time)

        for i, rover in enumerate(self.rovers):
            # Position error
            if rover.estimate is not None:
                error = np.linalg.norm(rover.true_pos - rover.estimate)
            else:
                error = float("inf")
            self.performance_data["rover_errors"][i].append(error)

            # Communication status (number of active base station links)
            comm_count = sum(
                1
                for base in self.base_stations
                if base.transceiver.can_communicate(rover.true_pos, base.true_pos)
            )
            self.performance_data["communication_status"][i].append(comm_count)

            # Waypoint progress
            progress = rover.current_waypoint_index / max(1, len(rover.waypoints))
            self.performance_data["waypoint_progress"][i].append(progress)

        # Keep only recent data
        max_points = 500
        if len(self.performance_data["time"]) > max_points:
            self.performance_data["time"] = self.performance_data["time"][-max_points:]
            for i in range(len(self.rovers)):
                self.performance_data["rover_errors"][i] = self.performance_data[
                    "rover_errors"
                ][i][-max_points:]
                self.performance_data["communication_status"][i] = (
                    self.performance_data["communication_status"][i][-max_points:]
                )
                self.performance_data["waypoint_progress"][i] = self.performance_data[
                    "waypoint_progress"
                ][i][-max_points:]

    def update_rover_status(self):
        """Update rover status display"""
        for i, rover in enumerate(self.rovers):
            labels = self.rover_status_labels[i]

            # Position
            labels["position"].setText(
                f"Pos: ({rover.true_pos[0]:.1f}, {rover.true_pos[1]:.1f})"
            )
            labels["position"].setStyleSheet(
                "font-size: 9px; margin: 1px; padding: 1px; color: #cccccc;"
            )

            # Waypoint progress
            total_waypoints = len(rover.waypoints)
            current_wp = min(rover.current_waypoint_index + 1, total_waypoints)
            if current_wp == total_waypoints:
                waypoint_color = "#4CAF50"
            elif current_wp > total_waypoints * 0.5:
                waypoint_color = "#FF9800"
            else:
                waypoint_color = "#64B5F6"
            labels["waypoint"].setText(f"WP: {current_wp}/{total_waypoints}")
            labels["waypoint"].setStyleSheet(
                f"color: {waypoint_color}; font-weight: bold; font-size: 9px; margin: 1px; padding: 1px;"
            )

            # GPS status
            gps_text = "OK" if rover.has_gps else "DENIED"
            gps_color = "#4CAF50" if rover.has_gps else "#f44336"
            labels["gps_status"].setText(f"GPS: {gps_text}")
            labels["gps_status"].setStyleSheet(
                f"color: {gps_color}; font-weight: bold; font-size: 9px; margin: 1px; padding: 1px;"
            )

            # Estimate
            if rover.estimate is not None:
                labels["estimate"].setText(
                    f"Est: ({rover.estimate[0]:.1f}, {rover.estimate[1]:.1f})"
                )
                labels["estimate"].setStyleSheet(
                    "font-size: 9px; margin: 1px; padding: 1px; color: #64B5F6;"
                )
            else:
                labels["estimate"].setText("Est: N/A")
                labels["estimate"].setStyleSheet(
                    "font-size: 9px; margin: 1px; padding: 1px; color: #888888;"
                )

            # Error
            if rover.estimate is not None:
                error = np.linalg.norm(rover.true_pos - rover.estimate)
                labels["error"].setText(f"Err: {error:.2f}m")

                if error > 5.0:
                    error_color = "#f44336"
                elif error > 2.0:
                    error_color = "#FF9800"
                else:
                    error_color = "#4CAF50"
                labels["error"].setStyleSheet(
                    f"color: {error_color}; font-weight: bold; font-size: 9px; margin: 1px; padding: 1px;"
                )
            else:
                labels["error"].setText("Err: N/A")
                labels["error"].setStyleSheet(
                    "color: #888888; font-size: 9px; margin: 1px; padding: 1px;"
                )

            # RTK corrections
            corrections_text = "OK" if rover.queued_corrections else "None"
            corrections_color = "#4CAF50" if rover.queued_corrections else "#888888"
            labels["corrections"].setText(f"RTK: {corrections_text}")
            labels["corrections"].setStyleSheet(
                f"color: {corrections_color}; font-size: 9px; margin: 1px; padding: 1px;"
            )

            # Communication (range-based)
            comm_count = sum(
                1
                for base in self.base_stations
                if base.transceiver.in_range(rover.true_pos, base.true_pos)
            )
            total_bases = len(self.base_stations)
            if comm_count >= 2:
                comm_color = "#4CAF50"
            elif comm_count == 1:
                comm_color = "#FF9800"
            else:
                comm_color = "#f44336"
            labels["communication"].setText(f"Comm: {comm_count}/{total_bases}")
            labels["communication"].setStyleSheet(
                f"color: {comm_color}; font-weight: bold; font-size: 9px; margin: 1px; padding: 1px;"
            )

            # Line of sight (NEW)
            los_count = sum(
                1
                for base in self.base_stations
                if base.transceiver.can_communicate(rover.true_pos, base.true_pos)
            )
            if los_count >= 2:
                los_color = "#4CAF50"
            elif los_count == 1:
                los_color = "#FF9800"
            else:
                los_color = "#f44336"
            labels["line_of_sight"].setText(f"LoS: {los_count}/{total_bases}")
            labels["line_of_sight"].setStyleSheet(
                f"color: {los_color}; font-weight: bold; font-size: 9px; margin: 1px; padding: 1px;"
            )

    def update_performance_metrics(self):
        """Update performance metrics display"""
        if not self.performance_data["time"]:
            return

        # Average error across all rovers
        recent_errors = []
        for rover_errors in self.performance_data["rover_errors"]:
            if rover_errors and rover_errors[-1] != float("inf"):
                recent_errors.append(rover_errors[-1])

        avg_error = np.mean(recent_errors) if recent_errors else 0
        self.metrics_labels["avg_error"].setText(f"Average Error: {avg_error:.2f}m")

        # Communication quality (range-based)
        recent_comm = []
        for comm_status in self.performance_data["communication_status"]:
            if comm_status:
                recent_comm.append(comm_status[-1])

        total_possible = len(self.rovers) * len(self.base_stations)
        total_active = sum(recent_comm) if recent_comm else 0
        comm_quality = (
            (total_active / total_possible) * 100 if total_possible > 0 else 0
        )
        self.metrics_labels["comm_quality"].setText(
            f"Communication Quality: {comm_quality:.1f}%"
        )

        # Line-of-sight quality (NEW)
        los_active = sum(
            1
            for rover in self.rovers
            for base in self.base_stations
            if base.transceiver.can_communicate(rover.true_pos, base.true_pos)
        )
        los_quality = (los_active / total_possible) * 100 if total_possible > 0 else 0
        self.metrics_labels["line_of_sight_quality"].setText(
            f"Line-of-Sight Quality: {los_quality:.1f}%"
        )

        # Waypoints completed
        total_waypoints = sum(len(rover.waypoints) for rover in self.rovers)
        completed_waypoints = sum(rover.current_waypoint_index for rover in self.rovers)
        self.metrics_labels["waypoints_completed"].setText(
            f"Waypoints Completed: {completed_waypoints}/{total_waypoints}"
        )

        # Simulation time
        self.metrics_labels["simulation_time"].setText(
            f"Simulation Time: {self.simulation_time:.1f}s"
        )

    def update_plot(self):
        """Update the main plot and error plot"""
        # Clear main plot
        self.ax_main.clear()
        self.ax_main.set_xlim(self.simulation_bounds[0], self.simulation_bounds[1])
        self.ax_main.set_ylim(self.simulation_bounds[2], self.simulation_bounds[3])
        self.ax_main.set_xlabel("X Position (m)", fontsize=11)
        self.ax_main.set_ylabel("Y Position (m)", fontsize=11)
        self.ax_main.set_title(
            "GPS-Denied Multi-Vehicle RTK Simulation with Line-of-Sight Obstacles",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect("equal")

        # Draw GPS-denied areas (RED)
        for center_x, center_y, radius in self.gps_denied_areas:
            circle = plt.Circle(
                (center_x, center_y),
                radius,
                color="red",
                alpha=0.3,
                label="GPS-Denied Area",
            )
            self.ax_main.add_artist(circle)

        # Draw physical obstacles (GREY) - NEW
        for center_x, center_y, radius in self.physical_obstacles:
            circle = plt.Circle(
                (center_x, center_y),
                radius,
                color="gray",
                alpha=0.6,
                label="Physical Obstacle",
            )
            self.ax_main.add_artist(circle)

        # Draw base stations
        for base_station in self.base_stations:
            self.ax_main.scatter(
                *base_station.true_pos,
                color="purple",
                marker="^",
                s=100,
                edgecolor="black",
                linewidth=2,
                label="Base Station",
                zorder=10,
            )

            # Draw communication range
            comm_circle = plt.Circle(
                base_station.true_pos,
                base_station.comm_range,
                color="purple",
                alpha=0.1,
                linestyle="--",
                fill=False,
            )
            self.ax_main.add_artist(comm_circle)

        # Draw satellites
        for sat_pos in self.satellite_positions[:8]:
            self.ax_main.scatter(
                sat_pos[0], sat_pos[1], color="yellow", marker="*", s=20, alpha=0.7
            )

        # Draw rovers
        for rover in self.rovers:
            # Draw trajectory
            if len(rover.trajectory) > 1:
                trajectory_array = np.array(rover.trajectory)
                self.ax_main.plot(
                    trajectory_array[:, 0],
                    trajectory_array[:, 1],
                    color=rover.color,
                    alpha=0.3,
                    linewidth=1,
                )

            # Draw waypoints
            if rover.waypoints:
                waypoint_array = np.array(rover.waypoints)
                self.ax_main.plot(
                    waypoint_array[:, 0],
                    waypoint_array[:, 1],
                    color=rover.color,
                    linestyle="--",
                    alpha=0.5,
                    linewidth=2,
                )

                # Mark waypoints
                for i, wp in enumerate(rover.waypoints):
                    if i <= rover.current_waypoint_index:
                        self.ax_main.scatter(
                            *wp, color=rover.color, marker="x", s=50, alpha=0.7
                        )
                    else:
                        self.ax_main.scatter(
                            *wp, color=rover.color, marker="o", s=30, alpha=0.5
                        )

            # Draw current position
            marker = "o" if rover.has_gps else "s"
            self.ax_main.scatter(
                *rover.true_pos,
                color=rover.color,
                marker=marker,
                s=100,
                edgecolor="black",
                linewidth=2,
                zorder=5,
            )

            # Draw estimate if available
            if rover.estimate is not None and not rover.has_gps:
                self.ax_main.scatter(
                    *rover.estimate,
                    color="lime",
                    marker="*",
                    s=150,
                    edgecolor="darkgreen",
                    linewidth=2,
                    zorder=4,
                )

                # Draw uncertainty ellipse if covariance available
                if hasattr(rover, "covariance") and rover.covariance is not None:
                    self.draw_uncertainty_ellipse(
                        rover.estimate, rover.covariance[:2, :2]
                    )

                # Draw error line
                self.ax_main.plot(
                    [rover.true_pos[0], rover.estimate[0]],
                    [rover.true_pos[1], rover.estimate[1]],
                    "r--",
                    alpha=0.5,
                    linewidth=1,
                )

            # Draw communication links with line-of-sight awareness (ENHANCED)
            for base_station in self.base_stations:
                in_range = base_station.transceiver.in_range(
                    rover.true_pos, base_station.true_pos
                )
                has_los = base_station.transceiver.can_communicate(
                    rover.true_pos, base_station.true_pos
                )

                if in_range:
                    if has_los:
                        # Active communication - GREEN solid line
                        self.ax_main.plot(
                            [rover.true_pos[0], base_station.true_pos[0]],
                            [rover.true_pos[1], base_station.true_pos[1]],
                            "g-",
                            alpha=0.7,
                            linewidth=2,
                            zorder=1,
                        )
                    else:
                        # Blocked by obstacle - RED dashed line
                        self.ax_main.plot(
                            [rover.true_pos[0], base_station.true_pos[0]],
                            [rover.true_pos[1], base_station.true_pos[1]],
                            "r--",
                            alpha=0.5,
                            linewidth=1,
                            zorder=1,
                        )

            # Label
            self.ax_main.annotate(
                rover.id,
                rover.true_pos,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=rover.color,
                fontweight="bold",
            )

        # Update error plot
        self.ax_error.clear()
        has_data = False
        if self.performance_data["time"]:
            for i, rover in enumerate(self.rovers):
                if self.performance_data["rover_errors"][i]:
                    # Filter out infinite errors for plotting
                    times = []
                    errors = []
                    for t, err in zip(
                        self.performance_data["time"],
                        self.performance_data["rover_errors"][i],
                    ):
                        if err != float("inf"):
                            times.append(t)
                            errors.append(err)

                    if times:
                        self.ax_error.plot(
                            times,
                            errors,
                            color=rover.color,
                            label=f"{rover.id} Error",
                            linewidth=2,
                        )
                        has_data = True

        self.ax_error.set_xlabel("Time (s)", fontsize=11)
        self.ax_error.set_ylabel("Position Error (m)", fontsize=11)
        self.ax_error.set_title(
            "Position Error (Estimate vs True Position)", fontsize=12, fontweight="bold"
        )
        self.ax_error.grid(True, alpha=0.3)

        if has_data:
            self.ax_error.legend(fontsize=10)

        # Draw canvas
        self.canvas.draw()

    def draw_uncertainty_ellipse(self, center, covariance, n_std=2.0):
        """Draw uncertainty ellipse for position estimate"""
        eigenvals, eigenvecs = np.linalg.eig(covariance)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigenvals)

        ellipse = plt.matplotlib.patches.Ellipse(
            center, width, height, angle=angle, color="lime", alpha=0.2, fill=True
        )
        self.ax_main.add_artist(ellipse)

    def on_click(self, event):
        """Handle mouse click to start drawing areas/obstacles"""
        if event.inaxes != self.ax_main:
            return

        self.drawing_state["drawing"] = True
        self.drawing_state["start"] = (event.xdata, event.ydata)

        # Pause simulation while drawing
        self.was_running = self.running
        if self.running:
            self.pause_simulation()

    def on_drag(self, event):
        """Handle mouse drag to show area/obstacle preview"""
        if not self.drawing_state["drawing"] or event.inaxes != self.ax_main:
            return

        start = self.drawing_state["start"]
        radius = np.linalg.norm([event.xdata - start[0], event.ydata - start[1]])

        # Remove previous preview
        if self.drawing_state["temp_circle"]:
            self.drawing_state["temp_circle"].remove()

        # Draw preview with appropriate color
        if self.drawing_mode == "gps":
            color = "red"
            alpha = 0.3
        else:  # obstacle mode
            color = "gray"
            alpha = 0.6

        self.drawing_state["temp_circle"] = plt.Circle(
            start, radius, color=color, alpha=alpha
        )
        self.ax_main.add_artist(self.drawing_state["temp_circle"])
        self.canvas.draw()

    def on_release(self, event):
        """Handle mouse release to finalize area/obstacle"""
        if not self.drawing_state["drawing"] or event.inaxes != self.ax_main:
            return

        start = self.drawing_state["start"]
        radius = np.linalg.norm([event.xdata - start[0], event.ydata - start[1]])

        # Add area/obstacle if large enough
        if radius >= 2.0:
            new_item = (start[0], start[1], radius)

            if self.drawing_mode == "gps":
                self.gps_denied_areas.append(new_item)
                self.gps_denied_areas_history.append(new_item)
            else:  # obstacle mode
                self.physical_obstacles.append(new_item)
                self.physical_obstacles_history.append(new_item)

            # Automatically start simulation when area/obstacle is added
            if not self.running:
                self.start_simulation()

        # Clean up
        self.drawing_state["drawing"] = False
        self.drawing_state["start"] = None
        if self.drawing_state["temp_circle"]:
            self.drawing_state["temp_circle"].remove()
            self.drawing_state["temp_circle"] = None

        # Update plot
        self.update_plot()

        # Resume simulation if it was running before drawing
        if hasattr(self, "was_running") and self.was_running:
            self.start_simulation()

    def start_simulation(self):
        """Start the simulation"""
        self.running = True
        self.paused = False
        self.timer.start(100)  # 10 FPS

    def pause_simulation(self):
        """Pause the simulation"""
        self.paused = True
        self.timer.stop()

    def stop_application(self):
        """Stop and close the application"""
        self.close()

    def undo_last_area(self):
        """Undo the last drawn area/obstacle"""
        if self.drawing_mode == "gps" and self.gps_denied_areas_history:
            last_area = self.gps_denied_areas_history.pop()
            if last_area in self.gps_denied_areas:
                self.gps_denied_areas.remove(last_area)
        elif self.drawing_mode == "obstacle" and self.physical_obstacles_history:
            last_obstacle = self.physical_obstacles_history.pop()
            if last_obstacle in self.physical_obstacles:
                self.physical_obstacles.remove(last_obstacle)
        self.update_plot()

    def clear_areas(self):
        """Clear all areas and obstacles"""
        self.gps_denied_areas.clear()
        self.gps_denied_areas_history.clear()
        self.physical_obstacles.clear()
        self.physical_obstacles_history.clear()
        self.update_plot()

    def reset_simulation(self):
        """Reset the entire simulation to initial state"""
        # Stop simulation
        self.running = False
        self.paused = False
        self.timer.stop()

        # Reset simulation time
        self.simulation_time = 0.0

        # Clear areas and obstacles
        self.gps_denied_areas.clear()
        self.gps_denied_areas_history.clear()
        self.physical_obstacles.clear()
        self.physical_obstacles_history.clear()

        # Reset rovers to initial state
        initial_positions = [(15, 15), (85, 15), (15, 85), (85, 85)]
        waypoint_sets = [
            [(25, 25), (50, 30), (75, 50), (80, 80)],
            [(75, 25), (50, 40), (25, 60), (20, 80)],
            [(25, 75), (40, 50), (65, 30), (80, 20)],
            [(75, 75), (60, 60), (35, 40), (20, 20)],
        ]

        for i, rover in enumerate(self.rovers):
            rover.reset(initial_positions[i], waypoint_sets[i])

        # Clear performance data
        self.performance_data = {
            "time": [],
            "rover_errors": [[] for _ in self.rovers],
            "communication_status": [[] for _ in self.rovers],
            "waypoint_progress": [[] for _ in self.rovers],
        }

        # Clear grid obstacles
        self.grid.clear_obstacles()

        # Update displays
        self.update_rover_status()
        self.update_performance_metrics()
        self.update_plot()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    window = GPSRTKSimulation()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
