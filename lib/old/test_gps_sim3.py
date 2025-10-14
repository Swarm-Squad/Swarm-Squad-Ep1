#!/usr/bin/env python3
"""
Test script for GPS Simulation 3 with Line-of-Sight Obstacles
Tests both GPS-denied areas and physical obstacles that block communication
"""

import sys

from main_gps_sim3 import GPSRTKSimulation
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


def main():
    """Test the enhanced GPS simulation with line-of-sight obstacles"""
    app = QApplication(sys.argv)

    # Create simulation window
    simulation = GPSRTKSimulation()
    simulation.show()

    # Auto-start the simulation after 2 seconds for demo
    def auto_start():
        simulation.start_simulation()

        # Add test scenarios after 3 seconds
        def add_test_scenarios():
            # Scenario: Physical obstacles that block communication
            simulation.physical_obstacles = [
                (35, 35, 4),  # Small obstacle blocking some paths
                (55, 55, 6),  # Medium obstacle in center
                (75, 25, 5),  # Obstacle near Rover-2's path
                (25, 75, 5),  # Obstacle near Rover-3's path
                (60, 40, 4),  # Small obstacle creating line-of-sight issues
            ]

            # Scenario: GPS-denied areas (separate from physical obstacles)
            simulation.gps_denied_areas = [
                (45, 65, 8),  # GPS-denied area (doesn't block radio)
                (80, 80, 6),  # GPS-denied area near destination
            ]

            # Set the drawing mode to show obstacle functionality
            simulation.obstacle_mode_radio.setChecked(True)
            simulation.drawing_mode = "obstacle"

            simulation.update_plot()

            # Print test scenario info
            print("\n" + "=" * 60)
            print("GPS SIMULATION 3 - LINE-OF-SIGHT OBSTACLE TEST")
            print("=" * 60)
            print("🔴 RED AREAS: GPS-denied zones (satellite blocking)")
            print("🔘 GREY CIRCLES: Physical obstacles (radio blocking)")
            print("🔗 GREEN LINES: Active communication links")
            print("🔗 RED DASHED: Blocked communication links")
            print("\nTest Scenario:")
            print("- Physical obstacles block radio communication")
            print("- GPS-denied areas block satellite signals")
            print("- Rovers must navigate around both types of obstacles")
            print("- RTK accuracy degrades when communication is blocked")
            print("- A* replanning considers all obstacle types")
            print("\nControls:")
            print("- Toggle between GPS Areas and Physical Obstacles modes")
            print("- Click and drag to add obstacles/areas")
            print("- Watch rovers adapt to communication blocking")
            print("=" * 60)

        QTimer.singleShot(1000, add_test_scenarios)

    QTimer.singleShot(2000, auto_start)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
