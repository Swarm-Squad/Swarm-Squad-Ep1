#!/usr/bin/env python3
"""
Quick test script for GUI improvements and layout verification
"""

import sys

from main_gps_sim2 import GPSRTKSimulation
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


def main():
    """Test the improved GUI layout"""
    app = QApplication(sys.argv)
    
    # Create simulation window
    simulation = GPSRTKSimulation()
    simulation.show()
    
    # Auto-start the simulation after 2 seconds for demo
    def auto_start():
        simulation.start_simulation()
        
        # Add some test GPS-denied areas after 3 seconds
        def add_test_areas():
            simulation.gps_denied_areas = [
                (35, 35, 6),   # Small area near rover paths
                (35, 60, 7),   # Small area near rover paths
                (55, 55, 8),   # Medium area in center-right
                (58, 35, 8),   # Medium area in center-right
                (75, 55, 8),   # Medium area in center-right
            ]
            simulation.update_plot()
        
        QTimer.singleShot(0, add_test_areas)
    
    QTimer.singleShot(2000, auto_start)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 