"""
Main entry point for the formation control simulation.
"""

import matplotlib

matplotlib.use("QtAgg")
import sys

from PyQt5.QtWidgets import QApplication

from swarm_squad.gui.formation_control_gui import FormationControlGUI


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    gui = FormationControlGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
