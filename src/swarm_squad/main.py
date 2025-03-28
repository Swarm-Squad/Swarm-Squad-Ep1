"""
Main entry point for the formation control simulation.
"""

import logging
import sys

import matplotlib

matplotlib.use("QtAgg")

from PyQt5.QtWidgets import QApplication, QMessageBox

from swarm_squad.config import LLM_ENABLED, LLM_MODEL
from swarm_squad.gui.formation_control_gui import FormationControlGUI
from swarm_squad.utils import check_ollama_running, get_ollama_api_url

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)

    # Check if Ollama is running if LLM is enabled
    if LLM_ENABLED and not check_ollama_running():
        base_url = get_ollama_api_url()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Ollama Check")
        msg.setText(f"Cannot connect to Ollama at {base_url}")
        msg.setInformativeText(
            f"LLM feedback will be disabled. Please:\n1. Make sure Ollama is running\n2. Check that model {LLM_MODEL} is available"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # Start the GUI
    gui = FormationControlGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
