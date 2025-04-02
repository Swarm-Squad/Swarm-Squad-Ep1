"""
Main entry point for the formation control simulation.
"""

import argparse
import logging
import sys

import matplotlib

matplotlib.use("QtAgg")

from PyQt5.QtWidgets import QApplication, QMessageBox

from swarm_squad.config import LLM_ENABLED, LLM_FEEDBACK_INTERVAL, LLM_MODEL
from swarm_squad.gui.formation_control_gui import FormationControlGUI
from swarm_squad.utils import check_ollama_running, get_ollama_api_url

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Swarm Squad Ep1: Surviving the Jam")

    parser.add_argument(
        "-m",
        "--model",
        default=LLM_MODEL,
        help=f"LLM model to use (default: {LLM_MODEL})",
    )

    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=LLM_FEEDBACK_INTERVAL,
        help=f"LLM feedback interval in simulation steps (default: {LLM_FEEDBACK_INTERVAL})",
    )

    return parser.parse_args()


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()

    # Override config values with command line arguments
    model = args.model
    interval = args.interval

    app = QApplication(sys.argv)

    # Check if Ollama is running if LLM is enabled
    if LLM_ENABLED and not check_ollama_running(model=model):
        base_url = get_ollama_api_url()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Ollama Check")
        msg.setText(f"Cannot connect to Ollama at {base_url}")
        msg.setInformativeText(
            f"LLM feedback will be disabled. Please:\n1. Make sure Ollama is running\n2. Check that model {model} is available"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # Start the GUI with custom LLM settings
    gui = FormationControlGUI(llm_model=model, llm_feedback_interval=interval)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
