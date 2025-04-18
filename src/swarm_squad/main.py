"""
Main entry point for the formation control simulation.
"""

import argparse
import logging
import os
import subprocess
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

    # GUI/Simulation options
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

    # Testing options
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Run all test modules in the tests directory",
    )

    parser.add_argument(
        "--test-ollama",
        nargs="?",
        const=LLM_MODEL,
        metavar="MODEL",
        help=f"Test Ollama connectivity with optional model name (default: {LLM_MODEL})",
    )

    return parser.parse_args()


def run_tests():
    """Run all tests in the tests directory"""
    from importlib.util import find_spec

    # Find the path to the tests directory
    try:
        test_ollama_spec = find_spec("swarm_squad.tests.test_ollama")
        if not test_ollama_spec or not test_ollama_spec.origin:
            logger.error("Could not find test_ollama module")
            return 1

        tests_dir = os.path.dirname(test_ollama_spec.origin)
        logger.info(f"Running tests from directory: {tests_dir}")

        # Run the test script with no arguments to run all tests
        result = subprocess.run(
            [sys.executable, os.path.join(tests_dir, "test_ollama.py")]
        )
        return result.returncode
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return 1


def test_ollama_model(model):
    """Test Ollama connectivity with a specific model"""
    from importlib.util import find_spec

    try:
        test_ollama_spec = find_spec("swarm_squad.tests.test_ollama")
        if not test_ollama_spec or not test_ollama_spec.origin:
            logger.error("Could not find test_ollama module")
            return 1

        test_script = test_ollama_spec.origin
        logger.info(f"Testing Ollama with model: {model}")

        # Run the test script with model argument
        result = subprocess.run([sys.executable, test_script, "-m", model])
        return result.returncode
    except Exception as e:
        logger.error(f"Error testing Ollama: {e}")
        return 1


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()

    # Handle test commands
    if args.test:
        sys.exit(run_tests())

    if args.test_ollama is not None:
        sys.exit(test_ollama_model(args.test_ollama))

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
