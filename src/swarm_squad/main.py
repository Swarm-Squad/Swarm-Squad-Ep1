"""
Main entry point for the formation control simulation.
"""

import matplotlib

matplotlib.use("TkAgg")
import tkinter as tk

from swarm_squad.gui.formation_control_gui import FormationControlGUI


def main():
    """Main entry point for the application"""
    root = tk.Tk()
    FormationControlGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
