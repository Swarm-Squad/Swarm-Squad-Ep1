"""
GUI for the formation control simulation.
"""

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import swarm_squad.config as config
import swarm_squad.visualization as visualization
from swarm_squad.controllers.controller_factory import ControllerFactory, ControllerType
from swarm_squad.models.swarm_state import SwarmState


class FormationControlGUI:
    """
    GUI for the formation control simulation.

    This class handles the visualization and user interaction for the
    formation control simulation.
    """

    def __init__(self, root):
        """
        Initialize the GUI.

        Args:
            root: The tkinter root window
        """
        self.root = root
        self.root.title("Formation Control Simulation")

        # Create main figure with subplots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 10))

        # Create canvas for all plots
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Only bind mouse events to the formation scene subplot
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        # Initialize state and controllers
        self.swarm_state = SwarmState()
        self.controller_factory = ControllerFactory(self.swarm_state)

        # Set the combined controller as the active controller
        self.controller_factory.set_active_controller(ControllerType.COMBINED)

        # Initialize simulation control variables
        self.running = False
        self.paused = False
        self.max_iter = config.MAX_ITER

        # Add drawing state variables
        self.drawing_obstacle = False
        self.obstacle_start = None
        self.temp_circle = None  # Store temporary circle while drawing

        # Create control buttons
        self.create_plot_controls()

        # Auto-start the simulation
        self.running = True
        self.simulation_step()

    def create_plot_controls(self):
        """Create control buttons for the simulation"""
        # Create main control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Button styles and colors
        button_width = 10
        button_height = 2

        # Create buttons
        self.pause_button = tk.Button(
            control_frame,
            text="Pause",
            command=self.pause_simulation,
            bg="#fdf2ca",  # Yellow
            width=button_width,
            height=button_height,
        )

        self.continue_button = tk.Button(
            control_frame,
            text="Continue",
            command=self.continue_simulation,
            bg="#e3f0d8",  # Green
            width=button_width,
            height=button_height,
        )

        self.reset_button = tk.Button(
            control_frame,
            text="Reset",
            command=self.reset_simulation,
            bg="#d8e3f0",  # Blue
            width=button_width,
            height=button_height,
        )

        self.stop_button = tk.Button(
            control_frame,
            text="Stop",
            command=self.stop_simulation,
            bg="#f9aeae",  # Red
            width=button_width,
            height=button_height,
        )

        # Add Undo button
        self.undo_button = tk.Button(
            control_frame,
            text="Undo",
            command=self.undo_last_obstacle,
            bg="#e6e6e6",  # Light Gray
            width=button_width,
            height=button_height,
        )

        # Pack buttons horizontally with spacing
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.continue_button.pack(side=tk.LEFT, padx=5)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.undo_button.pack(side=tk.LEFT, padx=5)

    def update_plot(self):
        """Update the plot with the current swarm state"""
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
        )
        self.canvas.draw()

    def simulation_step(self):
        """Perform a single step of the simulation"""
        if (
            self.running
            and not self.paused
            and self.swarm_state.iteration < self.max_iter
        ):
            # Perform the control step
            self.controller_factory.step()

            # Update the plot
            self.update_plot()

            # Direct check for convergence (only if not already converged)
            if (
                not self.swarm_state.Jn_converged
                and self.swarm_state.check_convergence()
            ):
                print(
                    f"Formation completed: Jn values has converged in {round(self.swarm_state.t_elapsed[-1], 2)} seconds {self.swarm_state.iteration - 20} iterations.\nSimulation paused."
                )
                self.swarm_state.Jn_converged = True
                self.running = False
                self.update_plot()
                return

            # Check if swarm center is close to destination
            if self.swarm_state.check_destination_reached():
                print(
                    f"Swarm has reached the destination in {round(self.swarm_state.t_elapsed[-1], 2)} seconds {self.swarm_state.iteration} iterations!"
                )
                self.running = False
            else:
                # Schedule the next step
                self.root.after(50, self.simulation_step)

    def pause_simulation(self):
        """Pause the simulation"""
        self.paused = True
        self.running = False  # Stop the simulation loop

    def continue_simulation(self):
        """Continue the simulation after pause"""
        if not self.running:  # Only restart if not already running
            self.running = True
            self.paused = False

            # Debug the controller status
            print(
                f"DEBUG: continue_simulation - active controller: {self.controller_factory.active_controller_type}"
            )

            if (
                self.swarm_state.Jn_converged
            ):  # Check if this is after formation convergence
                print("Simulation resumed.\nSwarm start reaching to the destination...")

                # Make sure we're using the combined controller
                self.controller_factory.set_active_controller(ControllerType.COMBINED)
                print(
                    f"DEBUG: Set active controller to {self.controller_factory.active_controller_type}"
                )

            self.simulation_step()

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        # Reset the simulation
        self.running = False
        self.paused = False

        # Reset the swarm state
        self.swarm_state.reset()

        # Update the plot
        self.update_plot()

    def stop_simulation(self):
        """Stop the simulation and close the application"""
        self.running = False
        self.root.quit()  # This will close the application
        self.root.destroy()

    def on_click(self, event):
        """Handle mouse click events for drawing obstacles"""
        if event.inaxes == self.axs[0, 0]:  # Only allow drawing in formation scene
            # Pause simulation when starting to draw
            self.paused = True
            self.drawing_obstacle = True
            self.obstacle_start = (event.xdata, event.ydata)

    def on_drag(self, event):
        """Handle mouse drag events for drawing obstacles"""
        if self.drawing_obstacle and event.inaxes:
            # Calculate radius from drag distance
            radius = np.sqrt(
                (event.xdata - self.obstacle_start[0]) ** 2
                + (event.ydata - self.obstacle_start[1]) ** 2
            )

            # Remove previous temporary circle if it exists
            if self.temp_circle is not None:
                self.temp_circle.remove()

            # Draw new temporary circle
            self.temp_circle = plt.Circle(
                self.obstacle_start, radius, color="red", alpha=0.3
            )
            self.axs[0, 0].add_artist(self.temp_circle)
            self.canvas.draw()

    def on_release(self, event):
        """Handle mouse release events for placing obstacles"""
        if self.drawing_obstacle and event.inaxes:
            radius = np.sqrt(
                (event.xdata - self.obstacle_start[0]) ** 2
                + (event.ydata - self.obstacle_start[1]) ** 2
            )

            # Add permanent obstacle
            self.swarm_state.add_obstacle(
                self.obstacle_start[0], self.obstacle_start[1], radius
            )

            # Clean up
            self.drawing_obstacle = False
            self.obstacle_start = None
            if self.temp_circle is not None:
                self.temp_circle.remove()
                self.temp_circle = None

            # Update plot with new obstacle
            self.update_plot()

            # Resume simulation properly
            self.paused = False
            self.running = True
            self.simulation_step()  # Restart the simulation loop

    def undo_last_obstacle(self):
        """Remove the most recently added obstacle"""
        self.swarm_state.remove_last_obstacle()
        self.update_plot()  # Update the visualization
