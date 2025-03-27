"""
Visualization module for the formation control simulation.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_formation_scene(
    ax,
    swarm_position,
    PT,
    communication_qualities_matrix,
    node_colors,
    line_colors,
    obstacles,
    swarm_destination,
):
    """
    Plot the formation scene.

    Args:
        ax: The axis to plot on
        swarm_position: The positions of the swarm
        PT: The reception probability threshold
        communication_qualities_matrix: Communication quality between agents
        node_colors: The colors of the nodes
        line_colors: The colors of the lines
        obstacles: List of obstacles
        swarm_destination: The destination of the swarm
    """
    ax.set_title("Formation Scene")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$", rotation=0)

    # Plot the nodes
    for i in range(swarm_position.shape[0]):
        ax.scatter(*swarm_position[i], color=node_colors[i])

    # Plot the edges
    for i in range(swarm_position.shape[0]):
        for j in range(i + 1, swarm_position.shape[0]):
            if communication_qualities_matrix[i, j] > PT:
                ax.plot(
                    *zip(swarm_position[i], swarm_position[j]),
                    color=line_colors[i, j],
                    linestyle="--",
                )

    ax.axis("equal")

    # Add obstacles to formation scene
    for obstacle in obstacles:
        x, y, radius = obstacle
        circle = plt.Circle((x, y), radius, color="red", alpha=0.3)
        ax.add_artist(circle)

    # Plot destination in formation scene
    ax.plot(
        swarm_destination[0],
        swarm_destination[1],
        marker="s",
        markersize=10,
        color="none",
        markeredgecolor="black",
    )
    ax.text(
        swarm_destination[0],
        swarm_destination[1] + 3,
        "Destination",
        ha="center",
        va="bottom",
    )


def plot_swarm_trajectories(
    ax, swarm_position, swarm_paths, node_colors, obstacles, swarm_destination
):
    """
    Plot the swarm trajectories.

    Args:
        ax: The axis to plot on
        swarm_position: The positions of the swarm
        swarm_paths: The paths of the swarm
        node_colors: The colors of the nodes
        obstacles: List of obstacles
        swarm_destination: The destination of the swarm
    """
    ax.set_title("Swarm Trajectories")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$", rotation=0)

    swarm_size = swarm_position.shape[0]

    # If no paths have been recorded yet, just show current positions
    if not swarm_paths:
        # Just plot the current positions
        for i in range(swarm_size):
            ax.scatter(*swarm_position[i], color=node_colors[i])
    else:
        # Convert the list of positions to a numpy array
        trajectory_array = np.array(swarm_paths)

        # Plot the trajectories
        for i in range(swarm_size):
            # Plot the trajectory line
            ax.plot(
                trajectory_array[:, i, 0],
                trajectory_array[:, i, 1],
                color=node_colors[i],
            )

            # Don't try to plot arrows if we have only one position
            if len(trajectory_array) > 1:
                # Calculate the differences between consecutive points
                step = max(1, len(trajectory_array) // swarm_size)
                sampled_trajectory = trajectory_array[::step]

                if len(sampled_trajectory) > 1:  # Need at least 2 points for diff
                    dx = np.diff(sampled_trajectory[:, i, 0])
                    dy = np.diff(sampled_trajectory[:, i, 1])

                    # Initialize normalized arrays with zeros
                    dx_norm = np.zeros_like(dx)
                    dy_norm = np.zeros_like(dy)

                    # Normalize the vectors where dx and dy are not both zero
                    for j in range(len(dx)):
                        if dx[j] != 0 or dy[j] != 0:
                            norm = np.sqrt(dx[j] ** 2 + dy[j] ** 2)
                            dx_norm[j] = dx[j] / norm
                            dy_norm[j] = dy[j] / norm

                    # Scale the vectors by a constant factor
                    scale_factor = 2
                    dx_scaled = dx_norm * scale_factor
                    dy_scaled = dy_norm * scale_factor

                    # Plot the trajectory with larger arrows
                    ax.quiver(
                        sampled_trajectory[:-1, i, 0],
                        sampled_trajectory[:-1, i, 1],
                        dx_scaled,
                        dy_scaled,
                        color=node_colors[i],
                        scale_units="xy",
                        angles="xy",
                        scale=1,
                        headlength=10,
                        headaxislength=9,
                        headwidth=8,
                    )

        # Plot the initial positions if we have any paths
        if len(trajectory_array) > 0:
            ax.scatter(
                trajectory_array[0, :, 0], trajectory_array[0, :, 1], color=node_colors
            )

    # Add obstacles to trajectory plot
    for obstacle in obstacles:
        x, y, radius = obstacle
        circle = plt.Circle((x, y), radius, color="red", alpha=0.3)
        ax.add_artist(circle)

    # Plot destination in trajectory plot
    ax.plot(
        swarm_destination[0],
        swarm_destination[1],
        marker="s",
        markersize=10,
        color="none",
        markeredgecolor="black",
    )
    ax.text(
        swarm_destination[0],
        swarm_destination[1] + 3,
        "Destination",
        ha="center",
        va="bottom",
    )


def plot_jn_performance(ax, t_elapsed, Jn):
    """
    Plot the Jn performance.

    Args:
        ax: The axis to plot on
        t_elapsed: The elapsed time
        Jn: The Jn values
    """
    ax.set_title("Average Communication Performance Indicator")
    ax.plot(t_elapsed, Jn)
    ax.set_xlabel("$t(s)$")
    ax.set_ylabel("$J_n$", rotation=0, labelpad=20)
    if len(Jn) > 0:  # Only add text if there are values
        ax.text(t_elapsed[-1], Jn[-1], "Jn={:.4f}".format(Jn[-1]), ha="right", va="top")


def plot_rn_performance(ax, t_elapsed, rn):
    """
    Plot the rn performance.

    Args:
        ax: The axis to plot on
        t_elapsed: The elapsed time
        rn: The rn values
    """
    ax.set_title("Average Distance Performance Indicator")
    ax.plot(t_elapsed, rn)
    ax.set_xlabel("$t(s)$")
    ax.set_ylabel("$r_n$", rotation=0, labelpad=20)
    if len(rn) > 0:  # Only add text if there are values
        ax.text(
            t_elapsed[-1], rn[-1], "$r_n$={:.4f}".format(rn[-1]), ha="right", va="top"
        )


def plot_all_figures(
    axs,
    t_elapsed,
    Jn,
    rn,
    swarm_position,
    PT,
    communication_qualities_matrix,
    swarm_size,
    swarm_paths,
    node_colors,
    line_colors,
    obstacles,
    swarm_destination,
):
    """
    Plot all figures for the simulation.

    Args:
        axs: The axes of the figure
        t_elapsed: The elapsed time
        Jn: The Jn values
        rn: The rn values
        swarm_position: The positions of the swarm
        PT: The reception probability threshold
        communication_qualities_matrix: Communication quality between agents
        swarm_size: The number of agents in the swarm
        swarm_paths: The paths of the swarm
        node_colors: The colors of the nodes
        line_colors: The colors of the lines
        obstacles: List of obstacles
        swarm_destination: The destination of the swarm
    """
    for ax in axs.flatten():
        ax.clear()

    # Plot formation scene
    plot_formation_scene(
        axs[0, 0],
        swarm_position,
        PT,
        communication_qualities_matrix,
        node_colors,
        line_colors,
        obstacles,
        swarm_destination,
    )

    # Plot swarm trajectories
    plot_swarm_trajectories(
        axs[0, 1],
        swarm_position,
        swarm_paths,
        node_colors,
        obstacles,
        swarm_destination,
    )

    # Plot Jn performance
    plot_jn_performance(axs[1, 0], t_elapsed, Jn)

    # Plot rn performance
    plot_rn_performance(axs[1, 1], t_elapsed, rn)

    plt.tight_layout()
