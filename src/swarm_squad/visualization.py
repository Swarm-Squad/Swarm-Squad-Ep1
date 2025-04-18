"""
Visualization module for the formation control simulation.
"""

import math
import re

import matplotlib.pyplot as plt
import numpy as np

import swarm_squad.config as config


def parse_llm_movement_advice(feedback_text):
    """
    Parse LLM feedback to extract movement directions and magnitudes.

    Args:
        feedback_text: LLM feedback text containing movement advice

    Returns:
        Dictionary mapping agent indices to (direction, magnitude, purpose, advice_type) tuples
        where advice_type is one of: 'evasion', 'avoidance', 'regular'
    """
    if not feedback_text:
        return {}

    movement_advice = {}

    # Convert angles to unit vectors (x, y) where angles are in degrees
    # x = cos(angle), y = sin(angle) - using math module for clarity
    def angle_to_vector(angle_degrees):
        # Convert to radians
        angle_rad = math.radians(angle_degrees)
        # Return unit vector (x, y)
        return (math.cos(angle_rad), math.sin(angle_rad))

    # Map direction words to vectors using the angles provided
    direction_vectors = {
        # Cardinal directions (4)
        "north": angle_to_vector(90),
        "east": angle_to_vector(0),
        "south": angle_to_vector(270),
        "west": angle_to_vector(180),
        # Ordinal directions (4)
        "northeast": angle_to_vector(45),
        "southeast": angle_to_vector(315),
        "southwest": angle_to_vector(225),
        "northwest": angle_to_vector(135),
        # Secondary intercardinal directions (8)
        "north-northeast": angle_to_vector(67.5),
        "east-northeast": angle_to_vector(22.5),
        "east-southeast": angle_to_vector(337.5),
        "south-southeast": angle_to_vector(292.5),
        "south-southwest": angle_to_vector(247.5),
        "west-southwest": angle_to_vector(202.5),
        "west-northwest": angle_to_vector(157.5),
        "north-northwest": angle_to_vector(112.5),
        # Tertiary intercardinal directions (16)
        "north by northeast": angle_to_vector(78.75),
        "northeast by north": angle_to_vector(56.25),
        "northeast by east": angle_to_vector(33.75),
        "east by northeast": angle_to_vector(11.25),
        "east by southeast": angle_to_vector(348.75),
        "southeast by east": angle_to_vector(326.25),
        "southeast by south": angle_to_vector(303.75),
        "south by southeast": angle_to_vector(281.25),
        "south by southwest": angle_to_vector(258.75),
        "southwest by south": angle_to_vector(236.25),
        "southwest by west": angle_to_vector(213.75),
        "west by southwest": angle_to_vector(191.25),
        "west by northwest": angle_to_vector(168.75),
        "northwest by west": angle_to_vector(146.25),
        "northwest by north": angle_to_vector(123.75),
        "north by northwest": angle_to_vector(101.25),
    }

    # Pattern for standard movement advice
    # Examples: "Agent-2: Move 8 units northwest to escape jamming"
    #           "Agent-4: Reposition 7 units west-southwest to exit interference field"
    std_pattern = r"Agent-(\d+):\s+(?:Move|Reposition)\s+(\d+(?:\.\d+)?)\s+units?\s+([\w\-\s]+?)(?:\s+to\s+(.+?))?(?:\.|\s|$)"

    # Pattern for RECOMMENDED EVASION/AVOIDANCE
    # Examples: "RECOMMENDED EVASION: Agent-0 should move 8 units north-northwest to exit the jamming field."
    #           "RECOMMENDED AVOIDANCE: Agent-1 should move 6 units north by northeast to avoid the jamming field."
    rec_pattern = r"RECOMMENDED (\w+): Agent-(\d+) should move (\d+(?:\.\d+)?)\s+units?\s+([\w\-\s]+?)(?:\s+to\s+(.+?))?(?:\.|\s|$)"

    # First look for standard advice
    matches = re.finditer(std_pattern, feedback_text)
    for match in matches:
        agent_idx = int(match.group(1))
        magnitude = float(match.group(2))
        direction = match.group(3).lower().strip()
        purpose = match.group(4) if match.group(4) else ""

        # Check if the direction is valid
        if direction in direction_vectors:
            # Add as regular advice
            movement_advice[agent_idx] = (
                direction_vectors[direction],
                magnitude,
                purpose,
                "regular",
            )

    # Then look for recommended evasion/avoidance
    matches = re.finditer(rec_pattern, feedback_text)
    for match in matches:
        advice_type = match.group(1).lower()  # EVASION or AVOIDANCE
        agent_idx = int(match.group(2))
        magnitude = float(match.group(3))
        direction = match.group(4).lower().strip()
        purpose = match.group(5) if match.group(5) else ""

        # Check if the direction is valid
        if direction in direction_vectors:
            # If agent already has advice but this is evasion/avoidance, prioritize this one
            if advice_type in ["evasion", "avoidance"]:
                movement_advice[agent_idx] = (
                    direction_vectors[direction],
                    magnitude,
                    purpose,
                    advice_type,
                )

    return movement_advice


def plot_formation_scene(
    ax,
    swarm_position,
    PT,
    communication_qualities_matrix,
    node_colors,
    line_colors,
    obstacles,
    swarm_destination,
    agent_status=None,
    jamming_affected=None,
    llm_feedback=None,
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
        agent_status: Status of each agent (active or returning)
        jamming_affected: Whether agents are affected by jamming
        llm_feedback: LLM feedback text with movement recommendations
    """
    ax.set_title("Formation Scene")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$", rotation=0)

    # Plot the nodes with status indicators
    for i in range(swarm_position.shape[0]):
        # Define marker style based on agent status
        marker_style = "o"  # Default marker

        # Get status if provided
        is_active = True
        is_jammed = False
        if agent_status is not None:
            is_active = agent_status[i]
        if jamming_affected is not None:
            is_jammed = jamming_affected[i]

        # Change marker for returning agents
        if not is_active:
            marker_style = "x"  # X for inactive/returning agents

        # Add special outline for jamming-affected agents
        if is_jammed:
            # Draw outer ring for jamming-affected agents
            ax.scatter(*swarm_position[i], s=100, color="yellow", alpha=0.3)

        # Draw the agent marker
        ax.scatter(*swarm_position[i], color=node_colors[i], marker=marker_style)

    # Plot the edges
    for i in range(swarm_position.shape[0]):
        for j in range(i + 1, swarm_position.shape[0]):
            if communication_qualities_matrix[i, j] > PT:
                ax.plot(
                    *zip(swarm_position[i], swarm_position[j]),
                    color=line_colors[i, j],
                    linestyle="--",
                )

    # Draw movement vectors from LLM advice
    if llm_feedback:
        movement_advice = parse_llm_movement_advice(llm_feedback)
        for agent_idx, (
            direction_vector,
            magnitude,
            purpose,
            advice_type,
        ) in movement_advice.items():
            if agent_idx < len(swarm_position):
                # Create vector with correct direction and magnitude
                dx, dy = direction_vector
                # Scale by magnitude
                dx *= magnitude
                dy *= magnitude

                # Set color and style based on advice type
                if advice_type == "evasion":
                    vector_color = "red"
                    text_color = "red"
                    width = 0.007
                    alpha = 0.9
                    headwidth = 5
                    headlength = 6
                elif advice_type == "avoidance":
                    vector_color = "darkorange"
                    text_color = "darkorange"
                    width = 0.006
                    alpha = 0.85
                    headwidth = 5
                    headlength = 6
                else:  # regular
                    vector_color = "blue"
                    text_color = "blue"
                    width = 0.005
                    alpha = 0.8
                    headwidth = 4
                    headlength = 5

                # Draw the vector
                ax.quiver(
                    swarm_position[agent_idx, 0],
                    swarm_position[agent_idx, 1],
                    dx,
                    dy,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color=vector_color,
                    width=width,
                    headwidth=headwidth,
                    headlength=headlength,
                    alpha=alpha,
                )

                # Add purpose text at the end of the vector
                if purpose:
                    text_pos = (
                        swarm_position[agent_idx, 0] + dx * 1.1,
                        swarm_position[agent_idx, 1] + dy * 1.1,
                    )
                    ax.text(
                        text_pos[0],
                        text_pos[1],
                        f"{purpose}",
                        fontsize=8,
                        color=text_color,
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor="white", alpha=0.7, edgecolor="none", pad=1
                        ),
                    )

    ax.axis("equal")

    # Add obstacles to formation scene based on type
    for obstacle in obstacles:
        x, y, radius = obstacle

        # Default obstacle color for hard obstacles
        obstacle_color = "gray"  # Gray for hard obstacles
        obstacle_alpha = 0.4

        # Show obstacle based on current mode
        if config.OBSTACLE_MODE == config.ObstacleMode.LOW_POWER_JAMMING:
            obstacle_color = "yellow"  # Yellow for low-power jamming

            # Show jamming radius
            jamming_radius = radius * config.JAMMING_RADIUS_MULTIPLIER
            jamming_circle = plt.Circle(
                (x, y), jamming_radius, color=obstacle_color, alpha=0.15
            )
            ax.add_artist(jamming_circle)

        elif config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING:
            obstacle_color = "red"  # Red for high-power jamming

            # Show jamming radius
            jamming_radius = radius * config.JAMMING_RADIUS_MULTIPLIER
            jamming_circle = plt.Circle(
                (x, y), jamming_radius, color=obstacle_color, alpha=0.15
            )
            ax.add_artist(jamming_circle)

        # Draw the physical obstacle
        circle = plt.Circle((x, y), radius, color=obstacle_color, alpha=obstacle_alpha)
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
    ax,
    swarm_position,
    swarm_paths,
    node_colors,
    obstacles,
    swarm_destination,
    agent_status=None,
    jamming_affected=None,
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
        agent_status: Status of each agent (active or returning)
        jamming_affected: Whether agents are affected by jamming
    """
    ax.set_title("Swarm Trajectories")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$", rotation=0)

    swarm_size = swarm_position.shape[0]

    # If no paths have been recorded yet, just show current positions
    if not swarm_paths:
        # Just plot the current positions
        for i in range(swarm_size):
            # Define marker style based on agent status
            marker_style = "o"  # Default marker

            # Get status if provided
            is_active = True
            is_jammed = False
            if agent_status is not None:
                is_active = agent_status[i]
            if jamming_affected is not None:
                is_jammed = jamming_affected[i]

            # Change marker for returning agents
            if not is_active:
                marker_style = "x"  # X for inactive/returning agents

            # Add special outline for jamming-affected agents
            if is_jammed:
                # Draw outer ring for jamming-affected agents
                ax.scatter(*swarm_position[i], s=100, color="yellow", alpha=0.3)

            # Draw the agent marker
            ax.scatter(*swarm_position[i], color=node_colors[i], marker=marker_style)
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

    # Add obstacles to trajectory plot with type differentiation
    for obstacle in obstacles:
        x, y, radius = obstacle

        # Default obstacle color for hard obstacles
        obstacle_color = "gray"  # Gray for hard obstacles
        obstacle_alpha = 0.4

        # Show obstacle based on current mode
        if config.OBSTACLE_MODE == config.ObstacleMode.LOW_POWER_JAMMING:
            obstacle_color = "yellow"  # Yellow for low-power jamming

            # Show jamming radius
            jamming_radius = radius * config.JAMMING_RADIUS_MULTIPLIER
            jamming_circle = plt.Circle(
                (x, y), jamming_radius, color=obstacle_color, alpha=0.15
            )
            ax.add_artist(jamming_circle)

        elif config.OBSTACLE_MODE == config.ObstacleMode.HIGH_POWER_JAMMING:
            obstacle_color = "red"  # Red for high-power jamming

            # Show jamming radius
            jamming_radius = radius * config.JAMMING_RADIUS_MULTIPLIER
            jamming_circle = plt.Circle(
                (x, y), jamming_radius, color=obstacle_color, alpha=0.15
            )
            ax.add_artist(jamming_circle)

        # Draw the physical obstacle
        circle = plt.Circle((x, y), radius, color=obstacle_color, alpha=obstacle_alpha)
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
    agent_status=None,
    jamming_affected=None,
    llm_controller=None,
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
        agent_status: Status of each agent (active or returning)
        jamming_affected: Whether agents are affected by jamming
        llm_controller: The LLM controller for feedback
    """
    # Clear all axes
    for ax in axs.flat:
        ax.clear()

    # Get LLM feedback for visualization
    llm_feedback = None
    if llm_controller:
        llm_feedback = llm_controller.get_last_feedback()

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
        agent_status,
        jamming_affected,
        llm_feedback,
    )

    # Plot swarm trajectories
    plot_swarm_trajectories(
        axs[0, 1],
        swarm_position,
        swarm_paths,
        node_colors,
        obstacles,
        swarm_destination,
        agent_status,
        jamming_affected,
    )

    # Plot Jn performance
    plot_jn_performance(axs[1, 0], t_elapsed, Jn)

    # Plot rn performance
    plot_rn_performance(axs[1, 1], t_elapsed, rn)

    # Adjust the layout
    plt.tight_layout()
