import logging
from typing import Optional

import numpy as np
import requests

from swarm_squad.config import LLM_ENABLED, LLM_ENDPOINT, LLM_MODEL

# Set up logging
logger = logging.getLogger(__name__)


def calculate_distance(agent_i, agent_j):
    """
    Calculate the distance between two agents

    Parameters:
        agent_i (list): The position of agent i
        agent_j (list): The position of agent j

    Returns:
        float: The distance between agent i and agent j
    """
    return np.sqrt((agent_i[0] - agent_j[0]) ** 2 + (agent_i[1] - agent_j[1]) ** 2)


def calculate_aij(alpha, delta, rij, r0, v):
    """
    Calculate the aij value

    Parameters:
        alpha (float): System parameter about antenna characteristics
        delta (float): The required application data rate
        rij (float): The distance between two agents
        r0 (float): Reference distance value
        v (float): Path loss exponent

    Returns:
        float: The calculated aij (communication quality in antenna far-field) value
    """
    return np.exp(-alpha * (2**delta - 1) * (rij / r0) ** v)


def calculate_gij(rij, r0):
    """
    Calculate the gij value

    Parameters:
        rij (float): The distance between two agents
        r0 (float): Reference distance value

    Returns:
        float: The calculated gij (communication quality in antenna near-field) value
    """
    return rij / np.sqrt(rij**2 + r0**2)


def calculate_rho_ij(beta, v, rij, r0):
    """
    Calculate the rho_ij (the derivative of phi_ij) value

    Parameters:
        beta (float): alpha * (2**delta - 1)
        v (float): Path loss exponent
        rij (float): The distance between two agents
        r0 (float): Reference distance value

    Returns:
        float: The calculated rho_ij value
    """
    return (
        (-beta * v * rij ** (v + 2) - beta * v * (r0**2) * (rij**v) + r0 ** (v + 2))
        * np.exp(-beta * (rij / r0) ** v)
        / np.sqrt((rij**2 + r0**2) ** 3)
    )


def calculate_Jn(
    communication_qualities_matrix, neighbor_agent_matrix, PT, agent_status=None
):
    """
    Calculate the Jn (average communication performance indicator) value

    Parameters:
        communication_qualities_matrix (numpy.ndarray): The communication qualities matrix among agents
        neighbor_agent_matrix (numpy.ndarray): The neighbor_agent matrix which is adjacency matrix of aij value
        PT (float): The reception probability threshold
        agent_status (numpy.ndarray, optional): Boolean array indicating which agents are active

    Returns:
        float: The calculated Jn value
    """
    total_communication_quality = 0
    total_neighbors = 0
    swarm_size = communication_qualities_matrix.shape[0]

    for i in range(swarm_size):
        # Skip inactive agents
        if agent_status is not None and not agent_status[i]:
            continue

        for j in [x for x in range(swarm_size) if x != i]:
            # Skip inactive agents
            if agent_status is not None and not agent_status[j]:
                continue

            if neighbor_agent_matrix[i, j] > PT:
                total_communication_quality += communication_qualities_matrix[i, j]
                total_neighbors += 1

    # Return 0 if no valid neighbors
    if total_neighbors == 0:
        return 0

    return total_communication_quality / total_neighbors


def calculate_rn(distances_matrix, neighbor_agent_matrix, PT, agent_status=None):
    """
    Calculate the rn (average neighboring distance performance indicator) value

    Parameters:
        distances_matrix (numpy.ndarray): The distances matrix among agents
        neighbor_agent_matrix (numpy.ndarray): The neighbor_agent matrix which is adjacency matrix of aij value
        PT (float): The reception probability threshold
        agent_status (numpy.ndarray, optional): Boolean array indicating which agents are active

    Returns:
        float: The calculated rn value
    """
    total_distance = 0
    total_neighbors = 0
    swarm_size = distances_matrix.shape[0]

    for i in range(swarm_size):
        # Skip inactive agents
        if agent_status is not None and not agent_status[i]:
            continue

        for j in [x for x in range(swarm_size) if x != i]:
            # Skip inactive agents
            if agent_status is not None and not agent_status[j]:
                continue

            if neighbor_agent_matrix[i, j] > PT:
                total_distance += distances_matrix[i, j]
                total_neighbors += 1

    # Return 0 if no valid neighbors
    if total_neighbors == 0:
        return 0

    return total_distance / total_neighbors


def get_direction(from_pos, to_pos):
    """
    Get the cardinal/intercardinal direction from one position to another.

    Args:
        from_pos: Starting position [x, y]
        to_pos: Target position [x, y]

    Returns:
        Direction as string (N, NE, E, SE, S, SW, W, NW)
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    # Calculate angle in radians, convert to degrees
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # Convert angle to 0-360 range
    if angle < 0:
        angle += 360

    # Determine direction based on angle
    if 22.5 <= angle < 67.5:
        return "NE"
    elif 67.5 <= angle < 112.5:
        return "N"
    elif 112.5 <= angle < 157.5:
        return "NW"
    elif 157.5 <= angle < 202.5:
        return "W"
    elif 202.5 <= angle < 247.5:
        return "SW"
    elif 247.5 <= angle < 292.5:
        return "S"
    elif 292.5 <= angle < 337.5:
        return "SE"
    else:  # 337.5 <= angle < 360 or 0 <= angle < 22.5
        return "E"


def format_swarm_state_for_llm(swarm_state, agent_names=None):
    """
    Format the current swarm state into natural language for LLM consumption.

    Args:
        swarm_state: SwarmState object containing all agent information
        agent_names: List of names for agents (default: uses names from config)

    Returns:
        String containing natural language description of swarm state
    """
    from swarm_squad.config import AGENT_NAMES, PT

    if agent_names is None:
        agent_names = AGENT_NAMES

    # Get positions and matrices
    positions = swarm_state.swarm_position
    neighbor_matrix = swarm_state.neighbor_agent_matrix
    comm_quality_matrix = swarm_state.communication_qualities_matrix
    distances_matrix = swarm_state.distances_matrix
    destination = swarm_state.swarm_destination
    obstacles = swarm_state.obstacles

    # Start building the state description
    state_desc = []
    state_desc.append(
        f"Swarm Destination: [{destination[0]:.1f}, {destination[1]:.1f}]"
    )

    # Add obstacle information if there are obstacles
    if obstacles and len(obstacles) > 0:
        obstacle_desc = []
        for i, obs in enumerate(obstacles):
            x, y, radius = obs  # Unpack obstacle tuple
            obstacle_desc.append(
                f"Obstacle {i + 1}: Position [{x:.1f}, {y:.1f}], Radius {radius:.1f}"
            )
        state_desc.append("Obstacles: " + " | ".join(obstacle_desc))

    # Add information for each agent
    for i in range(swarm_state.swarm_size):
        agent_info = []
        pos = positions[i]
        name = agent_names[i]

        # Calculate distance and direction to destination
        dest_vector = destination - pos
        dist_to_dest = np.linalg.norm(dest_vector)
        dir_to_dest = get_direction(pos, destination)

        agent_info.append(f"{name} at position [{pos[0]:.1f}, {pos[1]:.1f}]")
        agent_info.append(
            f"Distance to destination: {dist_to_dest:.1f} units, Direction: {dir_to_dest}"
        )

        # Find connections based on PT threshold
        connected_agents = []
        for j in range(swarm_state.swarm_size):
            if i != j and neighbor_matrix[i, j] > PT:
                neighbor_pos = positions[j]
                distance = distances_matrix[i, j]
                direction = get_direction(pos, neighbor_pos)
                comm_quality = comm_quality_matrix[i, j]
                quality_desc = "poor" if comm_quality < 0.7 else "good"

                connected_agents.append(
                    f"{agent_names[j]} ({direction}, {distance:.1f} units, {comm_quality:.2f} {quality_desc} link)"
                )

        if connected_agents:
            agent_info.append("Connected to: " + ", ".join(connected_agents))
        else:
            agent_info.append("No connections to other agents")

        state_desc.append(" | ".join(agent_info))

    return "\n".join(state_desc)


def check_ollama_running() -> bool:
    """
    Check if Ollama is running and the specified model is available.

    Returns:
        bool: True if Ollama is running and the model is available, False otherwise
    """
    if not LLM_ENABLED:
        return True

    try:
        # Get base URL by removing endpoint path
        base_url = LLM_ENDPOINT.split("/api/")[0]
        logger.info(f"Checking Ollama at base URL: {base_url}")

        # Use list models endpoint
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            # Check if our model is available
            tags = response.json().get("models", [])
            model_names = [tag.get("name", "") for tag in tags]

            logger.info(f"Available models: {model_names}")

            if LLM_MODEL in model_names:
                logger.info(f"Ollama is running with {LLM_MODEL} available")
                return True
            else:
                logger.warning(f"Ollama is running but {LLM_MODEL} is not available")
                return False
        else:
            logger.warning(f"Ollama check failed with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking Ollama: {str(e)}")
        return False


def get_ollama_api_url() -> str:
    """
    Get the base URL for the Ollama API.

    Returns:
        str: The base URL for the Ollama API
    """
    return LLM_ENDPOINT.split("/api/")[0]


def format_llm_feedback(feedback: str, timestamp: Optional[str] = None) -> str:
    """
    Format LLM feedback for display.

    Args:
        feedback: Raw feedback from LLM
        timestamp: Optional timestamp string

    Returns:
        Formatted feedback string
    """
    if not feedback:
        return "No tactical advice available"

    result = f"TACTICAL ADVICE: {feedback}"

    if timestamp:
        result += f"\n\nReceived at: {timestamp}"

    return result
