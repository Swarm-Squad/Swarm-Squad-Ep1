import numpy as np


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


def calculate_Jn(communication_qualities_matrix, neighbor_agent_matrix, PT):
    """
    Calculate the Jn (average communication performance indicator) value

    Parameters:
        communication_qualities_matrix (numpy.ndarray): The communication qualities matrix among agents
        neighbor_agent_matrix (numpy.ndarray): The neighbor_agent matrix which is adjacency matrix of aij value
        PT (float): The reception probability threshold

    Returns:
        float: The calculated Jn value
    """
    total_communication_quality = 0
    total_neighbors = 0
    swarm_size = communication_qualities_matrix.shape[0]
    for i in range(swarm_size):
        for j in [x for x in range(swarm_size) if x != i]:
            if neighbor_agent_matrix[i, j] > PT:
                total_communication_quality += communication_qualities_matrix[i, j]
                total_neighbors += 1
    return total_communication_quality / total_neighbors


def calculate_rn(distances_matrix, neighbor_agent_matrix, PT):
    """
    Calculate the rn (average neighboring distance performance indicator) value

    Parameters:
        distances_matrix (numpy.ndarray): The distances matrix among agents
        neighbor_agent_matrix (numpy.ndarray): The neighbor_agent matrix which is adjacency matrix of aij value
        PT (float): The reception probability threshold

    Returns:
        float: The calculated rn value
    """
    total_distance = 0
    total_neighbors = 0
    swarm_size = distances_matrix.shape[0]
    for i in range(swarm_size):
        for j in [x for x in range(swarm_size) if x != i]:
            if neighbor_agent_matrix[i, j] > PT:
                total_distance += distances_matrix[i, j]
                total_neighbors += 1
    return total_distance / total_neighbors
