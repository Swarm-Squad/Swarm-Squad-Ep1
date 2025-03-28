"""
Configuration parameters for the formation control simulation.
"""

from enum import Enum

import numpy as np


# Obstacle modes
class ObstacleMode(Enum):
    HARD = "physical"  # Physical obstacle
    LOW_POWER_JAMMING = "low_power_jamming"  # Communication degradation
    HIGH_POWER_JAMMING = "high_power_jamming"  # Abrupt disruption


# Active obstacle mode
OBSTACLE_MODE = ObstacleMode.HARD  # Default obstacle mode

# Jamming parameters
JAMMING_RADIUS_MULTIPLIER = (
    2.0  # How much larger jamming radius is than physical radius
)
LOWPOWER_JAMMING_DEGRADATION = 0.8  # Base factor for low power jamming (higher = less degradation at edge of field)
# Note: Actual degradation is gradual based on penetration depth into jamming field
# At the edge: degradation_factor = LOWPOWER_JAMMING_DEGRADATION (mild effect)
# Deep inside: degradation_factor approaches 0.2 (severe effect)

# Simulation parameters
MAX_ITER = 1000
ALPHA = 10 ** (-5)
DELTA = 2
BETA = ALPHA * (2**DELTA - 1)
V = 3
R0 = 5
PT = 0.94

# Initial swarm positions
INITIAL_SWARM_POSITIONS = np.array(
    [[-5, 14], [-5, -19], [0, 0], [35, -4], [68, 0], [72, 13], [72, -18]],
    dtype=float,
)

# Default destination
DEFAULT_DESTINATION = np.array([35, 150], dtype=float)

# Node and line visualization colors
NODE_COLORS = [
    [108 / 255, 155 / 255, 207 / 255],  # Light Blue
    [247 / 255, 147 / 255, 39 / 255],  # Orange
    [242 / 255, 102 / 255, 171 / 255],  # Light Pink
    [255 / 255, 217 / 255, 90 / 255],  # Light Gold
    [122 / 255, 168 / 255, 116 / 255],  # Green
    [147 / 255, 132 / 255, 209 / 255],  # Purple
    [245 / 255, 80 / 255, 80 / 255],  # Red
]

# Controller parameters
DESTINATION_ATTRACTION_MAGNITUDE = 1.0  # am parameter
DESTINATION_DISTANCE_THRESHOLD = 1.0  # bm parameter
OBSTACLE_AVOIDANCE_MAGNITUDE = 3.0  # ao parameter
OBSTACLE_INFLUENCE_RANGE = 6.0  # bo parameter
WALL_FOLLOWING_MAGNITUDE = 2.0  # af parameter
WALL_DISTANCE = 10.0  # df parameter
