"""
The 32-bearing command vocabulary shared by all supervisors.

Bearing names and angles are the ones used by the telemetry hints and the
command parser in LLMController (0 deg = east, counter-clockwise).
"""

import math

import numpy as np

BEARINGS = {
    "east": 0.0,
    "east by northeast": 11.25,
    "east-northeast": 22.5,
    "northeast by east": 33.75,
    "northeast": 45.0,
    "northeast by north": 56.25,
    "north-northeast": 67.5,
    "north by northeast": 78.75,
    "north": 90.0,
    "north by northwest": 101.25,
    "north-northwest": 112.5,
    "northwest by north": 123.75,
    "northwest": 135.0,
    "northwest by west": 146.25,
    "west-northwest": 157.5,
    "west by northwest": 168.75,
    "west": 180.0,
    "west by southwest": 191.25,
    "west-southwest": 202.5,
    "southwest by west": 213.75,
    "southwest": 225.0,
    "southwest by south": 236.25,
    "south-southwest": 247.5,
    "south by southwest": 258.75,
    "south": 270.0,
    "south by southeast": 281.25,
    "south-southeast": 292.5,
    "southeast by south": 303.75,
    "southeast": 315.0,
    "southeast by east": 326.25,
    "east-southeast": 337.5,
    "east by southeast": 348.75,
}

BEARING_NAMES = list(BEARINGS)


def unit_vector(name: str) -> np.ndarray:
    """Unit vector of a bearing name."""
    angle = math.radians(BEARINGS[name])
    return np.array([math.cos(angle), math.sin(angle)])


def quantize(vector) -> str:
    """Closest of the 32 bearings to a 2D vector."""
    angle = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
    return min(
        BEARINGS,
        key=lambda name: min(
            abs(angle - BEARINGS[name]), 360.0 - abs(angle - BEARINGS[name])
        ),
    )


def command_text(commands) -> str:
    """
    Format commands in the vocabulary the LLM is asked to use.

    Args:
        commands: iterable of (agent_index, distance_units, bearing_name)

    Returns:
        Text that LLMController's parser turns into control inputs; a
        decision without commands is "No action." so that it replaces the
        previous decision, as an LLM response without commands does.
    """
    parts = [
        f"Agent-{agent}: Move {int(distance)} units {bearing}."
        for agent, distance, bearing in commands
    ]
    return " ".join(parts) if parts else "No action."
