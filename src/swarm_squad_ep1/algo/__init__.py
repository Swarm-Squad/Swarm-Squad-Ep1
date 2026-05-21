"""
Multi-Vehicle Algorithm Framework

Contains formation control, path planning, jamming response, MAVLink protocol,
spoofing attacks, and cryptographic authentication for multi-vehicle coordination.
"""

from swarm_squad_ep1.algo.base import (
    FormationState,
    JammingZone,
    MultiVehicleController,
    VehicleCommand,
)
from swarm_squad_ep1.algo.controller import UnifiedController
from swarm_squad_ep1.algo.crypto_auth import (
    CryptoAuth,
    get_crypto_auth,
    reset_crypto_auth,
)
from swarm_squad_ep1.algo.formation import FORMATION_TYPES, FormationGenerator
from swarm_squad_ep1.algo.jamming_response import JAMMING_STRATEGIES, JammingResponse
from swarm_squad_ep1.algo.mavlink import (
    MAVLinkBus,
    MAVLinkMessage,
    MessageType,
    get_mavlink_bus,
    reset_mavlink_bus,
)
from swarm_squad_ep1.algo.path_planning import (
    PATH_ALGORITHMS,
    PathPlanner,
    get_available_path_algorithms,
    list_registered_path_algorithms,
    register_path_algorithm,
    unregister_path_algorithm,
)
from swarm_squad_ep1.algo.spoofing import (
    SpoofingEngine,
    SpoofingZone,
    SpoofType,
    get_spoofing_engine,
    reset_spoofing_engine,
)
from swarm_squad_ep1.algo.v2v_channel import (
    ChannelParams,
    LinkType,
    ObstacleKind,
    ObstacleSpec,
    V2VChannelModel,
    get_channel_model,
    reset_channel_model,
)

__all__ = [
    # Base classes
    "MultiVehicleController",
    "JammingZone",
    "VehicleCommand",
    "FormationState",
    # Controller
    "UnifiedController",
    # Formation
    "FormationGenerator",
    "FORMATION_TYPES",
    # Path Planning
    "PathPlanner",
    "PATH_ALGORITHMS",
    "get_available_path_algorithms",
    "list_registered_path_algorithms",
    "register_path_algorithm",
    "unregister_path_algorithm",
    # Jamming
    "JammingResponse",
    "JAMMING_STRATEGIES",
    # MAVLink Protocol
    "MAVLinkBus",
    "MAVLinkMessage",
    "MessageType",
    "get_mavlink_bus",
    "reset_mavlink_bus",
    # Spoofing
    "SpoofingZone",
    "SpoofType",
    "SpoofingEngine",
    "get_spoofing_engine",
    "reset_spoofing_engine",
    # Crypto Auth
    "CryptoAuth",
    "get_crypto_auth",
    "reset_crypto_auth",
    # V2V Channel Model
    "V2VChannelModel",
    "ChannelParams",
    "LinkType",
    "ObstacleKind",
    "ObstacleSpec",
    "get_channel_model",
    "reset_channel_model",
]
