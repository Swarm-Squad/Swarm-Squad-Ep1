"""Swarm Squad Ep1 package exports."""

__all__ = ["SwarmSquadClient"]


def __getattr__(name: str):
    if name == "SwarmSquadClient":
        from .client import SwarmSquadClient

        return SwarmSquadClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
