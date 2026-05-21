#!/usr/bin/env python3
"""Runtime launcher for Swarm Squad Ep1 services."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional

from swarm_squad_ep1.config import CHAT_API_PORT, QDRANT_HOST, QDRANT_PORT, SIM_API_PORT

ServiceMode = Literal["dual", "simulation", "chat"]


@dataclass
class Service:
    """A managed runtime service process."""

    name: str
    command: list[str]
    port: int
    process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """Start the service process."""
        print(f"[START] {self.name} on port {self.port}...")
        try:
            self.process = subprocess.Popen(self.command, stdout=None, stderr=None)
            time.sleep(1)
            if self.process.poll() is not None:
                print(f"[ERROR] {self.name} failed to start")
                return False
            print(f"[OK] {self.name} started (PID: {self.process.pid})")
            return True
        except Exception as exc:
            print(f"[ERROR] {self.name}: {exc}")
            return False

    def stop(self) -> None:
        """Stop the service if running."""
        if self.process and self.process.poll() is None:
            print(f"[STOP] {self.name}...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def is_running(self) -> bool:
        """Check whether the process is still alive."""
        return bool(self.process and self.process.poll() is None)


class ServiceManager:
    """Lifecycle manager for one or more services."""

    def __init__(self) -> None:
        self.services: list[Service] = []
        self._setup_signals()

    def _setup_signals(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, sig, frame) -> None:  # noqa: ARG002
        print("\n[SIGNAL] Shutting down...")
        self.stop_all()
        raise SystemExit(0)

    def add(self, service: Service) -> None:
        self.services.append(service)

    def start_all(self) -> bool:
        print("=" * 60)
        print("STARTING SERVICES")
        print("=" * 60)
        return all(service.start() for service in self.services)

    def stop_all(self) -> None:
        print("\n[SHUTDOWN] Stopping all services...")
        for service in reversed(self.services):
            service.stop()
        print("[SHUTDOWN] Done")

    def monitor(self) -> None:
        print("\n" + "=" * 60)
        print("SERVICES RUNNING")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")
        try:
            while True:
                for service in self.services:
                    if not service.is_running():
                        print(f"[WARN] {service.name} stopped unexpectedly!")
                time.sleep(2)
        except KeyboardInterrupt:
            pass


def check_qdrant() -> bool:
    """Best-effort connectivity check for local Qdrant."""
    print("[CHECK] Qdrant...")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
        client.get_collections()
        print("[OK] Qdrant connected")
        return True
    except Exception as exc:
        print(f"[WARN] Qdrant not available: {exc}")
        print("[INFO] Start with: docker compose up -d")
        return False


def _build_services(mode: ServiceMode) -> list[Service]:
    services: list[Service] = []
    if mode in {"dual", "simulation"}:
        services.append(
            Service(
                name="Simulation API",
                command=[
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "swarm_squad_ep1.simulation.api:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(SIM_API_PORT),
                ],
                port=SIM_API_PORT,
            )
        )
    if mode in {"dual", "chat"}:
        services.append(
            Service(
                name="Chat API",
                command=[
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "swarm_squad_ep1.chat.app:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(CHAT_API_PORT),
                ],
                port=CHAT_API_PORT,
            )
        )
    return services


def launch(mode: ServiceMode = "dual", monitor: bool = True) -> int:
    """Launch selected service set and optionally monitor until Ctrl+C."""
    print("=" * 60)
    print("SWARM SQUAD EP1 STARTUP")
    print("=" * 60)
    check_qdrant()

    manager = ServiceManager()
    for service in _build_services(mode):
        manager.add(service)

    ok = manager.start_all()
    if not ok:
        print("\n[WARN] One or more services failed to start")

    print("\n" + "=" * 60)
    print("ACCESS POINTS")
    print("=" * 60)
    if mode in {"dual", "chat"}:
        print(f"  Dashboard:       http://localhost:{CHAT_API_PORT}")
        print(f"  Chat API:        http://localhost:{CHAT_API_PORT}/chat")
    if mode in {"dual", "simulation"}:
        print(f"  Simulation API:  http://localhost:{SIM_API_PORT}")
    print("=" * 60 + "\n")

    if monitor:
        manager.monitor()
        manager.stop_all()

    return 0 if ok else 1


def main() -> None:
    """Entry point for `python -m swarm_squad_ep1.runtime`."""
    raise SystemExit(launch(mode="dual", monitor=True))


if __name__ == "__main__":
    main()
