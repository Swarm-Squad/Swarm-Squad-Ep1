"""Runtime registry for user-defined cryptographic algorithms."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Callable, Optional

CryptoSignCallable = Callable[..., bytes]
CryptoVerifyCallable = Callable[..., bool]


@dataclass(slots=True)
class CustomCryptoAlgorithm:
    name: str
    sign_import_path: str
    verify_import_path: str
    description: str
    sign_callable: CryptoSignCallable
    verify_callable: CryptoVerifyCallable

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "sign_import_path": self.sign_import_path,
            "verify_import_path": self.verify_import_path,
            "description": self.description,
        }


_registry: dict[str, CustomCryptoAlgorithm] = {}
_registry_lock = RLock()


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Algorithm name cannot be empty")
    if not all(ch.isalnum() or ch == "_" for ch in normalized):
        raise ValueError(
            "Algorithm name must contain only letters, numbers, underscores, or hyphens"
        )
    return normalized


def _resolve_callable(import_path: str) -> Callable:
    path = (import_path or "").strip()
    if not path:
        raise ValueError("import_path is required")
    if ":" in path:
        module_path, attr_name = path.split(":", maxsplit=1)
    elif "." in path:
        module_path, attr_name = path.rsplit(".", maxsplit=1)
    else:
        raise ValueError(
            "import_path must be in 'module:function' or 'module.function' format"
        )

    module = import_module(module_path)
    resolved = getattr(module, attr_name, None)
    if resolved is None or not callable(resolved):
        raise ValueError(f"Resolved symbol is not callable: {path}")
    return resolved


def _validate_contract(
    sign_callable: CryptoSignCallable, verify_callable: CryptoVerifyCallable
) -> None:
    key = os.urandom(32)
    data = b"swarm_squad_crypto_probe"
    sender_id = "agent_probe"
    signature = sign_callable(
        key=key,
        data=data,
        sender_id=sender_id,
        crypto=None,
    )
    if not isinstance(signature, (bytes, bytearray)):
        raise ValueError("Custom sign function must return bytes")

    verdict = verify_callable(
        key=key,
        data=data,
        signature=bytes(signature),
        sender_id=sender_id,
        crypto=None,
    )
    if not isinstance(verdict, bool):
        raise ValueError("Custom verify function must return bool")


def register_custom_crypto_algorithm(
    *,
    name: str,
    sign_import_path: str,
    verify_import_path: str,
    description: str = "",
    replace: bool = False,
    reserved_names: Optional[list[str]] = None,
) -> dict[str, str]:
    normalized_name = _normalize_name(name)
    reserved = set(reserved_names or [])
    if normalized_name in reserved:
        raise ValueError(
            f"'{normalized_name}' is reserved by a built-in algorithm; choose a different name"
        )

    sign_callable = _resolve_callable(sign_import_path)
    verify_callable = _resolve_callable(verify_import_path)
    _validate_contract(sign_callable, verify_callable)

    entry = CustomCryptoAlgorithm(
        name=normalized_name,
        sign_import_path=sign_import_path.strip(),
        verify_import_path=verify_import_path.strip(),
        description=(description or "").strip(),
        sign_callable=sign_callable,
        verify_callable=verify_callable,
    )

    with _registry_lock:
        if normalized_name in _registry and not replace:
            raise ValueError(
                f"Custom crypto algorithm '{normalized_name}' already exists (set replace=true to overwrite)"
            )
        _registry[normalized_name] = entry
        return entry.to_dict()


def unregister_custom_crypto_algorithm(name: str) -> Optional[dict[str, str]]:
    normalized_name = _normalize_name(name)
    with _registry_lock:
        removed = _registry.pop(normalized_name, None)
    return removed.to_dict() if removed else None


def list_custom_crypto_algorithms() -> list[dict[str, str]]:
    with _registry_lock:
        items = sorted(_registry.values(), key=lambda item: item.name)
    return [item.to_dict() for item in items]


def get_custom_crypto_algorithm(name: str) -> Optional[CustomCryptoAlgorithm]:
    normalized_name = _normalize_name(name)
    with _registry_lock:
        return _registry.get(normalized_name)


def has_custom_crypto_algorithm(name: str) -> bool:
    return get_custom_crypto_algorithm(name) is not None


def clear_custom_crypto_algorithms() -> None:
    with _registry_lock:
        _registry.clear()
