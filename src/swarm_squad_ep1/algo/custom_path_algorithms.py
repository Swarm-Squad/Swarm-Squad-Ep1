"""Runtime registry for user-defined path planning algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Any, Callable, Optional

CustomPathCallable = Callable[..., Any]
ALGORITHM_MODE = "waypoint"


@dataclass(slots=True)
class CustomPathAlgorithm:
    name: str
    import_path: str
    description: str
    mode: str
    callable_obj: CustomPathCallable

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "import_path": self.import_path,
            "description": self.description,
            "mode": self.mode,
        }


_registry: dict[str, CustomPathAlgorithm] = {}
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


def _resolve_callable(import_path: str) -> CustomPathCallable:
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


def register_custom_path_algorithm(
    *,
    name: str,
    import_path: str,
    description: str = "",
    mode: str = ALGORITHM_MODE,
    replace: bool = False,
    reserved_names: Optional[list[str]] = None,
) -> dict[str, str]:
    if mode != ALGORITHM_MODE:
        raise ValueError(
            f"Unsupported mode '{mode}'. Only '{ALGORITHM_MODE}' is currently supported."
        )

    normalized_name = _normalize_name(name)
    reserved = set(reserved_names or [])
    if normalized_name in reserved:
        raise ValueError(
            f"'{normalized_name}' is reserved by a built-in algorithm; choose a different name"
        )

    callable_obj = _resolve_callable(import_path)
    entry = CustomPathAlgorithm(
        name=normalized_name,
        import_path=import_path.strip(),
        description=(description or "").strip(),
        mode=mode,
        callable_obj=callable_obj,
    )

    with _registry_lock:
        if normalized_name in _registry and not replace:
            raise ValueError(
                f"Custom algorithm '{normalized_name}' already exists (set replace=true to overwrite)"
            )
        _registry[normalized_name] = entry
        return entry.to_dict()


def unregister_custom_path_algorithm(name: str) -> Optional[dict[str, str]]:
    normalized_name = _normalize_name(name)
    with _registry_lock:
        removed = _registry.pop(normalized_name, None)
    return removed.to_dict() if removed else None


def list_custom_path_algorithms() -> list[dict[str, str]]:
    with _registry_lock:
        items = sorted(_registry.values(), key=lambda item: item.name)
    return [item.to_dict() for item in items]


def get_custom_path_algorithm(name: str) -> Optional[CustomPathAlgorithm]:
    normalized_name = _normalize_name(name)
    with _registry_lock:
        return _registry.get(normalized_name)


def get_custom_path_callable(name: str) -> Optional[CustomPathCallable]:
    entry = get_custom_path_algorithm(name)
    return entry.callable_obj if entry else None


def has_custom_path_algorithm(name: str) -> bool:
    return get_custom_path_algorithm(name) is not None


def clear_custom_path_algorithms() -> None:
    with _registry_lock:
        _registry.clear()
