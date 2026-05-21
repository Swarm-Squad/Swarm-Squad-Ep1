from __future__ import annotations

import tomllib
from pathlib import Path

import swarm_squad_ep1.cli as cli


def test_pyproject_registers_swarm_squad_script():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert "swarm-squad-ep1" in scripts
    assert scripts["swarm-squad-ep1"] == "swarm_squad_ep1.cli:main"


def test_cli_default_launches_gui_stack(monkeypatch):
    called: dict[str, object] = {}

    def _fake_launch(mode="dual", monitor=True):
        called["mode"] = mode
        called["monitor"] = monitor
        return 0

    monkeypatch.setattr(cli.runtime, "launch", _fake_launch)
    assert cli.main([]) == 0
    assert called == {"mode": "dual", "monitor": True}


def test_cli_gui_dispatch(monkeypatch):
    called = {"gui": False}

    def _fake_launch(mode="dual", monitor=True):
        called["gui"] = True
        called["mode"] = mode
        called["monitor"] = monitor
        return 0

    monkeypatch.setattr(cli.runtime, "launch", _fake_launch)
    assert cli.main(["gui", "--no-monitor"]) == 0
    assert called["gui"] is True
    assert called["mode"] == "dual"
    assert called["monitor"] is False


def test_cli_services_dispatch(monkeypatch):
    called: dict[str, object] = {}

    def _fake_launch(mode="dual", monitor=True):
        called["mode"] = mode
        called["monitor"] = monitor
        return 0

    monkeypatch.setattr(cli.runtime, "launch", _fake_launch)
    assert cli.main(["services", "--simulation-only", "--no-monitor"]) == 0
    assert called == {"mode": "simulation", "monitor": False}


def test_cli_research_dispatch(monkeypatch):
    forwarded: list[str] = []

    def _fake_research(args):
        forwarded.extend(args)
        return 0

    monkeypatch.setattr(cli, "research_main", _fake_research)
    assert cli.main(["research", "list"]) == 0
    assert forwarded == ["list"]
