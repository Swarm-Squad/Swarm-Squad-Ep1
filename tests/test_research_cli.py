from __future__ import annotations

from pathlib import Path

import swarm_squad_ep1.research.__main__ as research_cli


def test_research_cli_list_command(capsys):
    code = research_cli.main(["list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Available experiments" in out
    assert "E1" in out


def test_research_cli_unknown_experiment_returns_error(capsys):
    code = research_cli.main(["run", "--experiment=DOES_NOT_EXIST", "--seeds=1"])
    assert code == 2
    err = capsys.readouterr().err
    assert "unknown experiment" in err


def test_research_cli_plot_missing_csv_returns_error(tmp_path):
    missing_csv = tmp_path / "missing.csv"
    code = research_cli.main(["plot", f"--csv={missing_csv}"])
    assert code == 2


def test_research_cli_smoke_dispatch(monkeypatch):
    called = {"smoke": False}

    def _fake_smoke(verbose=True):
        called["smoke"] = True
        return 0

    monkeypatch.setattr("swarm_squad_ep1.research.smoke_test.run_smoke", _fake_smoke)
    code = research_cli.main(["smoke"])
    assert code == 0
    assert called["smoke"] is True


def test_research_cli_run_dispatch(monkeypatch, tmp_path):
    calls = []

    def _fake_run_experiment(
        name, out_dir="results", seeds=3, max_steps=None, keep_trace=False, verbose=True
    ):
        calls.append(
            {
                "name": name,
                "out_dir": out_dir,
                "seeds": seeds,
                "max_steps": max_steps,
                "keep_trace": keep_trace,
                "verbose": verbose,
            }
        )
        return []

    monkeypatch.setattr(
        "swarm_squad_ep1.research.experiments.run_experiment", _fake_run_experiment
    )

    out_dir = Path(tmp_path)
    code = research_cli.main(
        [
            "run",
            "--experiment=E1",
            "--seeds=2",
            "--max-steps=50",
            f"--out-dir={out_dir}",
            "--keep-trace",
            "--quiet",
        ]
    )
    assert code == 0
    assert calls
    call = calls[0]
    assert call["name"] == "E1"
    assert call["out_dir"] == str(out_dir)
    assert call["seeds"] == 2
    assert call["max_steps"] == 50
    assert call["keep_trace"] is True
    assert call["verbose"] is False
