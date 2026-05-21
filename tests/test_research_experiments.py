from __future__ import annotations

import json
from pathlib import Path

import pytest

import swarm_squad_ep1.research.experiments as experiments
from swarm_squad_ep1.research.runner import Result


def _fake_result() -> Result:
    return Result(
        scenario={"name": "fake"},
        destination_reached=True,
        steps_to_destination=42,
        total_steps=50,
        duration_s=1.23,
        final_Jn=0.95,
        avg_Jn=0.93,
        avg_comm_quality=0.91,
        total_path_length=123.0,
        packet_loss_rate=0.02,
        spoof_tp=5,
        spoof_fp=0,
        spoof_fn=1,
        spoof_tn=20,
        detection_rate=0.8333,
        false_positive_rate=0.0,
        precision=1.0,
        llm_calls=4,
        llm_parse_success=4,
        llm_parse_fail=0,
        llm_repair_success=0,
        llm_fallback_used=0,
    )


def test_run_experiment_writes_csv_and_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(experiments, "run_scenario", lambda *a, **k: _fake_result())

    results = experiments.run_experiment(
        "E1", out_dir=tmp_path, seeds=1, max_steps=5, keep_trace=False, verbose=False
    )
    assert results, "expected fake run results to be returned"

    out_root = Path(tmp_path) / "E1"
    csv_files = list(out_root.glob("*.csv"))
    summary_files = list(out_root.glob("*_summary.json"))
    assert csv_files, "expected CSV output"
    assert summary_files, "expected summary JSON output"

    csv_text = csv_files[0].read_text(encoding="utf-8")
    assert "experiment,scenario_name,seed" in csv_text
    assert "E1" in csv_text

    summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
    assert summary["experiment"] == "E1"
    assert isinstance(summary["summary"], list)


def test_run_experiment_rejects_unknown_experiment():
    with pytest.raises(ValueError, match="Unknown experiment"):
        experiments.run_experiment("NOT_REAL", seeds=1, verbose=False)
