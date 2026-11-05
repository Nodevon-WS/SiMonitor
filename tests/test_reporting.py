import json

import pytest

from sim_efficiency_monitor.cli import main, verify
from sim_efficiency_monitor.reporting import summarize_resources
from sim_efficiency_monitor import Monitor
from test_monitor import FakeCollector


def test_report_cli_legacy_and_summary_tamper(tmp_path, capsys):
    directory = tmp_path / "run"
    with Monitor(directory, collector=FakeCollector()) as monitor:
        monitor.advance(1)
    report = verify(directory)
    assert report["resource_usage"]["cpu"]["mean"] == .5
    markdown = (directory / "report.md").read_text(encoding="utf-8")
    assert "CPU utilization | 50.00% | 50.00%" in markdown
    assert "GPU utilization (selected devices, equal mean) | 25.00%" in markdown
    assert "RAM utilization | 50.00%" in markdown
    assert "RAM used (total - available) | null" in markdown
    assert main(["report", str(directory)]) == 0
    assert markdown.strip() == capsys.readouterr().out.strip()
    assert main(["report", str(directory), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    report["resource_usage"]["cpu"]["mean"] = .99
    path = directory / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Resource usage differs"):
        verify(directory)
    del report["resource_usage"]
    path.write_text(json.dumps(report), encoding="utf-8")
    assert main(["report", str(directory)]) == 0
    assert "CPU utilization | 50.00%" in capsys.readouterr().out
    assert "resource_usage" not in json.loads(path.read_text(encoding="utf-8"))


def test_available_summary_uses_own_coverage_for_each_device(tmp_path):
    rows = [
        {"wall_time_s": 1, "cpu": None, "gpu": .3, "mem": .4, "gpus": [
            {"uuid": "GPU-0", "utilization": .3, "memory_used_bytes": 10}]},
        {"wall_time_s": 4, "cpu": .6, "gpu": None, "mem": .8, "gpus": []},
    ]
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = summarize_resources(tmp_path, {}, "available_time_weighted")
    assert result["cpu"]["mean"] == pytest.approx(.6)
    assert result["gpu"]["mean"] == pytest.approx(.3)
    assert result["mem"]["mean"] == pytest.approx(.7)
    assert result["gpus"][0]["utilization"]["coverage_seconds"] == 1
    assert result["gpus"][0]["memory_used_bytes"]["mean"] == 10
