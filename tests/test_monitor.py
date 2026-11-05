import json
import time
import pytest

from sim_efficiency_monitor import Monitor, MonitorConfig
from sim_efficiency_monitor.cli import verify


class FakeCollector:
    metadata = {"scope": "test_fixture"}

    def __init__(self):
        self.closed = False

    def prime(self):
        pass

    def sample(self):
        return {"cpu": .5, "gpu": .25, "mem": .5, "errors": []}

    def close(self):
        self.closed = True


def test_failure_does_not_count_failed_step(tmp_path):
    directory = tmp_path / "failed"
    cfg = MonitorConfig(rtf_threshold=1, cres_threshold=1)
    with pytest.raises(ArithmeticError):
        with Monitor(directory, cfg, collector=FakeCollector()) as monitor:
            monitor.advance(.1)
            with monitor.step(.1):
                raise ArithmeticError("diverged")
    report = verify(directory)
    assert report["metrics"]["sim_time_s"] == .1
    assert report["status"] == "failed"
    assert "diverged" in report["reason"]
    assert report["metrics"]["efficiency_Ec"] == 0


def test_clock_reset_rejected(tmp_path):
    monitor = Monitor(tmp_path / "clock", collector=FakeCollector()).start(10)
    try:
        monitor.record_sim_time(10.1)
        for value in (10.1, 0, float("nan")):
            with pytest.raises(ValueError):
                monitor.record_sim_time(value)
    finally:
        report = monitor.stop()
    assert report["metrics"]["sim_time_s"] == pytest.approx(.1)


def test_sync_error_saves_failure_report(tmp_path):
    calls = []

    def sync():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("device error")

    directory = tmp_path / "sync-error"
    with pytest.raises(RuntimeError, match="device error"):
        with Monitor(directory, synchronize=sync, collector=FakeCollector()) as monitor:
            monitor.advance(1)
    assert verify(directory)["status"] == "failed"


def test_collector_failure_is_missing(tmp_path):
    class Broken(FakeCollector):
        def sample(self):
            raise RuntimeError("sensor unavailable")
    with Monitor(tmp_path / "missing", collector=Broken()) as monitor:
        monitor.advance(1)
    assert monitor.report["metrics"]["cres"] is None
    assert "sensor unavailable" in (tmp_path / "missing" / "samples.jsonl").read_text()


def test_queries_after_every_step_without_background_polling(tmp_path):
    class Counting(FakeCollector):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def sample(self):
            self.calls += 1
            return super().sample()
    collector = Counting()
    directory = tmp_path / "per-step"
    with Monitor(directory, MonitorConfig(sample_interval_s=.01), collector=collector) as monitor:
        time.sleep(.03)
        assert collector.calls == 0
        monitor.advance(.1)
        assert collector.calls == 1
        monitor.record_sim_time(.2)
        assert collector.calls == 2
        with monitor.step(.1):
            pass
        assert collector.calls == 3
        with pytest.raises(ValueError, match="steps=1"):
            monitor.advance(.2, steps=2)
        assert collector.calls == 3
    assert collector.calls == 4
    report = verify(directory)
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    assert [r["physics_steps"] for r in rows] == [1, 2, 3, 3]
    assert [r["sample_phase"] for r in rows] == ["step", "step", "step", "final"]
    assert report["resource_averaging"] == "available_time_weighted"


def test_partial_samples_produce_means_and_cres(tmp_path):
    class Intermittent(FakeCollector):
        def __init__(self):
            self.values = iter([
                {"cpu": None, "gpu": None, "mem": .5},
                {"cpu": .2, "gpu": .6, "mem": .5},
                {"cpu": .4, "gpu": None, "mem": .5},
            ])
        def sample(self):
            return next(self.values)
    directory = tmp_path / "partial"
    with Monitor(directory, collector=Intermittent()) as monitor:
        monitor.advance(.1)
        monitor.advance(.1)
    report = verify(directory)
    m = report["metrics"]
    assert not m["resource_data_complete"]
    assert m["mean_utilization"]["gpu"] == pytest.approx(.6)
    for key in ("cpu", "gpu", "mem"):
        assert m["mean_utilization"][key] == pytest.approx(report["resource_usage"][key]["mean"])
    assert m["cres"] == pytest.approx(m["rtf"] * (.4*.6 + .3*m["mean_utilization"]["cpu"] + .3*.5))
    markdown = (directory / "report.md").read_text()
    assert "partial coverage makes it an estimate" in markdown


def test_verify_rejects_step_gaps_and_policy_changes(tmp_path):
    import hashlib
    directory = tmp_path / "step-integrity"
    with Monitor(directory, collector=FakeCollector()) as monitor:
        monitor.advance(.1)
        monitor.advance(.1)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    original = dict(report)
    report["resource_averaging"] = "complete_time_weighted"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="measurement policy"):
        verify(directory)
    report = original
    sample_path = directory / "samples.jsonl"
    rows = sample_path.read_text().splitlines()
    sample_path.write_text("\n".join(rows[1:]) + "\n")
    report["files_sha256"]["samples.jsonl"] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="one resource sample per completed step"):
        verify(directory)
