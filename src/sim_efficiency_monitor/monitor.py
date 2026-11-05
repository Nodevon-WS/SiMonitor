from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .config import MonitorConfig, positive
from .metrics import ResourceIntegral
from .resources import ProcessCollector
from .reporting import render_report, summarize_resources
from . import __version__


class Monitor:
    """One measured run. Use one instance per scene/repetition; no pause/reset API."""

    def __init__(self, output_dir, config=None, *, synchronize=None, collector=None, pid=None):
        self.config = config or MonitorConfig()
        self.output_dir = Path(output_dir)
        self.synchronize = synchronize
        self.collector = collector
        self.pid = pid
        self._state = "new"
        self._lock = threading.RLock()
        self._averaging = "available_time_weighted"
        self._integral = ResourceIntegral(self._averaging)
        self._sim_s = 0.0
        self._steps = 0
        self.report = None

    def _event(self, kind, **fields):
        record = {"event": kind, "utc": datetime.now(timezone.utc).isoformat(), **fields}
        self._events.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._events.flush()

    def start(self, sim_time_s=0.0):
        if self._state != "new":
            raise RuntimeError("Monitor instances cannot be reused")
        positive(sim_time_s, "sim_time_s", zero=True)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._state = "starting"
        try:
            self._samples = (self.output_dir / "samples.jsonl").open("x", encoding="utf-8")
            self._events = (self.output_dir / "events.jsonl").open("x", encoding="utf-8")
            self.collector = self.collector or ProcessCollector(self.config.gpu_devices, pid=self.pid)
            if self.synchronize:
                self.synchronize()
            self._initial_sim_s = sim_time_s
            self._absolute_sim_s = sim_time_s
            self._event("start", initial_sim_time_s=sim_time_s,
                        config_sha256=self.config.sha256,
                        sampling_mode="simulation_step", resource_averaging=self._averaging)
            self.collector.prime()
            self._start_ns = time.perf_counter_ns()
            self._state = "running"
            return self
        except BaseException:
            self._state = "failed"
            self._close()
            raise

    def _capture(self, phase):
        try:
            values = self.collector.sample()
        except Exception as exc:
            values = {"cpu": None, "gpu": None, "mem": None,
                      "errors": [f"Collector failure: {type(exc).__name__}: {exc}"]}
        elapsed = (time.perf_counter_ns() - self._start_ns) / 1e9
        self._integral.add(elapsed, values)
        live = self._integral.result(self._sim_s, self._steps, "completed", self.config)
        self._samples.write(json.dumps({"wall_time_s": elapsed, "sim_time_s": self._sim_s,
                                        "physics_steps": self._steps, "sample_phase": phase,
                                        "cumulative_rtf": live["rtf"],
                                        "cumulative_cres": live["cres"], **values},
                                       ensure_ascii=False, allow_nan=False) + "\n")
        self._samples.flush()

    def advance(self, dt_s, *, steps=1):
        """Report one completed physics step and immediately query resources."""
        positive(dt_s, "dt_s")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps != 1:
            raise ValueError("Per-step sampling requires steps=1; report after each simulation step")
        with self._lock:
            if self._state != "running":
                raise RuntimeError("Monitor is not running")
            next_sim = self._sim_s + dt_s
            if not math.isfinite(next_sim):
                raise ValueError("simulation time overflow")
            self._sim_s = next_sim
            self._absolute_sim_s = self._initial_sim_s + next_sim
            self._steps += steps
            self._capture("step")

    def record_sim_time(self, sim_time_s, *, steps=1):
        """Use a monotonic engine clock; a reset must start a separate run."""
        positive(sim_time_s, "sim_time_s", zero=True)
        with self._lock:
            self.advance(sim_time_s - self._absolute_sim_s, steps=steps)
            self._absolute_sim_s = sim_time_s

    @contextmanager
    def step(self, dt_s):
        """Count dt only if the wrapped step returns successfully."""
        positive(dt_s, "dt_s")
        yield
        self.advance(dt_s)

    def stop(self, status="completed", reason=None):
        if self._state == "stopped":
            return self.report
        if self._state != "running":
            raise RuntimeError("Monitor is not running")
        if status not in {"completed", "failed", "aborted", "timeout"}:
            raise ValueError("invalid run status")
        sync_error = None
        try:
            if self.synchronize:
                self.synchronize()
        except BaseException as exc:
            sync_error = exc
            status, reason = "failed", f"Synchronization failed: {type(exc).__name__}: {exc}"
        try:
            with self._lock:
                self._state = "stopping"
                self._capture("final")
                if self._sim_s <= 0:
                    status, reason = "failed", "No completed simulation time reported"
                limit = self.config.wall_time_limit_s
                if limit is not None and self._integral.last_t > limit:
                    status, reason = "timeout", "Configured wall time limit exceeded"
                self._event("stop", status=status, reason=reason)
                self._samples.close()
                self._events.close()
                hashes = {name: _sha256(self.output_dir / name)
                          for name in ("samples.jsonl", "events.jsonl")}
                self.report = {
                    "schema_version": "1.1", "monitor_version": __version__,
                    "sampling_mode": "simulation_step", "resource_averaging": self._averaging,
                    "status": status, "reason": reason,
                    "config": self.config.to_dict(), "config_sha256": self.config.sha256,
                    "environment": self.collector.metadata,
                    "synchronization_callback_provided": self.synchronize is not None,
                    "measurement_scope": self.collector.metadata.get("scope", "unknown"),
                    "weights": {"gpu": 0.4, "cpu": 0.3, "mem": 0.3},
                    "metrics": self._integral.result(self._sim_s, self._steps, status, self.config),
                    "files_sha256": hashes,
                }
                self._write_report()
                self._state = "stopped"
        finally:
            self._close()
        if sync_error is not None:
            raise sync_error
        return self.report

    def _write_report(self):
        self.report["resource_usage"] = summarize_resources(self.output_dir, self.report["environment"], self._averaging)
        temp = self.output_dir / "report.json.tmp"
        temp.write_text(json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")
        temp.replace(self.output_dir / "report.json")
        (self.output_dir / "report.md").write_text(
            render_report(self.report, self.report["resource_usage"]), encoding="utf-8")

    def _close(self):
        for name in ("_samples", "_events"):
            stream = getattr(self, name, None)
            if stream is not None and not stream.closed:
                stream.close()
        if self.collector is not None:
            try:
                self.collector.close()
            except Exception:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.stop("failed" if exc is not None else "completed",
                  f"{exc_type.__name__}: {exc}" if exc is not None else None)
        return False


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
