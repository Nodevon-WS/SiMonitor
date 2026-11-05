import argparse
import json
from pathlib import Path
import sys
import time

from .config import MonitorConfig
from .config import positive
from .metrics import ResourceIntegral
from .monitor import Monitor, _sha256
from .resources import ProcessCollector
from .reporting import render_report, summarize_resources


def verify(run_dir):
    """Detect accidental edits and independently recompute metrics; not a signature."""
    root = Path(run_dir)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report.get("schema_version") not in {"1.0", "1.1"}:
        raise ValueError("Unsupported report schema")
    averaging = "complete_time_weighted"
    if report["schema_version"] == "1.1":
        averaging = report.get("resource_averaging")
        if averaging != "available_time_weighted" or report.get("sampling_mode") != "simulation_step":
            raise ValueError("Unsupported measurement policy")
    if report.get("status") not in {"completed", "failed", "aborted", "timeout"}:
        raise ValueError("Invalid run status")
    config = MonitorConfig(**report["config"])
    if config.sha256 != report["config_sha256"]:
        raise ValueError("Configuration hash mismatch")
    for name in ("samples.jsonl", "events.jsonl"):
        if _sha256(root / name) != report["files_sha256"][name]:
            raise ValueError(f"File hash mismatch: {name}")
    events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    if (not events or events[0].get("event") != "start" or events[-1].get("event") != "stop"
            or events[-1].get("status") != report["status"]
            or events[0].get("config_sha256") != config.sha256):
        raise ValueError("Run events do not match report")
    if report["schema_version"] == "1.1" and (
            events[0].get("sampling_mode") != report["sampling_mode"] or
            events[0].get("resource_averaging") != averaging):
        raise ValueError("Measurement policy differs from start event")
    integral = ResourceIntegral(averaging)
    step_count = 0
    final_seen = False
    last = None
    with (root / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            positive(row["sim_time_s"], "sample sim_time_s", zero=True)
            if isinstance(row["physics_steps"], bool) or not isinstance(row["physics_steps"], int) or row["physics_steps"] < 0:
                raise ValueError("Invalid physics_steps")
            if report["schema_version"] == "1.1":
                if final_seen:
                    raise ValueError("Samples after final boundary")
                if row.get("sample_phase") == "step":
                    step_count += 1
                    if row["physics_steps"] != step_count or (last and row["sim_time_s"] <= last["sim_time_s"]):
                        raise ValueError("Expected one resource sample per completed step")
                elif row.get("sample_phase") == "final":
                    final_seen = True
                    if row["physics_steps"] != step_count or row["sim_time_s"] != (last["sim_time_s"] if last else 0):
                        raise ValueError("Final boundary must not advance physics")
                else:
                    raise ValueError("Invalid sample phase")
            integral.add(row["wall_time_s"], row)
            if last and (row["sim_time_s"] < last["sim_time_s"] or row["physics_steps"] < last["physics_steps"]):
                raise ValueError("Nonmonotonic simulation progress")
            last = row
    if report["schema_version"] == "1.1" and not final_seen:
        raise ValueError("Missing final boundary sample")
    if last is None:
        raise ValueError("No samples")
    if report["status"] == "completed" and (last["sim_time_s"] <= 0 or
            (config.wall_time_limit_s is not None and last["wall_time_s"] > config.wall_time_limit_s)):
        raise ValueError("Completed status contradicts run duration")
    expected = integral.result(last["sim_time_s"], last["physics_steps"], report["status"], config)
    if expected != report["metrics"]:
        raise ValueError("Metrics differ from raw samples")
    if ("resource_usage" in report and report["resource_usage"] !=
            summarize_resources(root, report["environment"], averaging)):
        raise ValueError("Resource usage differs from raw samples")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulation efficiency monitor")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check process resource availability")
    doctor.add_argument("--config")
    doctor.add_argument("--pid", type=int, help="Simulation PID (default: doctor process)")
    demo = commands.add_parser("demo", help="Synthetic integration smoke test; not a benchmark")
    demo.add_argument("--output", required=True)
    demo.add_argument("--config")
    demo.add_argument("--steps", type=int, default=120)
    demo.add_argument("--dt", type=float, default=1 / 60)
    demo.add_argument("--sleep", type=float, default=0.01)
    for name in ("verify", "report"):
        command = commands.add_parser(name)
        command.add_argument("run_dir")
        if name == "report":
            command.add_argument("--json", action="store_true", help="Print the stored machine-readable report")
    args = parser.parse_args(argv)
    try:
        config = MonitorConfig.from_file(args.config) if getattr(args, "config", None) else MonitorConfig()
        if args.command == "doctor":
            collector = ProcessCollector(config.gpu_devices, pid=args.pid)
            try:
                collector.prime()
                time.sleep(config.sample_interval_s)
                values = collector.sample()
                print(json.dumps({"environment": collector.metadata, "sample": values}, indent=2, ensure_ascii=False))
                return 0 if all(values[k] is not None for k in ("cpu", "gpu", "mem")) else 2
            finally:
                collector.close()
        if args.command == "demo":
            from .config import positive
            positive(args.steps, "steps")
            positive(args.dt, "dt")
            positive(args.sleep, "sleep", zero=True)
            with Monitor(args.output, config) as monitor:
                for _ in range(args.steps):
                    with monitor.step(args.dt):
                        time.sleep(args.sleep)
            print(render_report(monitor.report, monitor.report["resource_usage"]))
        elif args.command in {"verify", "report"}:
            report = verify(args.run_dir)
            if args.command == "verify":
                print("Verified")
            elif args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                resources = report.get("resource_usage")
                if resources is None:
                    resources = summarize_resources(args.run_dir, report["environment"],
                                                    report.get("resource_averaging", "complete_time_weighted"))
                print(render_report(report, resources))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f"sim-monitor: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
