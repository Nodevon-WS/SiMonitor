"""Resource summaries and a shared human-readable report."""

import json
import math
from pathlib import Path


def summarize_resources(run_dir, environment, averaging="complete_time_weighted"):
    """Integrate recorded samples; keep partial coverage explicit."""
    if averaging not in {"complete_time_weighted", "available_time_weighted"}:
        raise ValueError("Unsupported resource averaging policy")
    stats = {}
    devices = {g["uuid"]: dict(g) for g in environment.get("gpu_devices", [])}
    total_memory = environment.get("memory_total_bytes")
    previous = 0.0

    def add(key, value, dt, upper=None):
        stat = stats.setdefault(key, {"area": 0.0, "coverage": 0.0, "peak": None})
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (upper is not None and value > upper)):
            return
        stat["area"] += value * dt
        stat["coverage"] += dt
        stat["peak"] = value if stat["peak"] is None else max(stat["peak"], value)

    with (Path(run_dir) / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            t = row["wall_time_s"]
            dt = t - previous
            if not math.isfinite(t) or dt <= 0:
                raise ValueError("sample timestamps must strictly increase")
            for key in ("cpu", "gpu", "mem"):
                add(key, row.get(key), dt, 1)
            # Legacy logs only contain the fraction and fixed host capacity.
            used = row.get("memory_used_bytes")
            if "memory_used_bytes" not in row and total_memory and row.get("mem") is not None:
                used = row["mem"] * total_memory
            add("memory_used_bytes", used, dt)
            for gpu in row.get("gpus", []):
                uuid = gpu["uuid"]
                devices.setdefault(uuid, {"uuid": uuid, "memory_total_bytes": gpu.get("memory_total_bytes")})
                add((uuid, "utilization"), gpu.get("utilization"), dt, 1)
                add((uuid, "memory_used_bytes"), gpu.get("memory_used_bytes"), dt)
            previous = t

    def result(key):
        stat = stats.get(key, {"area": 0.0, "coverage": 0.0, "peak": None})
        denominator = stat["coverage"] if averaging == "available_time_weighted" else previous
        valid = denominator > 0 and (averaging == "available_time_weighted"
                                    or abs(stat["coverage"] - previous) < 1e-9)
        return {
            "mean": stat["area"] / denominator if valid else None,
            "sampled_peak": stat["peak"],
            "coverage_seconds": stat["coverage"],
        }

    return {
        "cpu": result("cpu"), "gpu": result("gpu"), "mem": result("mem"),
        "memory_used_bytes": result("memory_used_bytes"),
        "memory_total_bytes": total_memory,
        "gpus": [{**device, "utilization": result((uuid, "utilization")),
                  "memory_used_bytes": result((uuid, "memory_used_bytes"))}
                 for uuid, device in sorted(devices.items())],
    }


def render_report(report, resources):
    def number(value):
        return "null" if value is None else f"{value:.4f}"

    def percent(value):
        return "null" if value is None else f"{value * 100:.2f}%"

    def gib(value):
        return "null" if value is None else f"{value / 1024**3:.3f} GiB"

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    m = report["metrics"]
    scope = report.get("measurement_scope", report.get("environment", {}).get("scope", "whole_machine"))
    process_scope = scope == "process_tree"
    scope_label = "simulation process tree" if process_scope else scope.replace("_", " ")
    ram_label = "RAM used (process RSS sum)" if process_scope else "RAM used (total - available)"
    lines = ["# Simulation efficiency report", "", f"Status: {report['status']}", "",
             "| Metric | Value |", "| --- | --- |"]
    for key in ("wall_time_s", "sim_time_s", "physics_steps_per_wall_second", "rtf", "cres", "efficiency_Ec"):
        lines.append(f"| {key} | {number(m[key])} |")
    lines += ["", f"## Resource usage ({scope_label})", "",
              "| Resource | Time-weighted mean | Sampled peak | Coverage (s) |",
              "| --- | --- | --- | --- |"]
    for key, label, fmt in (("cpu", "CPU utilization", percent),
                            ("gpu", "GPU utilization (selected devices, equal mean)", percent),
                            ("mem", "RAM utilization", percent),
                            ("memory_used_bytes", ram_label, gib)):
        stat = resources[key]
        lines.append(f"| {label} | {fmt(stat['mean'])} | {fmt(stat['sampled_peak'])} | {number(stat['coverage_seconds'])} |")
    lines += ["", f"Host RAM capacity: {gib(resources['memory_total_bytes'])}"]
    if process_scope:
        lines += [f"Simulation root PID: {report.get('environment', {}).get('root_pid', 'unknown')} (including descendants)"]
    lines += ["", "## Per-GPU usage", ""]
    if resources["gpus"]:
        lines += ["| GPU / UUID | Mean utilization | Sampled peak | Mean VRAM used | Sampled peak VRAM | Total VRAM | Utilization / VRAM coverage (s) |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for gpu in resources["gpus"]:
            u, v = gpu["utilization"], gpu["memory_used_bytes"]
            label = cell(f"{gpu.get('name', 'GPU')} / {gpu['uuid']}")
            lines.append(f"| {label} | {percent(u['mean'])} | {percent(u['sampled_peak'])} | {gib(v['mean'])} | {gib(v['sampled_peak'])} | {gib(gpu.get('memory_total_bytes'))} | {number(u['coverage_seconds'])} / {number(v['coverage_seconds'])} |")
    else:
        lines.append("GPU data unavailable (null).")
    if report.get("resource_averaging") == "available_time_weighted":
        lines += ["", "Resources are queried after every reported simulation step, plus a final boundary sample.",
                  "Means are time-weighted over each resource's valid intervals; missing intervals are excluded, not filled with zero.",
                  f"Complete resource coverage: {m['resource_data_complete']}. Cres uses these available-data means; partial coverage makes it an estimate."]
    else:
        lines += ["", "Means require complete coverage."]
    lines += ["Peaks are maxima of available samples, not continuous-time peaks.",
              ("Only the simulation process tree is counted, including in-process monitoring overhead. CPU is normalized by host logical CPUs; RAM is the RSS sum (shared pages may be counted more than once; fraction capped at 100%). GPU uses process SM samples on selected devices; unavailable process counters remain null."
               if process_scope else "CPU and RAM include all host processes; GPU covers selected physical devices. VRAM is separate from RAM."),
              "Lower RTF and Cres are better. Missing data is null, never zero.",
              "Scores require organizer-supplied thresholds. Ec is not the overall competition score.",
              "", "See report.json and samples.jsonl for configuration, coverage and raw records."]
    return "\n".join(lines) + "\n"
