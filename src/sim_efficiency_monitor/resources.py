"""Resources attributable to the simulation process and its descendants."""

import os
import platform
import time


class ProcessCollector:
    def __init__(self, gpu_devices=("all",), *, pid=None):
        import psutil
        self.psutil = psutil
        self.root = psutil.Process(os.getpid() if pid is None else pid)
        self.root_identity = (self.root.pid, self.root.create_time())
        self._tracked = {self.root_identity: self.root}
        self._previous_cpu = {}
        self._previous_time = None
        self._previous_wall = None
        self._gpu_since = {}
        self.nvml = None
        self.handles = []
        self.gpu_error = None
        self.metadata = {
            "scope": "process_tree", "root_pid": self.root.pid,
            "root_create_time": self.root_identity[1], "include_children": True,
            "cpu_measurement": "process_user_system_seconds / wall_seconds / logical_cpu_count",
            "memory_measurement": "sum_process_rss",
            "gpu_measurement": "NVML process SM utilization; per-window sum capped at 100%",
            "vram_measurement": "NVML compute/graphics process bytes, deduplicated by PID",
            "platform": platform.platform(), "python": platform.python_version(),
            "psutil": psutil.__version__, "logical_cpu_count": psutil.cpu_count(),
            "memory_total_bytes": psutil.virtual_memory().total, "gpu_devices": [],
        }
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            ids = range(pynvml.nvmlDeviceGetCount()) if tuple(gpu_devices) == ("all",) else gpu_devices
            seen = set()
            for identifier in ids:
                handle = (pynvml.nvmlDeviceGetHandleByIndex(int(identifier))
                          if str(identifier).isdigit() else pynvml.nvmlDeviceGetHandleByUUID(identifier))
                uuid = self._text(pynvml.nvmlDeviceGetUUID(handle))
                if uuid in seen:
                    raise ValueError(f"GPU selected more than once: {uuid}")
                seen.add(uuid)
                self.handles.append(handle)
                self.metadata["gpu_devices"].append({
                    "uuid": uuid, "name": self._text(pynvml.nvmlDeviceGetName(handle)),
                    "memory_total_bytes": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                })
            if not self.handles:
                raise RuntimeError("No NVIDIA GPUs found")
            self.metadata["nvidia_driver"] = self._text(pynvml.nvmlSystemGetDriverVersion())
        except Exception as exc:
            self.gpu_error = f"{type(exc).__name__}: {exc}"
            self.handles = []
            self.metadata["gpu_error"] = self.gpu_error

    @staticmethod
    def _text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    def _snapshot(self):
        errors = []
        discovery_ok = True
        # Retain discovered descendants even if they are later reparented.
        for identity, process in list(self._tracked.items()):
            try:
                if not process.is_running():
                    del self._tracked[identity]
                    continue
                for child in process.children(recursive=True):
                    self._tracked[(child.pid, child.create_time())] = child
            except self.psutil.NoSuchProcess:
                self._tracked.pop(identity, None)
            except self.psutil.Error as exc:
                discovery_ok = False
                errors.append(f"Process discovery PID {process.pid}: {exc}")
        cpu, rss, identities = {}, [], {}
        memory_ok = discovery_ok
        for identity, process in list(self._tracked.items()):
            try:
                if not process.is_running():
                    del self._tracked[identity]
                    continue
                identities[identity[0]] = identity[1]
                try:
                    times = process.cpu_times()
                    cpu[identity] = times.user + times.system
                except self.psutil.Error as exc:
                    errors.append(f"CPU PID {process.pid}: {exc}")
                    cpu[identity] = None
                try:
                    rss.append(process.memory_info().rss)
                except self.psutil.Error as exc:
                    memory_ok = False
                    errors.append(f"Memory PID {process.pid}: {exc}")
            except self.psutil.Error as exc:
                memory_ok = False
                errors.append(f"Process PID {process.pid}: {exc}")
        if self.root_identity not in self._tracked:
            discovery_ok = memory_ok = False
            errors.append("Simulation root process exited")
        return cpu, sum(rss) if memory_ok else None, identities, discovery_ok, errors

    def prime(self):
        self._previous_cpu, _, _, ok, _ = self._snapshot()
        if not ok:
            self._previous_cpu = {key: None for key in self._previous_cpu}
        self._previous_time = time.perf_counter()
        self._previous_wall = time.time()
        self._gpu_since = {g["uuid"]: int(self._previous_wall * 1e6)
                           for g in self.metadata["gpu_devices"]}

    def _gpu_sample(self, handle, info, identities, errors, discovery_ok):
        uuid = info["uuid"]
        result = {"uuid": uuid, "utilization": None, "memory_used_bytes": None,
                  "memory_total_bytes": info["memory_total_bytes"]}
        if not discovery_ok:
            return result
        try:
            # Both lists are needed to include graphics-only engines.
            rows = list(self.nvml.nvmlDeviceGetComputeRunningProcesses(handle))
            rows += list(self.nvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
            used = {}
            for row in rows:
                if row.pid not in identities:
                    continue
                value = row.usedGpuMemory
                if value is None or not isinstance(value, int) or value < 0 or value >= 2**64 - 1:
                    raise RuntimeError(f"Process VRAM unavailable for PID {row.pid}")
                used[row.pid] = max(used.get(row.pid, 0), value)
            result["memory_used_bytes"] = sum(used.values())
        except Exception as exc:
            errors.append(f"GPU {uuid} process VRAM: {type(exc).__name__}: {exc}")
        try:
            since = self._gpu_since[uuid]
            result["query_since_us"] = since
            rows = self.nvml.nvmlDeviceGetProcessUtilization(handle, since)
            windows = {}
            for row in rows:
                if row.timeStamp <= since:
                    continue
                window = windows.setdefault(row.timeStamp, {})
                if row.pid in identities and row.timeStamp >= int(identities[row.pid] * 1e6):
                    if not 0 <= row.smUtil <= 100:
                        raise RuntimeError("Invalid process SM utilization")
                    window[row.pid] = max(window.get(row.pid, 0), row.smUtil)
            if not windows:
                raise RuntimeError("No fresh process GPU utilization samples")
            self._gpu_since[uuid] = max(windows)
            result["sample_first_us"] = min(windows)
            result["sample_last_us"] = max(windows)
            result["driver_sample_count"] = len(windows)
            result["utilization"] = sum(min(100, sum(w.values())) for w in windows.values()) / (100 * len(windows))
        except Exception as exc:
            errors.append(f"GPU {uuid} process utilization: {type(exc).__name__}: {exc}")
        return result

    def sample(self):
        current, memory_used, identities, discovery_ok, errors = self._snapshot()
        gpu_scope_complete = discovery_ok and not (self._previous_cpu.keys() - current.keys())
        lost = self._previous_cpu.keys() - current.keys()
        if lost:
            errors.append(f"Processes missing since previous CPU baseline: {sorted(pid for pid, _ in lost)}")
        if not gpu_scope_complete:
            errors.append("Process GPU interval incomplete (process tree changed or inaccessible)")
        now, wall = time.perf_counter(), time.time()
        cpu = None
        if self._previous_time is not None and now > self._previous_time and discovery_ok:
            delta = 0.0
            valid = not (self._previous_cpu.keys() - current.keys())
            for identity, value in current.items():
                # A zero baseline is safe only for a process born during this interval.
                previous = self._previous_cpu.get(identity, 0.0 if identity[1] >= self._previous_wall else None)
                if value is None or previous is None or value < previous:
                    valid = False
                else:
                    delta += value - previous
            count = self.metadata["logical_cpu_count"]
            if valid and count:
                cpu = min(1.0, max(0.0, delta / (now - self._previous_time) / count))
        if cpu is None:
            errors.append("Process CPU interval incomplete (missing baseline, exited process or inaccessible counters)")
        self._previous_cpu = current if discovery_ok else {key: None for key in current}
        self._previous_time, self._previous_wall = now, wall
        total = self.metadata["memory_total_bytes"]
        mem = min(1.0, memory_used / total) if memory_used is not None and total else None
        per_gpu = [self._gpu_sample(handle, info, identities, errors, gpu_scope_complete)
                   for handle, info in zip(self.handles, self.metadata["gpu_devices"])]
        if not self.handles:
            errors.append(f"GPU: {self.gpu_error}")
        gpu = (sum(g["utilization"] for g in per_gpu) / len(per_gpu)
               if per_gpu and all(g["utilization"] is not None for g in per_gpu) else None)
        return {"cpu": cpu, "gpu": gpu, "mem": mem,
                "memory_used_bytes": memory_used, "memory_total_bytes": total,
                "process_ids": sorted(identities), "gpus": per_gpu, "errors": errors}

    def close(self):
        if self.nvml is not None:
            self.nvml.nvmlShutdown()


# Legacy import compatibility; collection now uses process_tree scope.
SystemCollector = ProcessCollector
