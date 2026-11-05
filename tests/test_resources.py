import sys
from types import SimpleNamespace as NS

import psutil
import pytest
from sim_efficiency_monitor.resources import ProcessCollector


class Process:
    def __init__(self, pid, created=1, cpu=0, rss=100):
        self.pid, self.created, self.cpu, self.rss = pid, created, cpu, rss
        self.alive, self.descendants = True, []
        self.denied = False

    def create_time(self): return self.created
    def is_running(self): return self.alive
    def children(self, recursive): return self.descendants
    def cpu_times(self):
        if self.denied: raise psutil.AccessDenied(self.pid)
        return NS(user=self.cpu, system=0)
    def memory_info(self):
        if self.denied: raise psutil.AccessDenied(self.pid)
        return NS(rss=self.rss)


class FakeNVML:
    def nvmlInit(self): pass
    def nvmlShutdown(self): pass
    def nvmlDeviceGetCount(self): return 2
    def nvmlDeviceGetHandleByIndex(self, i): return i
    def nvmlDeviceGetUUID(self, h): return f"GPU-{h}"
    def nvmlDeviceGetName(self, h): return "test GPU"
    def nvmlDeviceGetMemoryInfo(self, h): return NS(total=1000, used=999)
    def nvmlSystemGetDriverVersion(self): return "test"
    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [NS(pid=10, usedGpuMemory=20), NS(pid=11, usedGpuMemory=30),
                NS(pid=999, usedGpuMemory=900)]
    def nvmlDeviceGetGraphicsRunningProcesses(self, h):
        return [NS(pid=10, usedGpuMemory=20), NS(pid=12, usedGpuMemory=10)]
    def nvmlDeviceGetProcessUtilization(self, h, since):
        return [NS(pid=pid, timeStamp=since+100, smUtil=u) for pid, u in
                [(10, 10+40*h), (11, 10), (999, 90)]]


@pytest.fixture
def setup(monkeypatch):
    root, child, graphics = Process(10, cpu=10), Process(11, cpu=20), Process(12)
    root.descendants = [child, graphics]
    monkeypatch.setattr(psutil, "Process", lambda pid: root)
    monkeypatch.setattr(psutil, "cpu_count", lambda: 4)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: NS(total=1000, available=1))
    monkeypatch.setitem(sys.modules, "pynvml", FakeNVML())
    clock = NS(mono=100.0, wall=100.0)
    monkeypatch.setattr("sim_efficiency_monitor.resources.time.perf_counter", lambda: clock.mono)
    monkeypatch.setattr("sim_efficiency_monitor.resources.time.time", lambda: clock.wall)
    collector = ProcessCollector(pid=10)
    collector.prime()
    clock.mono = clock.wall = 101.0
    yield collector, root, child, clock
    collector.close()


def test_only_process_tree_cpu_ram_and_gpu(setup):
    collector, root, child, _ = setup
    root.cpu += 1
    child.cpu += 1
    result = collector.sample()
    assert result["cpu"] == .5
    assert result["mem"] == .3
    assert result["memory_used_bytes"] == 300
    assert result["process_ids"] == [10, 11, 12]
    assert result["gpu"] == pytest.approx(.4)
    assert result["gpus"][0]["memory_used_bytes"] == 60
    assert collector.metadata["scope"] == "process_tree"


def test_new_child_uses_only_lifetime_cpu(setup):
    collector, root, _, _ = setup
    root.descendants.append(Process(13, created=100.5, cpu=1, rss=50))
    result = collector.sample()
    assert result["cpu"] == .25
    assert result["memory_used_bytes"] == 350


def test_exited_child_marks_cpu_interval_missing_then_recovers(setup):
    collector, root, child, clock = setup
    child.alive = False
    root.descendants.remove(child)
    assert collector.sample()["cpu"] is None
    clock.mono += 1
    clock.wall += 1
    assert collector.sample()["cpu"] == 0


def test_access_denied_is_missing_not_zero(setup):
    collector, _, child, _ = setup
    child.denied = True
    result = collector.sample()
    assert result["cpu"] is None
    assert result["mem"] is None
    assert result["memory_used_bytes"] is None


def test_one_gpu_failure_preserves_vram_and_invalidates_aggregate(setup):
    collector, *_ = setup
    def sample(handle, since):
        if handle == 1: raise RuntimeError("unsupported")
        return FakeNVML().nvmlDeviceGetProcessUtilization(handle, since)
    collector.nvml.nvmlDeviceGetProcessUtilization = sample
    result = collector.sample()
    assert result["gpu"] is None
    assert result["gpus"][0]["utilization"] == .2
    assert result["gpus"][1]["memory_used_bytes"] == 60
    assert any("unsupported" in e for e in result["errors"])


def test_unavailable_wddm_vram_does_not_use_device_total(setup):
    collector, *_ = setup
    collector.nvml.nvmlDeviceGetComputeRunningProcesses = lambda h: [NS(pid=10, usedGpuMemory=None)]
    result = collector.sample()
    assert result["gpu"] == pytest.approx(.4)
    assert result["gpus"][0]["memory_used_bytes"] is None


def test_no_fresh_gpu_data_is_missing(setup):
    collector, *_ = setup
    collector.nvml.nvmlDeviceGetProcessUtilization = lambda h, since: [NS(pid=10, timeStamp=since, smUtil=90)]
    assert collector.sample()["gpu"] is None


def test_gpu_multiple_windows_are_not_added_as_one(setup):
    collector, *_ = setup
    collector.nvml.nvmlDeviceGetProcessUtilization = lambda h, since: [
        NS(pid=10, timeStamp=since+1, smUtil=60),
        NS(pid=11, timeStamp=since+1, smUtil=80),
        NS(pid=10, timeStamp=since+2, smUtil=20),
    ]
    assert collector.sample()["gpu"] == pytest.approx(.6)
