import math
import pytest

from sim_efficiency_monitor import MonitorConfig
from sim_efficiency_monitor.metrics import ResourceIntegral


def test_time_weighted_not_sample_weighted():
    samples = ResourceIntegral()
    samples.add(1, {"cpu": 0.2, "gpu": 0.5, "mem": 0.4})
    samples.add(4, {"cpu": 0.6, "gpu": 1.0, "mem": 0.8})
    result = samples.result(2, 120, "completed", MonitorConfig(rtf_threshold=1, cres_threshold=1))
    assert result["rtf"] == 2
    assert result["mean_utilization"] == pytest.approx({"cpu": .5, "gpu": .875, "mem": .7})
    assert result["cres"] == pytest.approx(1.42)
    assert result["mapped_scores"]["rtf"] == pytest.approx(math.exp(-1))
    assert result["mapped_scores"]["cres"] == pytest.approx(math.exp(-.42))
    assert result["physics_steps_per_wall_second"] == 30


def test_missing_interval_is_not_zero_or_rescaled():
    samples = ResourceIntegral()
    samples.add(1, {"cpu": .5, "gpu": None, "mem": .5})
    samples.add(2, {"cpu": .5, "gpu": 1., "mem": .5})
    result = samples.result(2, 120, "completed", MonitorConfig(rtf_threshold=1, cres_threshold=1))
    assert result["cres"] is None
    assert result["mean_utilization"]["gpu"] is None
    assert result["resource_coverage_seconds"]["gpu"] == 1
    assert result["mapped_scores"] == {"rtf": 1, "cres": 0}
    assert result["efficiency_Ec"] == .5


@pytest.mark.parametrize("bad", [float("nan"), 1.1, True])
def test_invalid_utilization_is_missing(bad):
    samples = ResourceIntegral()
    samples.add(1, {"cpu": bad, "gpu": 0., "mem": 0.})
    assert samples.result(1, 1, "completed", MonitorConfig())["cres"] is None


@pytest.mark.parametrize("kwargs", [
    {"sample_interval_s": 0}, {"cres_threshold": float("nan")}, {"gpu_devices": ["all", "0"]},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        MonitorConfig(**kwargs)


def test_available_intervals_are_time_weighted_not_zero_filled():
    samples = ResourceIntegral("available_time_weighted")
    samples.add(1, {"cpu": .2, "gpu": None, "mem": .4})
    samples.add(2, {"cpu": .4, "gpu": .2, "mem": .4})
    samples.add(5, {"cpu": .6, "gpu": .8, "mem": .4})
    result = samples.result(10, 3, "completed", MonitorConfig())
    assert result["mean_utilization"] == pytest.approx({"cpu": .48, "gpu": .65, "mem": .4})
    assert result["cres"] == pytest.approx(.262)
    assert result["resource_coverage_seconds"]["gpu"] == 4
    assert result["resource_data_complete"] is False
    missing = ResourceIntegral("available_time_weighted")
    missing.add(1, {"cpu": .1, "gpu": None, "mem": .2})
    assert missing.result(1, 1, "completed", MonitorConfig())["cres"] is None
