"""Pure functions; all load fractions are in [0, 1]."""

import math

WEIGHTS = {"gpu": 0.4, "cpu": 0.3, "mem": 0.3}


def mapped_score(value, threshold, failed=False):
    if threshold is None:
        return None
    if failed or value is None or not math.isfinite(value) or value < 0:
        return 0.0
    return 1.0 if value <= threshold else math.exp(-(value - threshold))


class ResourceIntegral:
    """Right-endpoint integration; CPU readings represent their full interval."""

    def __init__(self, averaging="complete_time_weighted"):
        if averaging not in {"complete_time_weighted", "available_time_weighted"}:
            raise ValueError("Unsupported resource averaging policy")
        self.averaging = averaging
        self.last_t = 0.0
        self.area = dict.fromkeys(WEIGHTS, 0.0)
        self.coverage = dict.fromkeys(WEIGHTS, 0.0)
        self.count = 0
        self.max_interval_s = 0.0

    def add(self, t, values):
        if not math.isfinite(t) or t <= self.last_t:
            raise ValueError("sample timestamps must strictly increase")
        dt = t - self.last_t
        for key in WEIGHTS:
            value = values.get(key)
            if value is not None and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
                self.area[key] += dt * value
                self.coverage[key] += dt
        self.count += 1
        self.max_interval_s = max(self.max_interval_s, dt)
        self.last_t = t

    def result(self, sim_s, steps, status, config):
        wall = self.last_t
        complete = wall > 0 and all(abs(self.coverage[k] - wall) < 1e-9 for k in WEIGHTS)
        means = {}
        for k in WEIGHTS:
            denominator = self.coverage[k] if self.averaging == "available_time_weighted" else wall
            valid = denominator > 0 and (self.averaging == "available_time_weighted"
                                        or abs(self.coverage[k] - wall) < 1e-9)
            means[k] = self.area[k] / denominator if valid else None
        rtf = wall / sim_s if sim_s > 0 and wall > 0 else None
        cres = (sum(WEIGHTS[k] * means[k] for k in WEIGHTS) * rtf
                if rtf is not None and all(v is not None for v in means.values()) else None)
        failed = status != "completed"
        scores = {
            "rtf": mapped_score(rtf, config.rtf_threshold, failed),
            "cres": mapped_score(cres, config.cres_threshold, failed),
        }
        return {
            "wall_time_s": wall, "sim_time_s": sim_s, "physics_steps": steps,
            "physics_steps_per_wall_second": steps / wall if wall > 0 else None,
            "rtf": rtf, "simulation_speed": 1 / rtf if rtf else None,
            "mean_utilization": means, "cres": cres,
            "resource_integral_seconds": dict(self.area),
            "resource_coverage_seconds": dict(self.coverage),
            "sample_count": self.count, "max_sample_interval_s": self.max_interval_s,
            "resource_data_complete": complete,
            "mapped_scores": scores,
            "observation_points": {k: v * 15 if v is not None else None for k, v in scores.items()},
            "efficiency_Ec": (sum(scores.values()) / 2
                              if all(v is not None for v in scores.values()) else None),
        }
