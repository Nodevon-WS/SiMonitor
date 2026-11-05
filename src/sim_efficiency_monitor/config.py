from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


@dataclass(frozen=True)
class MonitorConfig:
    scene_id: str = "development"
    team_id: str = "anonymous"
    engine: str = "unspecified"
    engine_version: str = "unspecified"
    sample_interval_s: float = 0.2
    gpu_devices: tuple[str, ...] = ("all",)
    rtf_threshold: float | None = None
    cres_threshold: float | None = None
    wall_time_limit_s: float | None = None

    def __post_init__(self):
        for name in ("scene_id", "team_id", "engine", "engine_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        positive(self.sample_interval_s, "sample_interval_s")
        if self.sample_interval_s < 0.01:
            raise ValueError("sample_interval_s must be >= 0.01")
        for name in ("rtf_threshold", "cres_threshold"):
            if getattr(self, name) is not None:
                positive(getattr(self, name), name, zero=True)
        if self.wall_time_limit_s is not None:
            positive(self.wall_time_limit_s, "wall_time_limit_s")
        if not isinstance(self.gpu_devices, (list, tuple)):
            raise ValueError("gpu_devices must be a list of NVML indices or UUIDs")
        devices = tuple(self.gpu_devices)
        if not devices or any(not isinstance(x, str) or not x for x in devices):
            raise ValueError("gpu_devices must contain nonempty strings")
        if len(set(devices)) != len(devices) or ("all" in devices and devices != ("all",)):
            raise ValueError("duplicate or ambiguous gpu_devices")
        object.__setattr__(self, "gpu_devices", devices)

    def to_dict(self):
        result = asdict(self)
        result["gpu_devices"] = list(self.gpu_devices)
        return result

    @property
    def sha256(self):
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("configuration must be a JSON object")
        return cls(**data)
