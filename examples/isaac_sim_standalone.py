"""CPU standalone integration example; run with Isaac Sim's Python launcher."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="runs/isaac-cpu-001")
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    import numpy as np
    from sim_efficiency_monitor import Monitor, MonitorConfig

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, device="cpu")
    world.scene.add_default_ground_plane()
    world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube",
                                 position=np.array([0.0, 0.0, 1.0])))
    world.reset()
    for _ in range(60):
        world.step(render=False)
    world.reset()
    config = MonitorConfig(scene_id="integration-cube", engine="Isaac Sim",
                           engine_version="6.0.0")
    monitor = Monitor(args.output, config)
    monitor.start(sim_time_s=float(world.current_time))
    try:
        for _ in range(600):
            if not app.is_running():
                raise RuntimeError("Simulation application stopped early")
            world.step(render=False)
            monitor.record_sim_time(float(world.current_time))
    except BaseException as exc:
        monitor.stop("failed", reason=f"{type(exc).__name__}: {exc}")
        raise
    else:
        monitor.stop()
finally:
    app.close()
