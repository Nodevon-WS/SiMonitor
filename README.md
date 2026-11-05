# sim-efficiency-monitor

轻量 Python 仿真效率监控工具。每个 simulation step 完成后查询仿真进程及其子进程的 CPU、GPU 和内存占用，输出平均占用率、RTF、Cres 与可复核日志。

支持 Python 3.10+、Windows / Linux。CPU 与主存使用 psutil，NVIDIA GPU 使用 NVML。

## 安装与试运行

在项目根目录执行：

```bash
python -m pip install ".[nvidia]"
sim-monitor doctor
sim-monitor demo --output runs/demo-001
sim-monitor verify runs/demo-001
sim-monitor report runs/demo-001
```

没有 NVIDIA GPU 时可安装 `python -m pip install .`；GPU 平均值和 Cres 将为 `null`。`doctor` 返回 0 表示此次三项资源可读，2 表示至少一项不可用。`demo` 用 sleep 模拟物理步，只用于验证接入。每次运行使用新的输出目录。

## 接入仿真

先完成场景加载、预热与初始状态恢复，再开始监控。以下 `engine` 替换为实际引擎对象：

```python
from sim_efficiency_monitor import Monitor, MonitorConfig

config = MonitorConfig(engine="my-engine", gpu_devices=("0",))
with Monitor("runs/simulation-001", config) as monitor:
    for _ in range(600):
        engine.step()
        monitor.advance(dt_s=1 / 60)  # 每个已完成物理步立即查询资源

print(monitor.report["metrics"]["mean_utilization"])
print(monitor.report["metrics"]["cres"])
```

也可使用 `with monitor.step(dt): engine.step()`，或 `record_sim_time(engine.current_time)` 上报物理时间戳；非零初始时钟需先调用 `monitor.start(sim_time_s=initial_time)`。每步必须单独上报，`steps` 只允许为 1，不计入仅渲染或暂停的帧。

异步引擎须在上报前等待该步完成。可向 `Monitor` 传入 `synchronize=engine.synchronize` 处理开始和结束边界，但该回调不会自动在每步执行。外部控制进程可用 `pid=simulation_pid` 指定被监控进程。

[Isaac Sim 示例](examples/isaac_sim_standalone.py) 可在已安装本包的 Isaac Sim Python 环境运行：

```bash
python examples/isaac_sim_standalone.py --output runs/isaac-001
```

## 指标与输出

| 指标 | 定义 |
| --- | --- |
| 平均占用率 | 各资源有效区间的时间加权平均，比例为 0～1 |
| RTF | 墙钟时间 / 仿真时间，越小越好 |
| Cres | `(0.4 × GPU均值 + 0.3 × CPU均值 + 0.3 × 内存均值) × RTF` |
| 物理 FPS | 已完成物理步数 / 墙钟秒 |

每步查询一次，结束时额外查询一次，因此 N 步产生 N+1 条采样。查询与日志开销计入墙钟时间，短步骤可能明显受监控影响。

GPU 驱动不保证每次查询都有新数据。缺失区间不补零，各资源分别用有效覆盖时长作为平均值分母；某项全程缺失时 Cres 仍为 `null`。部分覆盖的 Cres 是估计值，报告通过 `resource_data_complete` 和覆盖时长说明数据完整性。Windows 可能无法读取进程显存，不影响单独可用的 GPU 利用率。

每次运行生成 `samples.jsonl`（逐步数据）、`events.jsonl`（开始/结束）、`report.json`（完整结果）与 `report.md`（可读报告）。`sim-monitor report RUN_DIR --json` 输出 JSON；`verify` 校验哈希并从日志重算结果。

可选配置见 [development.json](configs/development.json)，通过 `--config` 传入 CLI，或用 `MonitorConfig.from_file()` 加载。`sample_interval_s` 仅用于 doctor 等待和旧配置兼容，不控制逐步采样。评分阈值默认未设置，`efficiency_Ec` 为 `null`；设置阈值后才计算评分。计量边界、缺失规则及评分公式见 [计量说明](docs/measurement.md)。

## 开发与发布

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
```

测试不需要真实 GPU 或仿真引擎，CI 在 Windows / Python 3.12 与 Linux / Python 3.10 上执行测试和构建。实际引擎与 GPU 驱动兼容性需在目标环境验证。

在 GitHub 创建空仓库，将 `OWNER/REPO` 替换为目标地址后执行：

```bash
git init
git add .
git commit -m "Initial release"
git branch -M main
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

GitHub Release 可附上 `dist/` 中的 wheel 和源码发行包。`.gitignore` 已排除运行记录、虚拟环境和构建缓存。

许可证：[MIT](LICENSE)。
