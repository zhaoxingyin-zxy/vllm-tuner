# vLLM-Tuner TODO

## ✅ 已实现功能

### 基础设施
- **config.py** — YAML 配置加载，自动推导 `local_path`（`working_dir/models/<model_name>`）
- **remote_env.py** — fabric 持久 SSH 连接封装；支持 `run`、`run_background`（nohup）、`read_log_tail`

### 框架抽象（Phase 0 / 2a）
- **frameworks/base.py** — `InferenceFramework` ABC；`tunable_infra_params` 属性（block_size / gpu_memory_utilization / max_num_seqs）
- **frameworks/vllm_framework.py** — 构建 `python -m vllm.entrypoints.openai.api_server` 启动命令
- **frameworks/lmdeploy_framework.py** — `lmdeploy serve api_server` 启动命令
- **frameworks/sglang_framework.py** — `python -m sglang.launch_server` 启动命令
- **frameworks/__init__.py** — `get_framework(name, host, port)` 工厂函数

### 硬件监控
- **hardware/ascend.py** — `AscendObserver`：npu-smi 单次快照（usages / common / power / temp / health），正则解析 HBM、AICore%、功耗、温度
- **hardware/cuda.py** — `CUDAObserver`：nvidia-smi CSV 格式快照
- **hardware/__init__.py** — `get_observer(hw_type, ...)` 工厂函数

### 评估技能（EvalSkill）
- **skills/throughput.py** — `ThroughputSkill`：ThreadPoolExecutor 并发 HTTP 请求，返回平均 tokens/s
- **skills/latency.py** — `LatencySkill`：顺序请求，返回 `{p50_ms, p99_ms}`
- **skills/task_metric.py** — `TaskMetricSkill`：SSH 执行用户 eval_script，解析 `{"metric": float}` 标准输出
- **skills/memory.py** — `MemorySkill`：调用 `observer.get_stats().hbm_util_pct`

### 核心调优流水线
- **actor.py** — 服务生命周期：`stop_service`（lsof+ps 安全验证，拒绝 kill sshd）、`start_service`、`restart_service`、`fast_fail_check`（/health + 3次推理探测）、`_wait_for_health`（轮询超时）
- **brain.py** — Claude API 决策引擎：`decide_next_config`（读取硬件信号 + 历史，输出下一个参数配置）、`diagnose`（分析错误日志，给出修复命令）
- **reporter.py** — results.tsv 追加写入；`compute_config_hash`（MD5[:6] 去重）；`config_json` 列支持 `--use-best-config` 回读；`generate_best_report` 生成 best_config.md
- **orchestrator.py** — Phase 2a（重启服务，双目标 KEEP：throughput OR latency 改善 + 另一个不回退 1%）；Phase 2b（不重启，单指标 KEEP）；patience 收敛；自愈 3 次重试（命令结构性去重）
- **runner.py** — 顺序执行 EvalSkill 列表，归一化为 `{throughput, latency_p99, task_metric, memory_pct}`
- **analysis.py** — Pareto 前沿（`is_pareto_dominated`，支持 maximize/minimize）；`get_pareto_frontier`（仅取 Phase 2b 行）；`select_bests`（Phase 2a 最优延迟 + 最高吞吐）

### 部署与摸底（Phase 0 / 1）
- **shipit.py** — `check_remote_env`（HBM / health / vLLM 安装 / 磁盘 / 端口 / 模型目录）；`auto_fix_env`；`pull_model`（huggingface-cli download）；`_clear_port`（安全验证进程名）；`self_heal`（3 次诊断重试，结构性命令去重）
- **sweep.py** — `BenchmarkSweep`：concurrency × input_len 矩阵，OOM 检测（health check 失败标记 OOM，跳过后续更大 input_len），`_recommend` 返回 max_num_seqs / block_size 建议

### CLI 入口
- **main.py** — `run` 子命令（完整 5 阶段流水线）；`deploy` 子命令（仅 Phase 0）；`--use-best-config`：从 best_config.md 解析 config_hash，查 results.tsv `config_json` 列还原最优 infra 参数

---

## 🐛 待解决的 Bug

### 1. Python 版本兼容性（高优先级）
- **现象**：部分模块在编写时使用了 Python 3.10+ 语法（`float | dict`、`list[int]`、`dict | None`），在项目实际运行的 Python 3.9.11 环境报 `TypeError: unsupported operand type(s) for |`
- **已修复**：`skills/base.py`、`reporter.py`、`orchestrator.py`、`main.py`、`analysis.py`、`sweep.py` 已添加 `from __future__ import annotations` 或改为兼容写法
- **待排查**：其余模块（`actor.py`、`brain.py` 等）如在 Python 3.9 环境导入时报类似错误，需同样处理

### 2. 依赖环境不一致（中优先级）
- **现象**：项目同时存在 Python 3.9.11（PATH 中第一个）和 Miniconda Python 3.12 两个解释器；`pip` 指向 Miniconda，导致 `pyyaml`、`fabric`、`anthropic`、`requests` 等包安装到错误环境
- **临时解法**：已用 `/c/Users/16500/AppData/Local/Programs/Python/Python39/python.exe -m pip install` 为 Python 3.9 单独安装
- **根本解法**：建议统一使用虚拟环境（`python -m venv .venv`），或在 `requirements.txt` 中锁定版本后一次性安装

### 3. `.gitignore` 缺少 IDE 目录
- **现象**：`.idea/`（JetBrains IDE 配置目录）未被忽略，出现在 `git status` 未追踪文件中
- **已修复**：已加入 `.gitignore`

### 4. `_wait_for_health` 中的 `time.sleep` 阻塞
- **现象**：`actor.py` 的 `_wait_for_health` 每秒轮询一次，最长等待 `health_timeout`（默认 120s）；在 NPU 内存不足或 CANN 环境未激活时，服务永远不会健康，导致 120s 超时才报错
- **建议**：在轮询中同时检查日志是否出现明确失败关键词（OOM、Segfault、Error）以快速失败

### 5. `BenchmarkSweep` 的 OOM 判断可能误判
- **现象**：`_measure_cell` 在 future 抛异常后立即调用 `_is_healthy()`；若网络抖动导致请求超时但服务仍健康，会被误标为 OOM
- **建议**：增加 2 次 health 重试，连续失败才判定 OOM

---

## 🗓 下一步计划

### 近期（功能完善）
1. **端到端集成测试**：搭建 mock vLLM 服务（或使用已部署的接口），跑一次完整 `run` 流水线验证联动
2. **`demo/eval_stub.py`**：补全示例 eval 脚本，满足 eval_script contract（接收 `--server-url / --data-dir / --sample-size`，输出 `{"metric": float}`）
3. **虚拟环境 & requirements.txt 锁版**：统一 Python 3.9 环境，`pip freeze` 输出固定版本，消除多解释器冲突
4. **`generate_best_report` 图表**：用 matplotlib 生成 throughput vs task_metric 的 Pareto 散点图，附在 best_config.md 旁

### 中期（鲁棒性）
5. **ShipIt 与远程实际测试**：在真实 Ascend NPU 机器上验证 `npu-smi` 命令解析、`huggingface-cli download`、端口清理的实际行为
6. **`--skip-shipit` / `--skip-sweep` 命令行选项**：当用户已有部署好的接口时，直接传入 `--server-url` 跳过 Phase 0 和 Phase 1
7. **多卡支持**：`npu-smi` 目前仅取单 NPU 信息；扩展为多 NPU 聚合统计
8. **断点续传**：从已有 results.tsv 恢复 `seen_hashes`，支持中断后继续调优

### 长期（生态扩展）
9. **Web UI**：实时展示各轮指标变化趋势（WebSocket + 简单前端）
10. **多节点支持**：扩展 `RemoteEnv` 支持多机 SSH，`BenchmarkSweep` 跨节点测压
11. **自动 CANN 版本检测**：在 `check_remote_env` 中检测 CANN 版本与 vLLM-Ascend 的兼容矩阵
