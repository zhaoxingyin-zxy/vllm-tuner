# vLLM-Tuner 设计文档

**日期：** 2026-03-22
**状态：** 待实现
**定位：** 唯一一个在昇腾 NPU 上以任务准确率为优化目标、联合调优推理框架参数与生成参数的全自动 AI 推理调优工具

---

## 1. 项目背景与价值定位

### 问题陈述

现有 vLLM 调优工具（`auto_tune.sh`、`openshift-psap/auto-tuning-vllm`、TokenPeak）存在三个共同盲区：

1. **仅优化服务指标**——只关注 throughput / latency，从不将任务准确率纳入优化目标
2. **不调生成参数**——`temperature`、`top_p`、`top_k`、`repetition_penalty` 对下游任务准确率影响极大，无工具覆盖
3. **不支持昇腾 NPU**——所有主流工具均为 CUDA-first，`vllm-ascend` 于 2025 年 2 月才创建，调优生态完全空白

### 核心价值主张

> 给定一个 HuggingFace 模型链接、一个评测脚本与数据集、一台远程机器——vLLM-Tuner 自动完成从裸机部署到最优推理配置的全闭环，输出三份最优报告：最低延迟、最大吞吐、最高准确率。

### 竞争差异化

| 能力 | vLLM auto_tune.sh | openshift-psap | vLLM-Tuner |
|------|:-----------------:|:--------------:|:----------:|
| 任务准确率作为优化目标 | ❌ | ❌ | ✅ |
| 生成参数调优 | ❌ | ❌ | ✅ |
| 昇腾 NPU 支持 | ❌ | ❌ | ✅ |
| 自动部署（裸机→服务） | ❌ | ❌ | ✅ |
| 因果推理决策（非统计） | ❌ | ❌ | ✅ |
| 自定义评测脚本 | ❌ | 部分 | ✅ |
| SSH 远程管理 | ❌ | ❌ | ✅ |

---

## 2. 系统架构

### 2.1 五阶段编排流程

```
输入：
  hf_model_url   → HuggingFace 模型链接
  eval_script    → 用户评测脚本（远程机路径）
  eval_data      → 评测数据集路径（远程机路径）
  metric         → 评测指标名称及方向（maximize/minimize）
  tuner_config   → YAML 配置文件

Phase 0: Ship-It（可独立使用）
  检查环境 → 修复问题 → 拉取模型 → 部署框架 → 验证服务

Phase 1: Sweep 性能摸底
  单实例下，并发数 × 输入长度 → 性能矩阵 → 推荐搜索空间
  （注：Sweep 全程不重启服务，仅变化请求参数）

Phase 2a: 速度/吞吐优化
  固定默认生成参数，调 vLLM 启动参数，多轮迭代
  每轮需重启服务（启动参数变更）

Phase 2b: 准确率优化
  固定最优基础设施参数，调生成参数，多轮迭代
  每轮无需重启（生成参数通过 API 传入）

Phase 3: 三份最优报告 + 收尾
  最低延迟 / 最大吞吐 / 最高准确率 / Pareto 综合推荐
  服务保持运行（用最优配置，供直接使用）
```

### 2.2 文件结构

```
vllm_tuner/
├── main.py                    # CLI 入口，五阶段编排
├── shipit.py                  # Phase 0：自动部署 Agent
├── sweep.py                   # Phase 1：性能摸底
├── orchestrator.py            # Phase 2：优化循环控制器
├── brain.py                   # Claude API：因果推理决策 + 自愈诊断
├── remote_env.py              # SSH 连接基础层（fabric.Connection）
├── actor.py                   # 服务生命周期管理
├── runner.py                  # 在单次评测内并行执行 EvalSkills
├── reporter.py                # results.tsv + 最终报告生成
│
├── frameworks/                # 推理框架抽象层
│   ├── base.py                # InferenceFramework 抽象类
│   ├── vllm_framework.py      # 主角，昇腾场景
│   ├── lmdeploy_framework.py  # 昇腾生态替代
│   └── sglang_framework.py    # CUDA 场景
│
├── hardware/                  # 硬件监控抽象层
│   ├── base.py                # HardwareObserver 抽象类
│   ├── ascend.py              # npu-smi 单次快照命令
│   └── cuda.py                # nvidia-smi
│
├── skills/                    # 评测技能插件
│   ├── base.py                # EvalSkill 抽象类
│   ├── task_metric.py         # 调用用户 eval_script，支持任意指标
│   ├── throughput.py          # 并发压测，线程池
│   ├── latency.py             # P50/P99 延迟
│   └── memory.py              # 显存/HBM 使用率
│
├── analysis.py                # Pareto 分析 + frontier 演进可视化
├── tuner_config.yaml          # 用户配置文件（含 HF token，加入 .gitignore）
└── .gitignore                 # results.tsv, results/, tuner_config.yaml 不入 git
```

---

## 3. 各模块详细设计

### 3.1 remote_env.py：SSH 连接层

**实现方式：** 使用 `fabric.Connection`（基于 paramiko），维持单个持久连接。

```python
# remote_env.py
class RemoteEnv:
    def __init__(self, config):
        self.conn = fabric.Connection(
            host=config.host,
            user=config.user,
            connect_kwargs={"key_filename": config.key_file}
        )
        # fabric 自动处理连接断开后的重连

    def run(self, cmd: str, timeout: int = 60) -> Result:
        """执行单次命令，返回 stdout/stderr/returncode"""

    def run_background(self, cmd: str, log_file: str):
        """后台非阻塞执行，输出重定向到 log_file"""
        self.conn.run(f"nohup {cmd} > {log_file} 2>&1 &")

    def read_log_tail(self, log_file: str, lines: int = 50) -> str:
        return self.run(f"tail -{lines} {log_file}").stdout
```

**并发说明：** `runner.py` 在单次评测内的并发通过 HTTP 请求（`ThreadPoolExecutor`）实现，不需要多个 SSH 通道并发执行命令。npu-smi 监控使用同一个 SSH 连接顺序执行。

---

### 3.2 Phase 0：Ship-It 自动部署 Agent

**职责：** 从裸机到可调用推理服务的全自动部署，可独立运行（`python main.py deploy`）。

#### check_remote_env()

```
- npu-smi info -t usages -i {npu_id} -c {chip_id}  → HBM 总量（单次快照，禁用 watch）
- npu-smi info -t health -i {npu_id}                → Health 状态
- which vllm; python -c "import vllm"               → 框架安装状态
- df -h {working_dir}                   → 磁盘空间检查
- lsof -i:{port}                        → 端口占用检查（获取 PID + 进程名）
- ls {model_dir}                        → 模型权重是否存在
```

#### pull_model(hf_model_url)

```
- local_path 推导规则：
    hf_url 最后一段路径即为模型目录名
    "https://huggingface.co/Qwen/Qwen2-7B" → {working_dir}/models/Qwen2-7B
    "https://huggingface.co/Qwen/Qwen2-7B-Instruct" → {working_dir}/models/Qwen2-7B-Instruct

- HF_TOKEN 优先从远程环境变量读取（$HF_TOKEN），
  若未设置则使用 tuner_config.yaml 中的 hf_token 字段（tuner_config.yaml 不入 git）

- 命令：
  huggingface-cli download {repo_id} \
    --local-dir {local_path} \
    --token {token}
```

#### auto_fix_env()

```
- 框架未安装      → pip install vllm-ascend / lmdeploy
- 端口被占用      →
    1. lsof -i:{port} 获取 PID 和进程名
    2. 验证进程名包含 "vllm" 或 "lmdeploy"（禁止盲杀 sshd 等系统进程）
    3. kill -15 {pid}（SIGTERM 优雅关闭）
    4. 等待 3 秒，若仍存在则 kill -9 {pid}
    5. 若 lsof 不可用则用：ss -tlnp | grep :{port} 解析 PID
- CANN 未激活     → source /usr/local/Ascend/ascend-toolkit/set_env.sh
- Health=Warning  → 记录告警，将 gpu_memory_utilization 上限降低 0.05
```

#### deploy_framework(model_path, config)

```
- 生成启动命令（调用 framework.build_start_cmd）
- 写入 {working_dir}/launch.sh
- remote_env.run_background("bash launch.sh", "vllm.log")
- 轮询 GET /health，间隔 1 秒，最长等待 120 秒
- 超时则读取 vllm.log 尾部触发 self_heal()
```

#### verify_deployment()

```
- GET /health → 200 OK
- POST /v1/chat/completions，发送 "Hello" → 验证响应非空
```

#### self_heal(error, attempt_history)

```
- 读取 vllm.log 尾部 50 行
- 将 (error, logs, attempt_history) 一起发给 Claude 诊断
  （attempt_history 包含前几次失败的修复指令及结果，避免重复相同修复）
- 执行 Claude 给出的修复指令列表
- 重新 deploy_framework()
- 最多重试 3 次，每次失败追加到 attempt_history
- 3 次均失败则抛出异常，打印完整日志路径
```

---

### 3.3 Phase 1：Benchmark Sweep 性能摸底

**职责：** 在单次部署的服务上（不重启），系统性探测性能边界，为 Phase 2 提供有效搜索空间。

**重要约束：** Sweep 全程使用默认启动参数，仅改变请求的并发数和 prompt 长度，不重启服务。

**扫描矩阵：**

```
CONCURRENCY  = [1, 2, 4, 8, 16, 32]
INPUT_LENGTHS = [128, 256, 512, 1024, 2048]

每格测量：requests_per_cell=10 个请求
记录：P99 延迟(ms)、吞吐(tok/s)、状态

OOM 检测方法：
  1. 请求返回 HTTP 5xx → 记录为 ERROR
  2. 请求超时（>30s）→ 检查 GET /health
     - /health 返回 200 → 记录为 TIMEOUT
     - /health 失败 → 服务崩溃，记录 OOM，停止当前行/列扫描
  3. 服务崩溃后重启默认配置继续剩余扫描
```

**Sweep 结果输出（sweep_result.json）：**

```json
{
  "matrix": {
    "1": {"128": {"p99_ms": 45, "tps": 820, "status": "OK"}, ...},
    "8": {"1024": {"status": "OOM"}, ...}
  },
  "oom_boundary": {"concurrency": 8, "input_len": 1024},
  "best_throughput": {"concurrency": 4, "input_len": 512, "tps": 820},
  "recommended_search_space": {
    "max_num_seqs": [4, 8, 16],
    "block_size": [8, 16, 32]
  }
}
```

---

### 3.4 Phase 2a / 2b：优化循环

**两阶段串联逻辑：**

```
Phase 2a（速度/吞吐优化）：
  调优参数：block_size, gpu_memory_utilization, max_num_seqs
  固定参数：temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0（完全默认）
  优化目标：throughput ↑, latency ↓（双目标，KEEP 条件：至少一项改善，另一项不退步）
  每轮需重启 vLLM 服务（启动参数变更）
  停止条件：patience=3 或 max_rounds

Phase 2b（准确率优化）：
  固定参数：best_infra_config（来自 Phase 2a）
  Phase 2b 基线（Round 1）：temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0
  调优参数：temperature, top_p, top_k, repetition_penalty
  优化目标：task_metric（用户指定方向）
  每轮无需重启（生成参数通过 API 请求体传入）
  停止条件：patience=3 或 max_rounds
```

**config_hash 定义：** 对调优参数字典按 key 排序后 JSON 序列化，取 MD5 前 6 位。

```python
import hashlib, json
def compute_config_hash(config: dict) -> str:
    s = json.dumps(config, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:6]
```

用于 results.tsv 去重（跳过已测试的相同参数组合）。

**轮次状态：** KEEP / DISCARD / CRASH

**results.tsv 格式（不入 git）：**

```
round  phase  config_hash  throughput  latency_p99  task_metric  memory_pct  status   reasoning
1      2a     a1b2c3       820.1       180           -            0.72        KEEP     基线
2      2a     d4e5f6       1050.2      145           -            0.83        KEEP     block_size=8 减少 KV 碎片
3      2a     g7h8i9       -           -             -            0.91        CRASH    OOM，自愈降低 gpu_mem_util
4      2b     j1k2l3       1050.2      145           0.612        0.83        KEEP     Phase 2b 基线（temp=1.0）
5      2b     k2l3m4       1050.2      145           0.654        0.83        KEEP     temp=0.7 提升准确率
```

---

### 3.5 brain.py：Claude 因果推理决策

**与传统 HPO（Optuna/Bayesian）的区别：**

传统工具是统计驱动——知道哪个参数好，不知道为什么。
brain.py 是因果推理——读取 npu-smi 信号和历史轮次，理解**为什么**慢，再决定**改什么**。

**输入上下文：**

```
- sweep_result.json（初始上下文）
- results.tsv 全部历史轮次
- 当前轮次 npu-smi 硬件快照
- 当前优化阶段（2a 速度 / 2b 准确率）
- 用户约束（tuner_config.yaml 中的参数范围）
```

**输出（结构化 JSON）：**

```json
{
  "optimization_focus": "speed",
  "next_config": {"block_size": 8, "gpu_memory_utilization": 0.88},
  "reasoning": "HBM 使用率 94%，AICore 利用率 23%，KV Cache 碎片严重，减小 block_size 预期降低延迟 20%",
  "confidence": 0.82,
  "skip_reason": null
}
```

**硬件信号到推理的映射：**

```
HBM 使用率 > 90% + AICore < 50%  → block_size 过大，KV Cache 碎片
AICore < 30% + 吞吐低             → max_num_seqs 不足，并发上限低
Temp > 85°C                       → skip_reason="等待散热"，本轮跳过
Health = Warning                  → confidence 降低，选择保守参数
CRASH（OOM）                      → gpu_memory_utilization 降低 0.05，回滚
```

**self_heal 去重逻辑：** 双重保障——
1. `attempt_history` 传入 Claude，要求其输出的 `fix_commands` 不与历史重复
2. 代码层结构检查：执行前将 `fix_commands` 与 `attempt_history` 中所有历史命令取差集，跳过已执行过的命令，不依赖 LLM 遵守指令

---

### 3.6 EvalSkill 插件系统

**抽象接口：**

```python
class EvalSkill(ABC):
    name: str
    def measure(self, server_url: str, config: dict) -> float:
        """返回用于比较的标量值（归一化或原始，取决于 metric_direction）"""
```

**runner.py 并发范围：** 并发仅发生在单次评测的 HTTP 请求层（throughput skill 的 ThreadPoolExecutor）。不同 EvalSkill 顺序执行（throughput → latency → task_metric → memory）。不跨配置并行（单卡无法同时运行两个服务实例）。

**内置技能：**

| Skill | 实现方式 | 输出 |
|-------|----------|------|
| `throughput` | ThreadPoolExecutor 并发 HTTP 请求 | tokens/s |
| `latency` | 顺序请求计时，统计 P50/P99 | ms |
| `task_metric` | SSH 执行 eval_script，解析 stdout | 用户指定指标值 |
| `memory` | npu-smi info -t usages（单次快照） | HBM 使用率 % |

---

### 3.7 task_metric.py：eval_script 接口契约

**eval_script 调用约定（用户必须遵守）：**

```bash
# vLLM-Tuner 调用方式：
python {eval_script} \
  --server-url http://{host}:{port} \
  --data-dir {eval_data} \
  --sample-size {sample_size}

# eval_script 必须将结果输出到 stdout，格式为单行 JSON：
{"metric": 0.678}

# 其中 "metric" 为固定 key，值为浮点数
# stderr 可输出任意日志，不影响解析
```

**超时配置：** `evaluation.timeout_seconds`，默认 600 秒（10 分钟）。超时则标记本轮为 CRASH。

**参数命名映射：** YAML 字段 `sample_size`（下划线）映射到 CLI 参数 `--sample-size`（连字符），遵循 CLI 惯例。实现时统一用 `str(config["sample_size"])` 填充 `--sample-size` 参数。

**task_metric 支持的指标方向：**

```yaml
metric: accuracy        # metric_direction: maximize
metric: edit_distance   # metric_direction: minimize
metric: f1              # metric_direction: maximize
metric: bleu            # metric_direction: maximize
```

---

### 3.8 hardware/ascend.py：npu-smi 实际命令

**所有命令均为单次快照，不使用 `watch` 子命令（`watch` 会阻塞 SSH 通道）：**

```bash
# 算力利用率（单次快照，替代 watch）
npu-smi info -t common -i {npu_id} -c {chip_id}
→ 输出包含 AICore(%) 字段

# 显存（HBM）使用
npu-smi info -t usages -i {npu_id} -c {chip_id}
→ HBM-Usage(MB): 3161 / 65536

# 功耗
npu-smi info -t power -i {npu_id} -c {chip_id}
→ Power(W): 93.6

# 温度
npu-smi info -t temp -i {npu_id} -c {chip_id}
→ Temp(C): 40

# 健康状态（fast-fail 检查）
npu-smi info -t health -i {npu_id}
→ OK / Warning / Alarm / Critical
```

---

### 3.9 多推理框架支持

**抽象接口：**

```python
class InferenceFramework(ABC):
    def build_start_cmd(self, model_weights: str, config: dict) -> str: ...
    def get_api_base(self, host: str, port: int) -> str: ...
    def get_health_endpoint(self) -> str: ...
    @property
    def tunable_infra_params(self) -> dict: ...  # Phase 2a 可调参数及范围
```

**支持框架：**

| 框架 | 场景 | 健康检查端点 |
|------|------|-------------|
| vLLM（via vllm-ascend） | 昇腾，主角 | `/health` |
| LMDeploy | 昇腾生态替代 | `/v1/models` |
| SGLang | CUDA 场景 | `/health` |

所有框架均暴露 OpenAI 兼容 API（`/v1/chat/completions`），runner.py 无需感知框架类型。

---

### 3.10 analysis.py：Pareto 分析

**Pareto 分析范围说明：**

- Phase 2a 行的 `task_metric = "-"`（未运行 eval_script），不参与 throughput vs accuracy 的二维 Pareto 计算
- Phase 2a 的最优结果独立选取：best_latency（P99 最低）和 best_throughput（tok/s 最高）
- 二维 Pareto frontier 仅在 **Phase 2b 行**上计算（throughput vs task_metric 均有值）

**Pareto 最优判断（2D，throughput vs task_metric，仅 Phase 2b 行）：**

```python
def is_pareto_dominated(candidate, others, metric_direction: str) -> bool:
    """
    metric_direction: "maximize" 或 "minimize"
    "不劣于"定义：
      - maximize：other.task_metric >= candidate.task_metric
      - minimize：other.task_metric <= candidate.task_metric（越小越好，反转比较方向）

    若 others 中存在某配置，其 throughput 不劣于 candidate.throughput
    且 task_metric 不劣于 candidate.task_metric（按 metric_direction），
    且至少一项严格更优，则 candidate 被支配，返回 True
    """

def get_pareto_frontier(phase2b_results: list, metric_direction: str) -> list:
    """仅接收 Phase 2b 行，metric_direction 从 tuner_config.yaml 传入，返回非支配配置集合"""
```

**归一化方式：** 各指标在 Phase 2b 全部已测配置中 min-max 归一化后计算支配关系，避免量纲影响。

**输出内容：**
1. Phase 2b 各轮次散点图（throughput vs task_metric），KEEP/DISCARD/CRASH 颜色区分
2. Pareto frontier 曲线
3. 推荐配置高亮标注
4. Phase 2a best_latency / best_throughput 单独标注在图例中

---

## 4. 配置文件设计

```yaml
# tuner_config.yaml
# 注意：此文件含 hf_token，已加入 .gitignore，不应提交至版本库
# 推荐：将 hf_token 设置为远程机环境变量 $HF_TOKEN，此处留空

remote:
  host: 192.168.1.100
  port: 22
  user: ubuntu
  key_file: ~/.ssh/id_rsa
  working_dir: /home/ubuntu/workspace
  vllm_port: 8000

hardware:
  type: ascend          # 或 cuda
  device_id: 0
  npu_id: 4             # npu-smi -i 参数
  chip_id: 0            # npu-smi -c 参数

framework: vllm         # 或 lmdeploy / sglang

model:
  hf_url: https://huggingface.co/Qwen/Qwen2-7B
  hf_token: ""          # 优先使用远程机 $HF_TOKEN 环境变量
  # local_path 自动推导：{working_dir}/models/{hf_url 最后路径段}
  # 示例：→ /home/ubuntu/workspace/models/Qwen2-7B

evaluation:
  script: /evals/run_mmlu.py     # 远程机绝对路径
  data: /datasets/mmlu_sample/   # 远程机绝对路径
  metric: accuracy
  metric_direction: maximize      # 或 minimize（如 edit_distance）
  sample_size: 50
  timeout_seconds: 600
  skills:
    - throughput
    - latency
    - task_metric
    - memory

sweep:
  concurrency_levels: [1, 2, 4, 8, 16, 32]
  input_lengths: [128, 256, 512, 1024, 2048]
  requests_per_cell: 10
  request_timeout_seconds: 30

optimization:
  phase_2a:
    max_rounds: 15
    patience: 3
    parameters:
      block_size: [8, 16, 32]
      gpu_memory_utilization: [0.75, 0.80, 0.85, 0.90]
      max_num_seqs: [64, 128, 256]
  phase_2b:
    max_rounds: 15
    patience: 3
    baseline:                     # Phase 2b Round 1 使用的生成参数基线
      temperature: 1.0
      top_p: 1.0
      top_k: -1                   # -1 表示禁用
      repetition_penalty: 1.0
    parameters:
      temperature: [0.6, 0.7, 0.8, 1.0]
      top_p: [0.85, 0.90, 0.95, 1.0]
      top_k: [-1, 20, 50, 100]    # -1=禁用，Qwen 推荐 20
      repetition_penalty: [1.0, 1.05, 1.1]

save_dir: ./results/
```

---

## 5. 结果追踪与报告

### 5.1 results.tsv（不入 git）

```
round  phase  config_hash  throughput  latency_p99  task_metric  memory_pct  status   reasoning
1      2a     a1b2c3       820.1       180           -            0.72        KEEP     基线
2      2a     d4e5f6       1050.2      145           -            0.83        KEEP     block_size=8 减少 KV 碎片
3      2a     g7h8i9       -           -             -            0.91        CRASH    OOM，自愈降低 gpu_mem_util 至 0.85
4      2b     j1k2l3       1050.2      145           0.612        0.83        KEEP     Phase 2b 基线（temp=1.0, top_k=-1）
5      2b     k2l3m4       1050.2      145           0.654        0.83        KEEP     temp=0.7 提升准确率
6      2b     m5n6o7       1050.2      145           0.678        0.83        KEEP     top_k=20 进一步提升
```

### 5.2 最终报告（results/best_config.md）

```markdown
# vLLM-Tuner Optimization Report
## 模型：Qwen2-7B | 硬件：Ascend 910B3 | 用时：约 2.5 小时

### Sweep 性能基线矩阵
（表格见 sweep_result.json）

### ① 最低延迟配置（Phase 2a Round 3）
block_size=8, gpu_mem=0.85, max_num_seqs=128
P99 延迟：115ms（基线 -36%）

### ② 最大吞吐配置（Phase 2a Round 5）
block_size=8, gpu_mem=0.88, max_num_seqs=256
吞吐量：1050 tok/s（基线 +28%）

### ③ 最高准确率配置（Phase 2b Round 6）
block_size=8, gpu_mem=0.88（固定）
temperature=0.65, top_p=0.90, top_k=20, repetition_penalty=1.05
任务准确率：67.8%（基线 +10.8%）

### ⭐ Pareto 综合推荐
同上 ③，吞吐损失 < 2%，准确率提升 +10.8%

### 一键重启最优配置
python main.py deploy --config tuner_config.yaml --use-best-config results/best_config.md
# --use-best-config：从指定报告文件读取最优基础设施参数和生成参数，
#   传入 deploy_framework() 代替 tuner_config.yaml 中的默认参数，直接拉起最优服务
```

---

## 6. 示例用户执行日志

```
$ python main.py run --config tuner_config.yaml

─── Phase 0: Ship-It ──────────────────────────────────────
[✓] SSH 连接 192.168.1.100（fabric.Connection 持久连接）
[✓] 昇腾 910B3，HBM 65536MB，Health=OK
[✓] 磁盘空间充足：450GB 可用
[↓] 拉取 Qwen/Qwen2-7B → /home/ubuntu/workspace/models/Qwen2-7B...
[✓] 模型拉取完成
[!] 端口 8000 被占用（进程：vllm，PID 12345）→ SIGTERM → 已清理
[✓] CANN 环境激活
[✓] vLLM 服务启动（默认参数，等待 52 秒）
[✓] 健康检查通过，测试请求正常

─── Phase 1: Sweep 性能摸底（不重启服务）───────────────────
[Sweep] 6 并发 × 5 输入长度 = 30 组合，每格 10 次请求...
[Sweep] (concurrency=8, input=1024): /health 无响应 → OOM，重启默认配置继续
[Sweep] OOM 边界：并发=8，输入长度=1024
[Sweep] 最佳吞吐点：并发=4，输入长度=512 → 820 tok/s
[Sweep] 推荐搜索空间：max_num_seqs=[64,128], block_size=[8,16]

─── Phase 2a: 速度/吞吐优化（每轮重启服务）────────────────
[Round 1/15] block_size=16, gpu_mem=0.85 → 820tok/s, P99=145ms   KEEP (a1b2c3)
[Round 2/15] block_size=8,  gpu_mem=0.85 → 980tok/s, P99=120ms   KEEP (d4e5f6)
[Round 3/15] block_size=8,  gpu_mem=0.90 → CRASH(OOM)            CRASH→自愈(g7h8i9)
[Round 4/15] block_size=8,  gpu_mem=0.88 → 1050tok/s,P99=115ms   KEEP (h8i9j0)
[收敛] 连续 3 轮无改善，Phase 2a 结束
[✓] best_infra_config：block_size=8, gpu_mem=0.88（config_hash: h8i9j0）

─── Phase 2b: 准确率优化（无需重启）───────────────────────
[Round 1/15] temp=1.0, top_p=1.0, top_k=-1  → accuracy=61.2%    KEEP 基线 (j1k2l3)
[Round 2/15] temp=0.7, top_p=0.95, top_k=-1 → accuracy=65.4%    KEEP (k2l3m4)
[Round 3/15] temp=0.65,top_p=0.90, top_k=20 → accuracy=67.8%    KEEP (m5n6o7)
[Round 4/15] temp=0.6, top_p=0.85, top_k=20 → accuracy=66.9%    DISCARD
[收敛] 连续 3 轮无改善，Phase 2b 结束

─── Phase 3: 最终报告 ──────────────────────────────────────
📄 results/best_config.md 已生成
📊 results/pareto_chart.png 已生成
① 最低延迟：P99=115ms（-36%） → results/config_best_latency.yaml
② 最大吞吐：1050 tok/s（+28%） → results/config_best_throughput.yaml
③ 最高准确率：67.8%（+10.8%） → results/config_best_accuracy.yaml
⭐ 综合推荐：③
[✓] 服务保持运行（最优配置），可直接访问 http://192.168.1.100:8000
```

---

## 7. 设计原则

1. **快速失败（Fast-Fail）**：每轮启动后发 3 个测试请求验证，异常立即标记 CRASH，不浪费评测预算
2. **单次快照，不阻塞 SSH**：所有 npu-smi 命令使用 `-t common/usages/power/temp/health`，禁用 `watch` 子命令
3. **端口清理安全第一**：杀进程前验证进程名，禁止盲杀系统进程
4. **因果推理优于统计搜索**：brain.py 读取硬件信号，理解为什么慢，不做随机搜索
5. **两阶段串联**：先锁定最优基础设施配置（需重启），再调生成参数（无需重启）
6. **Sweep 驱动搜索空间**：不盲搜，先摸底，在有效区域内搜索
7. **结果可审计**：每轮保存 reasoning 字段，config_hash 用于去重和追溯
8. **服务结束后保持运行**：优化完成后服务以最优配置运行，供直接使用

---

## 8. 依赖清单

```
# 控制机（运行 vllm-tuner）
fabric>=3.0          # SSH 连接（基于 paramiko，支持自动重连）
anthropic            # Claude API
requests             # HTTP 健康检查 + API 调用
pyyaml               # 配置文件解析
pandas               # results.tsv 读写分析
matplotlib           # Pareto 图生成

# 远程机（昇腾服务器）
vllm-ascend          # vLLM 昇腾插件
huggingface-cli      # 模型下载（随 huggingface_hub 安装）
# npu-smi            # 随 CANN 工具包预装，无需额外安装
```

## 9. .gitignore 条目

```
results.tsv
results/
tuner_config.yaml    # 含 hf_token，严禁提交
*.log
__pycache__/
```
