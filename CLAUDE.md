# vLLM-Tuner

Auto-deploy LLMs on remote Ascend NPU and tune inference + generation parameters using Claude as a causal reasoning engine.

## What this does

Given an HF model URL, an eval script, and SSH access to a remote machine, vLLM-Tuner:
1. **Deploys** vLLM/LMDeploy/SGLang on the remote machine (Phase 0: Ship-It)
2. **Sweeps** concurrency × input-length matrix to find performance boundaries (Phase 1)
3. **Tunes infra params** (block_size, gpu_memory_utilization, max_num_seqs) for throughput/latency — each round restarts the service (Phase 2a)
4. **Tunes generation params** (temperature, top_p, top_k, repetition_penalty) for task accuracy — no restart needed (Phase 2b)
5. **Reports** three best configs: lowest latency, highest throughput, best accuracy + Pareto recommendation (Phase 3)

## Key design decisions

- **Brain** (`brain.py`): Claude API makes all parameter decisions via causal reasoning (hardware signals → why slow → what to change), not statistical search
- **Two-phase**: infra params first (need restart) → generation params second (no restart)
- **Ascend-first**: npu-smi single-snapshot commands only — no `watch` (blocks SSH)
- **Safe port cleanup**: verify process name before kill; refuse to kill sshd/system processes
- **config_hash**: MD5(sorted JSON)[:6] — deduplicates already-tested configs across rounds
- **results.tsv**: tracks every round with KEEP/DISCARD/CRASH status + reasoning + config_json (enables `--use-best-config` round-trip)

## File map

```
vllm_tuner/
├── main.py           # CLI: run / deploy
├── config.py         # YAML → dataclasses
├── remote_env.py     # fabric SSH wrapper
├── shipit.py         # Phase 0 deploy agent
├── actor.py          # Service lifecycle (start/stop/fast-fail)
├── sweep.py          # Phase 1 matrix sweep
├── orchestrator.py   # Phase 2a + 2b loop
├── brain.py          # Claude decision engine
├── runner.py         # Sequential EvalSkill runner
├── reporter.py       # results.tsv + best_config.md
├── analysis.py       # Pareto frontier
├── frameworks/       # vLLM / LMDeploy / SGLang ABCs
├── hardware/         # Ascend npu-smi / CUDA nvidia-smi
└── skills/           # throughput / latency / task_metric / memory
```

## Eval script contract

User-provided eval scripts must accept:
```
python eval_script.py --server-url URL --data-dir DIR --sample-size N
```
And print to stdout (single line):
```json
{"metric": 0.678}
```

## Commands

```bash
python -m vllm_tuner.main run --config tuner_config.yaml
python -m vllm_tuner.main deploy --config tuner_config.yaml [--use-best-config results/best_config.md]
```

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Tests are TDD stubs — implement the module, then fill in the test body. All test functions exist; none have logic yet.

## Sensitive files (not committed)

- `tuner_config.yaml` — contains hf_token and SSH key path
- `results/`, `results.tsv`, `*.log`
