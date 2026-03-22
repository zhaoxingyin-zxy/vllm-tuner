# vllm_tuner/brain.py
import json
import anthropic


DECIDE_SYSTEM = """You are a performance engineer specializing in LLM inference on Ascend NPU.
Given hardware metrics and optimization history, decide the next parameter configuration to test.

Hardware signal interpretation:
- HBM usage > 90% + AICore < 50%: block_size too large, KV cache fragmentation
- AICore < 30% + low throughput: max_num_seqs too low, insufficient concurrency
- Temp > 85C: skip this round, wait for cooling
- Health = Warning: reduce confidence, pick conservative parameters

Always output valid JSON matching this schema:
{
  "optimization_focus": "speed" | "accuracy",
  "next_config": {<param>: <value>, ...},
  "reasoning": "<causal explanation>",
  "confidence": 0.0-1.0,
  "skip_reason": null | "<reason to skip this round>"
}

Do NOT suggest configs whose hash already appears in seen_hashes."""

DIAGNOSE_SYSTEM = """You are a systems debugger for vLLM on Ascend NPU.
Given error logs and previous failed fix attempts, diagnose the root cause and suggest new fix commands.
Do NOT repeat commands from attempt_history.

Output valid JSON:
{
  "diagnosis": "<root cause>",
  "fix_commands": ["<shell command>", ...],
  "adjusted_param": {<param>: <value>}
}"""


class Brain:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def decide_next_config(
        self,
        sweep_result: dict,
        history: list,
        hw_stats,
        phase: str,
        param_space: dict,
        seen_hashes: set,
    ) -> dict:
        hw_str = str(hw_stats) if hw_stats else "unavailable"
        history_str = json.dumps(history[-10:], ensure_ascii=False)  # last 10 rows

        user_msg = (
            f"Phase: {phase}\n"
            f"Parameter space: {json.dumps(param_space)}\n"
            f"Seen config hashes (skip these): {list(seen_hashes)}\n"
            f"Sweep baseline: {json.dumps(sweep_result)}\n"
            f"Recent history (last 10 rounds): {history_str}\n"
            f"Current hardware snapshot: {hw_str}\n\n"
            "Decide the next config to test."
        )

        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=DECIDE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return json.loads(msg.content[0].text)

    def diagnose(self, logs: str, error: str, attempt_history: list) -> dict:
        history_str = json.dumps(attempt_history, ensure_ascii=False)
        user_msg = (
            f"Error: {error}\n"
            f"Log tail:\n{logs}\n"
            f"Previous failed attempts (do NOT repeat): {history_str}\n\n"
            "Diagnose and suggest new fix commands."
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=DIAGNOSE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return json.loads(msg.content[0].text)
