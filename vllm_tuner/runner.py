# vllm_tuner/runner.py
class Runner:
    """Runs EvalSkills sequentially for a single round."""

    def __init__(self, skills: list, server_url: str):
        self.skills = skills
        self.server_url = server_url

    def run_all(self, gen_params: dict) -> dict:
        results = {}
        for skill in self.skills:
            try:
                val = skill.measure(self.server_url, gen_params)
                if isinstance(val, dict):
                    results.update(val)
                else:
                    results[skill.name] = val
            except Exception as e:
                results[skill.name] = None
                print(f"[!] Skill {skill.name} failed: {e}")

        return {
            "throughput": results.get("throughput"),
            "latency_p99": results.get("p99_ms") or results.get("latency"),
            "task_metric": results.get("task_metric"),
            "memory_pct": results.get("memory"),
        }
