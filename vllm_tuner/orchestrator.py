# vllm_tuner/orchestrator.py
from vllm_tuner.reporter import compute_config_hash


class Orchestrator:
    def __init__(self, config, actor, brain, runner, reporter, sweep_result, observer):
        self.cfg = config
        self.actor = actor
        self.brain = brain
        self.runner = runner
        self.reporter = reporter
        self.sweep_result = sweep_result
        self.observer = observer
        self._round = 0

    def run_phase_2a(self) -> dict:
        """Infra tuning: restarts service each round. Returns best_infra_config."""
        cfg2a = self.cfg.optimization.phase_2a
        best_config = {}
        best_throughput = 0.0
        best_lat_ms = None
        no_improve = 0

        for _ in range(cfg2a.max_rounds):
            self._round += 1
            history = self.reporter.load_all()
            seen = {r["config_hash"] for r in history}
            hw_stats = self.observer.get_stats() if self.observer else None

            decision = self.brain.decide_next_config(
                sweep_result=self.sweep_result,
                history=history,
                hw_stats=hw_stats,
                phase="2a",
                param_space=cfg2a.parameters,
                seen_hashes=seen,
            )

            if decision.get("skip_reason"):
                print(f"[Round {self._round}] Skipped: {decision['skip_reason']}")
                continue

            next_cfg = decision["next_config"]
            cfg_hash = compute_config_hash(next_cfg)
            if cfg_hash in seen:
                no_improve += 1
                if no_improve >= cfg2a.patience:
                    break
                continue

            # Restart with new infra config; self-heal on failure (max 3 retries)
            attempt_history = []
            deployed = False
            for attempt in range(3):
                try:
                    self.actor.restart_service(self.cfg.model.local_path, next_cfg)
                    if not self.actor.fast_fail_check():
                        raise RuntimeError("fast-fail check failed after restart")
                    deployed = True
                    break
                except Exception as e:
                    logs = self.actor.remote.read_log_tail(
                        f"{self.cfg.remote.working_dir}/vllm.log"
                    )
                    diagnosis = self.brain.diagnose(
                        logs=logs, error=str(e), attempt_history=attempt_history
                    )
                    # Structural dedup: flatten all previously tried commands into a set
                    already_tried = {cmd for h in attempt_history for cmd in h.get("cmd", [])}
                    new_cmds = [
                        cmd for cmd in diagnosis.get("fix_commands", [])
                        if cmd not in already_tried
                    ]
                    for cmd in new_cmds:
                        self.actor.remote.run(cmd, timeout=120)
                    attempt_history.append({"attempt": attempt, "error": str(e),
                                            "cmd": new_cmds})

            if not deployed:
                self.reporter.append_row(
                    self._round, "2a", next_cfg, None, None, None, None,
                    "CRASH", "Failed after 3 self-heal attempts"
                )
                print(f"[Round {self._round}] CRASH: exhausted self-heal retries")
                continue

            metrics = self.runner.run_all(gen_params={})
            tps = metrics.get("throughput") or 0.0
            lat = metrics.get("latency_p99") or 0.0

            # KEEP if: at least one of (throughput, latency) improves,
            # AND the other does not regress. Dual-objective KEEP condition.
            prev_lat = best_lat_ms if best_lat_ms is not None else float("inf")
            tps_improved = tps > best_throughput
            lat_improved = lat < prev_lat
            tps_ok = tps >= best_throughput * 0.99   # allow 1% tolerance
            lat_ok = lat <= prev_lat * 1.01
            improved = (tps_improved and lat_ok) or (lat_improved and tps_ok)
            status = "KEEP" if improved else "DISCARD"
            self.reporter.append_row(
                self._round, "2a", next_cfg,
                tps, lat, None, metrics.get("memory_pct"),
                status, decision.get("reasoning", "")
            )
            print(f"[Round {self._round}/2a] {next_cfg} → {tps:.0f}tok/s, P99={lat:.0f}ms  {status}")

            if improved:
                best_throughput = tps
                best_lat_ms = lat
                best_config = next_cfg
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= cfg2a.patience:
                print(f"[Converged] {no_improve} rounds without improvement")
                break

        return best_config

    def run_phase_2b(self, infra_config: dict) -> dict:
        """Generation param tuning: no restart. Returns best_gen_config."""
        cfg2b = self.cfg.optimization.phase_2b
        direction = self.cfg.evaluation.metric_direction  # maximize or minimize
        best_metric = float("-inf") if direction == "maximize" else float("inf")
        best_config = dict(cfg2b.baseline)
        no_improve = 0

        for _ in range(cfg2b.max_rounds):
            self._round += 1
            history = self.reporter.load_all()
            seen = {r["config_hash"] for r in history}

            decision = self.brain.decide_next_config(
                sweep_result=self.sweep_result,
                history=history,
                hw_stats=None,
                phase="2b",
                param_space=cfg2b.parameters,
                seen_hashes=seen,
            )

            if decision.get("skip_reason"):
                continue

            next_gen = decision["next_config"]
            cfg_hash = compute_config_hash(next_gen)
            if cfg_hash in seen:
                no_improve += 1
                if no_improve >= cfg2b.patience:
                    break
                continue

            # No restart — pass gen params via API
            metrics = self.runner.run_all(gen_params=next_gen)
            task_val = metrics.get("task_metric")

            if task_val is None:
                status = "CRASH"
            else:
                if direction == "maximize":
                    improved = task_val > best_metric
                else:
                    improved = task_val < best_metric
                status = "KEEP" if improved else "DISCARD"

            self.reporter.append_row(
                self._round, "2b", next_gen,
                metrics.get("throughput"), metrics.get("latency_p99"),
                task_val, metrics.get("memory_pct"),
                status, decision.get("reasoning", "")
            )
            print(f"[Round {self._round}/2b] {next_gen} → metric={task_val}  {status}")

            if status == "KEEP":
                best_metric = task_val
                best_config = next_gen
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= cfg2b.patience:
                print(f"[Converged] {no_improve} rounds without improvement")
                break

        return best_config
