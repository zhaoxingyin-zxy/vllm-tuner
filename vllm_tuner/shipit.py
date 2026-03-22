# vllm_tuner/shipit.py
import re
import time
from pathlib import Path


class ShipIt:
    """Phase 0: Automated deployment agent. Can be used standalone."""

    def __init__(self, remote, actor, framework, config, docker_manager=None):
        self.remote = remote
        self.actor = actor
        self.framework = framework
        self.cfg = config
        self.docker_manager = docker_manager

    def run(self, default_infra_config: dict) -> bool:
        """Full deployment pipeline. Returns True on success."""
        print("[Ship-It] Checking remote environment...")
        env = self.check_remote_env()
        print(f"[OK] HBM: {env['hbm_total_mb']}MB, Health: {env['health']}")

        self.auto_fix_env(env)

        if self.docker_manager:
            if not self.docker_manager.is_container_running():
                print("[Docker] Starting container...")
                model_dir = str(Path(self.cfg.model.local_path).parent)
                self.docker_manager.run_container(model_dir, self.cfg.remote.vllm_port)

        if not env["model_exists"]:
            print(f"[Down] Pulling model {self.cfg.model.hf_url}...")
            self.pull_model()
            print("[OK] Model pull complete")

        print("[...] Starting inference service...")
        self.actor.start_service(self.cfg.model.local_path, default_infra_config)

        if not self.actor.fast_fail_check():
            logs = self.remote.read_log_tail(
                f"{self.cfg.remote.working_dir}/vllm.log"
            )
            if hasattr(self, "brain") and self.brain is not None:
                self.self_heal(
                    error="fast-fail check failed",
                    attempt_history=[],
                    brain=self.brain,
                    default_infra_config=default_infra_config,
                )
            else:
                raise RuntimeError(f"Service fast-fail check failed. Logs:\n{logs}")

        print("[OK] Service healthy and responding")
        return True

    def self_heal(
        self,
        error: str,
        attempt_history: list,
        brain,
        default_infra_config: dict,
    ):
        """
        Retry deployment up to 3 times, asking Claude to diagnose and fix each time.
        Structurally deduplicates fix_commands vs attempt_history to avoid repeating
        the same commands (does not rely on LLM compliance alone).
        Raises RuntimeError after 3 failures.
        """
        for attempt in range(3):
            logs = self.remote.read_log_tail(
                f"{self.cfg.remote.working_dir}/vllm.log"
            )
            diagnosis = brain.diagnose(
                logs=logs, error=error, attempt_history=attempt_history
            )
            # Structural dedup: only run commands not already tried
            already_tried = {cmd for h in attempt_history for cmd in h.get("cmd", [])}
            new_cmds = [
                cmd for cmd in diagnosis.get("fix_commands", [])
                if cmd not in already_tried
            ]
            for cmd in new_cmds:
                self.remote.run(cmd, timeout=120)
            attempt_history.append({"attempt": attempt, "error": error, "cmd": new_cmds})

            try:
                self.actor.start_service(
                    self.cfg.model.local_path, default_infra_config
                )
                if self.actor.fast_fail_check():
                    return  # Success
                error = "fast-fail check failed after fix attempt"
            except Exception as e:
                error = str(e)

        raise RuntimeError(
            f"self-heal exhausted 3 attempts. "
            f"Check {self.cfg.remote.working_dir}/vllm.log on the remote machine."
        )

    def check_remote_env(self) -> dict:
        npu_id, chip_id = self.cfg.hardware.npu_id, self.cfg.hardware.chip_id
        port = self.cfg.remote.vllm_port
        work_dir = self.cfg.remote.working_dir

        # HBM
        usages_out = self.remote.run(
            f"npu-smi info -t usages -i {npu_id} -c {chip_id}"
        ).stdout
        m = re.search(r"HBM-Usage\(MB\):\s*(\d+)\s*/\s*(\d+)", usages_out)
        hbm_total = float(m.group(2)) if m else 0.0

        # Health
        health_out = self.remote.run(
            f"npu-smi info -t health -i {npu_id}"
        ).stdout.strip()
        health = health_out.split()[0] if health_out else "UNKNOWN"

        # vLLM installed — Docker: check image; bare: import check
        if self.docker_manager:
            vllm_installed = self.docker_manager.is_image_present()
        else:
            vllm_check = self.remote.run("python -c 'import vllm'")
            vllm_installed = vllm_check.returncode == 0

        # Disk
        df_out = self.remote.run(f"df -h {work_dir}").stdout
        disk_ok = bool(df_out.strip())

        # Port occupant
        lsof_out = self.remote.run(f"lsof -ti:{port} -sTCP:LISTEN").stdout.strip()
        port_occupied = bool(lsof_out)

        # Model exists
        model_check = self.remote.run(f"ls {self.cfg.model.local_path}")
        model_exists = bool(model_check.stdout.strip())

        return {
            "hbm_total_mb": hbm_total,
            "health": health,
            "vllm_installed": vllm_installed,
            "disk_ok": disk_ok,
            "port_occupied": port_occupied,
            "port_pid": lsof_out,
            "model_exists": model_exists,
        }

    def auto_fix_env(self, env: dict):
        if self.docker_manager:
            if not env["vllm_installed"]:
                print("[Fix] Pulling Docker image...")
                self.docker_manager.pull()
            if env["port_occupied"]:
                print(f"[!] Port {self.cfg.remote.vllm_port} occupied, clearing inside container...")
                self.docker_manager.exec_run(
                    f"fuser -k {self.cfg.remote.vllm_port}/tcp || true", timeout=10
                )
            return

        # Bare-process mode (unchanged)
        if not env["vllm_installed"]:
            print("[Fix] Installing vllm-ascend...")
            self.remote.run("pip install vllm-ascend", timeout=300)
        if env["port_occupied"]:
            print(f"[!] Port {self.cfg.remote.vllm_port} occupied, clearing...")
            self._clear_port(self.cfg.remote.vllm_port)
        if self.cfg.hardware.type == "ascend":
            self.remote.run(
                "source /usr/local/Ascend/ascend-toolkit/set_env.sh || true"
            )

    def pull_model(self):
        url = self.cfg.model.hf_url
        repo_id = "/".join(url.rstrip("/").split("/")[-2:])
        local_path = self.cfg.model.local_path

        token_part = ""
        token_check = self.remote.run("echo $HF_TOKEN").stdout.strip()
        if not token_check and self.cfg.model.hf_token:
            token_part = f" --token {self.cfg.model.hf_token}"

        cmd = (
            f"huggingface-cli download {repo_id}"
            f" --local-dir {local_path}{token_part}"
        )
        self.remote.run(cmd, timeout=3600)

    def _clear_port(self, port: int):
        pid_result = self.remote.run(f"lsof -ti:{port} -sTCP:LISTEN")
        pid = pid_result.stdout.strip()
        if not pid:
            return
        name_result = self.remote.run(f"ps -p {pid} -o comm=")
        proc_name = name_result.stdout.strip().lower()
        safe = ("python", "vllm", "lmdeploy", "sglang")
        if not any(s in proc_name for s in safe):
            raise RuntimeError(
                f"Refusing to kill PID {pid} (process: {proc_name!r}) on port {port}."
            )
        self.remote.run(f"kill -15 {pid}")
        time.sleep(3)
        check = self.remote.run(f"ps -p {pid} -o pid=")
        if check.stdout.strip():
            self.remote.run(f"kill -9 {pid}")
