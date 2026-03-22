import argparse
import os
import sys

from vllm_tuner.config import load_config
from vllm_tuner.remote_env import RemoteEnv
from vllm_tuner.frameworks import get_framework
from vllm_tuner.hardware import get_observer
from vllm_tuner.actor import Actor
from vllm_tuner.shipit import ShipIt
from vllm_tuner.sweep import BenchmarkSweep
from vllm_tuner.brain import Brain
from vllm_tuner.runner import Runner
from vllm_tuner.reporter import Reporter
from vllm_tuner.orchestrator import Orchestrator
from vllm_tuner.skills import ThroughputSkill, LatencySkill, TaskMetricSkill, MemorySkill


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with run and deploy subcommands."""
    raise NotImplementedError


def _build_skills(cfg, remote, observer) -> list:
    """Instantiate EvalSkill list from config skill names."""
    raise NotImplementedError


def _load_best_infra_from_report(report_path: str, save_dir: str) -> dict | None:
    """Parse best_config.md + results.tsv config_json to reconstruct best infra params."""
    raise NotImplementedError


def cmd_run(args):
    """Execute full 5-phase pipeline: deploy → sweep → 2a → 2b → report."""
    raise NotImplementedError


def cmd_deploy(args):
    """Deploy only (Phase 0 Ship-It), optionally loading best config from report."""
    raise NotImplementedError


def main():
    """CLI entry point."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
