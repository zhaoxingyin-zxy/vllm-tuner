import sys
from unittest.mock import patch, MagicMock
import pytest
from vllm_tuner.main import build_parser


def test_parser_run_requires_config():
    parser = build_parser()
    args = parser.parse_args(["run", "--config", "tuner_config.yaml"])
    assert args.config == "tuner_config.yaml"
    assert args.command == "run"


def test_parser_deploy_subcommand():
    parser = build_parser()
    args = parser.parse_args(["deploy", "--config", "tuner_config.yaml"])
    assert args.command == "deploy"


def test_parser_deploy_with_best_config():
    parser = build_parser()
    args = parser.parse_args([
        "deploy", "--config", "cfg.yaml",
        "--use-best-config", "results/best_config.md"
    ])
    assert args.use_best_config == "results/best_config.md"
