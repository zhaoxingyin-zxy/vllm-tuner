import os
import csv
import pytest
from vllm_tuner.reporter import Reporter, compute_config_hash


def test_config_hash_is_deterministic():
    pass


def test_config_hash_differs_for_different_configs():
    pass


def test_reporter_writes_tsv_row(tmp_path):
    pass


def test_reporter_skips_duplicate_hash(tmp_path):
    pass


def test_reporter_load_all_rounds(tmp_path):
    pass


def test_reporter_stores_config_json(tmp_path):
    pass
