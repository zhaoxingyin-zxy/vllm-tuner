"""
Example eval script satisfying vLLM-Tuner's eval_script contract.
Usage: python eval_stub.py --server-url URL --data-dir DIR --sample-size N
Stdout: single-line JSON {"metric": <float>}
"""
import argparse
import json
import requests


def main():
    """Run stub evaluation and print metric JSON to stdout."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
