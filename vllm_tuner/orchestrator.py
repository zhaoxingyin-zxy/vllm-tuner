from vllm_tuner.reporter import compute_config_hash


class Orchestrator:
    """Phase 2a + 2b optimization loop controller."""

    def __init__(self, config, actor, brain, runner, reporter, sweep_result, observer):
        pass

    def run_phase_2a(self) -> dict:
        """Infra tuning: restart service each round, return best_infra_config."""
        raise NotImplementedError

    def run_phase_2b(self, infra_config: dict) -> dict:
        """Generation param tuning: no restart, return best_gen_config."""
        raise NotImplementedError
