class Runner:
    """Runs EvalSkills sequentially for a single optimization round."""

    def __init__(self, skills: list, server_url: str):
        pass

    def run_all(self, gen_params: dict) -> dict:
        """Execute all skills in order, return unified metrics dict."""
        raise NotImplementedError
