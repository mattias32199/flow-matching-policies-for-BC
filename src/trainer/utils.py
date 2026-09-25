import json
from pathlib import Path


class JsonlLogger:
    """One JSON object per line. Load with pandas.read_json(path, lines=True)."""

    def __init__(self, path: str | Path, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def log(self, step: int, metrics: dict):
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self.echo:
            print(" | ".join([f"step {step:>7d}"] + [f"{k} {v:.4g}" for k, v in metrics.items()]))
