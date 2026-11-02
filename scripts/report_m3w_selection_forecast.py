"""Public-scalar-only report rebuild; not a new training or inference run."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_selection_forecast_report import render

if __name__ == "__main__":
    print(json.dumps(render(ROOT), indent=2, sort_keys=True))
