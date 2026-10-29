"""Package/verify/restore completed aggregate metrics, never source data or weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_metric_bundle import pack, verify


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment'); p.add_argument('--verify', action='store_true')
    p.add_argument('--restore', action='store_true'); a = p.parse_args()
    if not a.experiment or Path(a.experiment).name != a.experiment:
        raise ValueError('One owned experiment directory name required')
    public = ROOT/'outputs/publication_readiness_2026_09'/a.experiment
    print(json.dumps(verify(ROOT, public, restore=a.restore) if a.verify or a.restore else pack(ROOT, public)))
