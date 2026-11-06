"""Prepare readout storage, then run the unchanged registered inference engine."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_role_separated_cost as run


def main():
    if len(sys.argv)<2 or sys.argv[1] not in ('infer','evaluate','replay'):
        raise ValueError('Readout wrapper cannot start or modify training')
    (run.PRIVATE/'allocation').mkdir(parents=True,exist_ok=True)
    run.main()


if __name__=='__main__':main()
