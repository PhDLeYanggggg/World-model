"""Run the frozen support diagnostic with a separate heartbeat directory."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import diagnose_m3w_decision_cost_support as diagnostic

if __name__=='__main__':
    diagnostic.run.PRIVATE=diagnostic.run.PRIVATE/'support_runtime'
    diagnostic.diagnose()
