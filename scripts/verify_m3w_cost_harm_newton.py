"""Run the same independent scalar verifier in the solver-amended namespace."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import verify_m3w_cost_aligned_positive_harm as verifier

if __name__ == '__main__':
    verifier.PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cost_harm_newton_v1'
    verifier.main()
