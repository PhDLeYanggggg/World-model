"""Explicit versioned runner: identical probe, repaired internal rollout features."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.audit_m3w_normalization_response as audit
from src.world_model.m3w_observed_unit_frame_v2 import observed_unit_frame

# The registration hashes both runners and both frame implementations.
audit.observed_unit_frame = observed_unit_frame

if __name__ == '__main__':
    audit.main()
