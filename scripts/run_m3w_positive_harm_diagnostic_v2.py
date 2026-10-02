"""Precision-only amendment; reuse the frozen streaming runner unchanged."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_positive_harm_diagnostic as runner
from src.world_model import m3w_positive_harm_diagnostic_v2 as api

NAME = 'european_positive_harm_diagnostic_v2'
PUBLIC = runner.PUBLIC.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
OLD = runner.PUBLIC


def registration():
    old = json.loads((OLD/'registration.json').read_text())
    for path, digest in old['bindings'].items(): assert runner.sha(ROOT/path) == digest
    assert runner.prior.registration() == json.loads((runner.prior.PUBLIC/'registration.json').read_text())
    paths = runner.parent.forest.closure(ROOT, ['scripts.run_m3w_positive_harm_diagnostic_v2'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_positive_harm_diagnostic_v2.py']
    return dict(bindings={str(p.relative_to(ROOT)): runner.sha(p) for p in paths},
        original_registration_sha256=runner.sha(OLD/'registration.json'),
        parent_verification_sha256=runner.sha(runner.prior.PUBLIC/'verification.json'),
        preserved_v1_partial_results={str(p.relative_to(ROOT)): runner.sha(p) for p in sorted((OLD/'groups').glob('*.json'))},
        source_heads=72, training=False, policy_selection=False, independent_roles_read=False,
        amendment='Retain registered target precision and sequential normalization; no tolerance change')


def main():
    # Only diagnosis and output namespace change. Inference and readout stay frozen.
    runner.api, runner.NAME, runner.PUBLIC, runner.CONFIG = api, NAME, PUBLIC, CONFIG
    runner.PRIVATE = runner.prior.PRIVATE.parent/NAME
    runner.registration = registration
    runner.main()


if __name__ == '__main__': main()
