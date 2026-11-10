"""Locate Songdo release failures without new fits, cutoffs or outcome access."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_songdo_frozen_menu_capacity as archived
from src.evaluation import m3w_frozen_score_ranking as rank

old = archived.old
OUT = archived.PUBLIC/'songdo_score_ranking_v1'


def main():
    evidence, frozen, binding, _ = archived.verified_archive(archived.ARCHIVE)
    arms = rank.diagnostic(evidence, frozen, old)
    row = dict(result_source='fresh_run_on_cached_verified_scalar_scores', **binding,
        protocol_sha256=archived.sha256((OUT/'protocol.md').read_bytes()).hexdigest(), arms=arms,
        calibration_flights=40, confirmation_flights=frozen['flights'], components=frozen['source_components'],
        bootstrap_draws=len(evidence['source_resamples']), seeds=old.SEEDS,
        original_risk_limit=old.RISK_LIMIT, new_labels_read=False, threshold_changed=False,
        deployment_changed=False, new_confirmation=False)
    raw = (json.dumps(row, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    target = OUT/'readout.json'
    if target.exists() and target.read_bytes() != raw:
        raise FileExistsError('Refusing to overwrite a different ranking receipt')
    if not target.exists():
        with target.open('xb') as f:
            f.write(raw)
    for arm, v in arms.items():
        print(arm, v['calibration'], v['confirmation']['dual'])


if __name__ == '__main__':
    main()
