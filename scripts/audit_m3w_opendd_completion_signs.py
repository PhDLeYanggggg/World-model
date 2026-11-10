"""Compute both completion endpoints from the exact frozen paired summaries."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_step_visibility_candidate as prior
from scripts import build_m3w_fixed_joint_budget_candidate as budget
from src.evaluation.m3w_completion_contrasts import completion

OUT = prior.main.PUBLIC/'opendd_direct_completion_signs_v1'


def data():
    old, old_hashes = budget.inputs()
    new, new_hashes = prior.inputs()
    if (old['locations'], new['locations']) != (7, 7) or old['seeds'] != new['seeds']:
        raise ValueError('Expected the same seven-location design')
    r = old['summary']
    s = new['summary']['contrasts']['learned_minus_profile']
    rows = dict(support_count=r['original_quota']['contrasts']['supported_merit'],
                support_count_movement=r['movement_matched']['contrasts']['supported_merit'],
                learned_minus_profile=s)
    draws = np.random.default_rng(12773).integers(7, size=(3000, 7)).tolist()
    points = {a: dict(observed=v['ade_available_pp']['location_values'], lower=v['ade_lower_pp']['location_values'])
              for a, v in rows.items()}
    result = {a: completion(**v, draws=draws) for a, v in points.items()}
    for a, v in rows.items():
        for k, metric in [('observed', 'ade_available_pp'), ('lower', 'ade_lower_pp')]:
            x, y = result[a]['summary'][k], v[metric]
            np.testing.assert_allclose([x['point'], *x['ci95']], [y['point'], *y['ci95']], rtol=0., atol=1e-12)
    return dict(points=points, draws=draws), result, dict(movement_readout=old_hashes['readout'], visibility_readout=new_hashes['readout'])


if __name__ == '__main__':
    _, result, hashes = data()
    row = dict(result_source='fresh_completion_analysis_of_verified_paired_effects', results=result,
               hashes=hashes, locations=7, directions=126, selected_views_per_arm=2196,
               bootstrap_draws=3000, original_2pct=True, raw_labels_read=False,
               new_external_labels_opened=False, fit=False, deployment_changed=False)
    raw = prior.main.as_json(row)
    p = OUT/'readout.json'
    if p.exists() and p.read_bytes() != raw:
        raise FileExistsError('Refusing to replace a different frozen completion analysis')
    if not p.exists():
        with p.open('xb') as f:
            f.write(raw)
    for a, v in result.items():
        print(a, v['summary'], v['roster_sign'], v['leave_one_out_signs'])
