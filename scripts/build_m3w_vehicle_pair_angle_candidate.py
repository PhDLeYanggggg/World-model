"""Compile the matched angle-target ablation as a separate manuscript."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_angle_agreement_candidate as prior

builder = prior.builder
OUT = prior.OUT.parent/'evidence_manuscript_v189_pair_angle'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'opendd_vehicle_pair_angle_v1'


def inputs():
    raw = {k: (STUDY/f'{k}.json').read_bytes() for k in ('registration', 'freeze', 'readout', 'relative_bounds')}
    hashes = {k: builder.main.digest(v) for k, v in raw.items()}
    f, d, bounds = (json.loads(raw[k]) for k in ('freeze', 'readout', 'relative_bounds'))
    if (f['registration'] != hashes['registration'] or d['freeze'] != hashes['freeze']
            or not f['before_held_scoring'] or d['fits'] != 126 or d['updates'] != 25200 or d['directions'] != 126
            or d['external_outcomes_read'] or d['independent_confirmation'] or d['deployment_changed']
            or not d['original_2pct'] or bounds['readout'] != hashes['readout'] or bounds['freeze'] != hashes['freeze']):
        raise ValueError('Require frozen, fully fitted matched angle targets')
    return dict(d, bounds=bounds), hashes


def revise(source, d):
    start = 'under calibration. The intervention masks and policy stay unchanged.'
    if source.count(start) != 1:
        raise ValueError('Previous agreement paragraph changed')
    stats = d['training']
    a = d['counts']['anchor_difference']['ego_count']['agreement']
    b = d['counts']['direct']['ego_count']['agreement']
    g = d['direct_minus']['anchor_difference']['ego_count']['agreement']['relative']
    upper = d['bounds']['effects']['ego_count']['anchor_difference']['upper']['point']
    h = d['counts']['direct']['ego_count']['calibrated_agreement']
    edit = rf'''under calibration. The intervention masks and policy stay unchanged.

\paragraph{{Learn the replacement directly.}}
Subtracting two lower bounds is not a lower bound on their
difference. On the same {stats['rows']:,} donor pairs, the direct
bound changes the positive relative-motion label in
{100*stats.get('relative_sign_changed', 0)/stats['rows']:.1f}\% of pairs.
With network, training draws and intervention masks fixed,
the anchor-difference and direct targets yield
{a.get('qualified_joint_gain', 0)} of {a.get('swaps', 0)} and
{b.get('qualified_joint_gain', 0)} of {b.get('swaps', 0)} qualifying
replacements at the 526 quota. Direct minus anchor-difference
has conservative relative-motion gain
\({g['point']:+.6f}\) [{g['ci95'][0]:+.6f}, {g['ci95'][1]:+.6f}].
The missing-label upper bound is \({upper:+.6f}\), so this does not
identify the sign of the full-grid effect.
Calibration retains {h.get('swaps', 0)} direct replacements, of
which {h.get('qualified_joint_gain', 0)} qualify. These are development
comparisons, not a release decision.'''
    return source.replace(start, edit, 1), ('Learn the replacement directly', 'donor pairs')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=prior.OUT, read_inputs=inputs, revise_source=revise)
