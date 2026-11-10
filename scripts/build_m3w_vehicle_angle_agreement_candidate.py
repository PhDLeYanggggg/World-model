"""Add the angle agreement experiment to a separate manuscript revision."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_shortlist_candidate as previous

builder = previous.builder
OUT = previous.OUT.parent/'evidence_manuscript_v188_angle_agreement'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'opendd_vehicle_angle_agreement_v1'


def inputs():
    raw = {k: (STUDY/f'{k}.json').read_bytes() for k in ('registration', 'freeze', 'readout')}
    hashes = {k: builder.main.digest(v) for k, v in raw.items()}
    f, d = (json.loads(raw[k]) for k in ('freeze', 'readout'))
    if (f['registration'] != hashes['registration'] or d['freeze'] != hashes['freeze']
            or not f['before_held_scoring'] or d['directions'] != 126 or d['external_outcomes_read']
            or d['independent_confirmation'] or d['deployment_changed'] or not d['original_2pct']):
        raise ValueError('Require frozen angle agreement on the development comparison')
    return d, hashes


def revise(source, d):
    start = 'the strict criterion. The policy stays unchanged.'
    if source.count(start) != 1:
        raise ValueError('Prior shortlist paragraph changed')
    q, r = d['counts']['ego_count'], d['counts']['fixed_15pct']
    n, m = d['capacity']['ego_count'], d['capacity']['fixed_15pct']
    q_cal, r_cal = q['calibrated_agreement'], r['calibrated_agreement']
    edit = rf'''the strict criterion.

\paragraph{{Predict agreement before changing the angle.}}
The same trajectory and relative-motion heads can be used
as separate checks against the first angle. At the 526 quota,
{n['qualified']} views have a hindsight alternative improving both
conservative measures while passing risk and both references.
The agreement rule finds {q['agreement']['qualified_joint_gain']} of these
among {q['agreement']['swaps']} angle replacements. Opposite-site calibration
reduces the replacements to {q_cal['swaps']}, with
{q_cal.get('qualified_joint_gain', 0)} qualified improvements.
At 1,368 rotations the corresponding hindsight count is
{m['qualified']}, and the calibrated rule finds
{r_cal.get('qualified_joint_gain', 0)} among {r_cal['swaps']} replacements.
Neither budget shows a joint trajectory and relative-motion improvement
under calibration. The intervention masks and policy stay unchanged.'''
    return source.replace(start, edit, 1), ('Predict agreement before changing', 'hindsight alternative')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=previous.OUT, read_inputs=inputs, revise_source=revise)
