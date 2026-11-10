"""Compile an anonymous candidate with the direct action-value comparison."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_context_candidate as parent

builder = parent.previous
OUT = parent.OUT.parent/'evidence_manuscript_v182_action_value'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'opendd_vehicle_action_value_v1'
ANCHOR = "against ranking angles by the neighbor-free forecast's individual gain."


def inputs():
    digest = builder.main.digest
    hashes = {f: digest((STUDY/f'{f}.json').read_bytes()) for f in ('registration', 'freeze', 'readout')}
    d = json.loads((STUDY/'readout.json').read_bytes())
    if (any(d[f] != hashes[f] for f in ('registration', 'freeze')) or d['external_outcomes_read']
            or d['independent_confirmation'] or not d['original_2pct']
            or (d['fits'], d['updates'], d['recordings'], d['views']) != (126, 25200, 493, 8874)):
        raise ValueError('Expected the complete frozen action-value comparison')
    return d, hashes


def revise(source, d):
    if source.count(ANCHOR) != 1:
        raise ValueError('Expected exactly one dynamics paragraph')
    n = d['counts']['context']['rotations']
    if not all(d['counts'][a]['rotations'] == n for a in ('ego_at_context', 'constant_at_context', 'joint_at_context')):
        raise ValueError('Mismatched fixed-actor comparison')
    change = d['paired']['context_minus_original']['lower_pp']
    fixed = d['paired']['context_minus_constant_at_context']['lower_pp']
    ego = d['paired']['context_minus_ego_at_context']['lower_pp']
    joint = d['paired']['joint_at_context_minus_context']['lower_pp']
    rel = d['paired']['joint_at_context_minus_context']['relative_lower']
    c, j = (d['counts'][f'{a}_calibrated']['rotations'] for a in ('context', 'joint'))
    text = rf'''

\paragraph{{Learning the value of a change.}}
Instead of predicting a trajectory, we train on the error difference
between a legal angle and keeping the original direction. The targets
separate observed gain, harm including missing-step debit, and the
lower gain in relative motion. For the context model the conservative
gain over no rotation is \({change['point']:+.4f}\) pp.
On the same {n:,} rotating views, it beats the donor-selected
constant angle by \({fixed['point']:+.4f}\) pp (95\% location interval
[{fixed['ci95'][0]:+.4f}, {fixed['ci95'][1]:+.4f}]).
It still trails the ego-only model at those actors
(\({ego['point']:+.4f}\) pp). Adding the relative-gain term at the same
actors gains \({joint['point']:+.4f}\) pp in trajectory utility but loses
\({abs(rel['point']):.4f}\) native units of relative-motion utility.
After calibration, the context and joint rules rotate in {c:,}
and {j:,} of 8,874 views.
'''
    return source.replace(ANCHOR, ANCHOR+text, 1), ('Learning the value of a change', 'donor-selected')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=parent.OUT, read_inputs=inputs, revise_source=revise)
