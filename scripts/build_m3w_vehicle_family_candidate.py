"""Compile a separate manuscript with the training-family ablation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_nominee_candidate as previous

builder = previous.builder
OUT = previous.OUT.parent/'evidence_manuscript_v185_family_alignment'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'opendd_vehicle_family_alignment_v1'
ANCHOR = 'checks and positive incremental gain.'


def inputs():
    digest = builder.main.digest
    hashes = {k: digest((STUDY/f'{k}.json').read_bytes()) for k in ('registration', 'freeze', 'readout')}
    d = json.loads((STUDY/'readout.json').read_bytes())
    if (any(d[k] != hashes[k] for k in ('registration', 'freeze')) or d['external_outcomes_read']
            or d['independent_confirmation'] or not d['original_2pct']
            or d['totals'] != dict(views=8874, successful_nominees=420) or d['fits'] != 126):
        raise ValueError('Expected the complete family-aligned readout')
    return d, hashes


def revise(source, d):
    if source.count(ANCHOR) != 1:
        raise ValueError('Expected one exact nominee paragraph')
    a = d['training']
    percent = [100.*a[f'{k}_learned_mass']/a['actor_slots'] for k in ('before', 'after')]
    c, j = (d['counts'][k]['ego_count'] for k in ('aligned_pointwise', 'aligned_budget'))
    if c['count'] != j['count'] or c['count'] != d['counts']['margin']['ego_count']['count']:
        raise ValueError('Changed held rotation budget')
    q = d['paired']['aligned_budget_minus_budget_rank_ego_count']['lower_pp']
    para = rf'''

\paragraph{{Match the training path family.}}
The original policy selects learned forecasts for 98.34\% of
its recipient slots. We replay the same training actors and
queries, substituting a learned path only when it is legal.
This changes the learned-family share from {percent[0]:.1f}\% to
{percent[1]:.1f}\%. At the same {c['count']:,} held rotations, the
aligned pointwise and ranking heads give {c.get('strict', 0):,}
and {j.get('strict', 0):,} strict successes. The aligned-minus-uniform
ranking contrast in conservative trajectory utility is
\({q['point']:+.6f}\) pp
[{q['ci95'][0]:+.6f}, {q['ci95'][1]:+.6f}]. The held paths, budgets,
model and losses are unchanged.
'''
    return source.replace(ANCHOR, ANCHOR+para), ('Match the training path', 'ranking contrast')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=previous.OUT, read_inputs=inputs, revise_source=revise)
