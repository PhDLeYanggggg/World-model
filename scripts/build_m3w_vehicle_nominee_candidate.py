"""Add the exact-action ranking ablation to a separate anonymous draft."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_opportunity_candidate as previous

builder = previous.builder
OUT = previous.OUT.parent/'evidence_manuscript_v184_nominee_rank'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'opendd_vehicle_nominee_rank_v1'
ANCHOR = 'comparisons with identical proposals, not external confirmation.'


def inputs():
    digest = builder.main.digest
    hashes = {k: digest((STUDY/f'{k}.json').read_bytes()) for k in ('registration', 'freeze', 'readout', 'teacher_freeze')}
    result = json.loads((STUDY/'readout.json').read_bytes())
    teachers = json.loads((STUDY/'teacher_freeze.json').read_bytes())
    if (any(result[k] != hashes[k] for k in ('registration', 'freeze'))
            or result['external_outcomes_read'] or result['independent_confirmation']
            or not result['original_2pct'] or teachers['external_outcomes_read']
            or any(len(teachers['checkpoints'][a]) != 105 for a in ('forecast', 'ego'))
            or (result['fits'], result['totals']['successful_nominees']) != (126, 420)):
        raise ValueError('Expect complete nested nominee comparison')
    return result, hashes


def revise(source, d):
    if source.count(ANCHOR) != 1:
        raise ValueError('Expected the intact vehicle screen paragraph')
    fixed = d['counts']['margin']['ego_count']
    a, b = (d['counts'][k]['ego_count'] for k in ('pointwise', 'budget_rank'))
    if a['count'] != b['count'] or a['count'] != fixed['count']:
        raise ValueError('Fixed action counts changed')
    g = d['paired']['budget_rank_minus_margin_ego_count']['lower_pp']
    r = d['counts']['budget_rank']['calibrated']
    para = rf'''

\paragraph{{Train on the exact nominee.}}
Nested source exclusions let us predict the donor nominee before
computing its outcome. A pointwise success head and a within-source
top-negative ranking head then score the same held proposals.
At {fixed['count']:,} rotations, they realize {a.get('strict', 0):,} and
{b.get('strict', 0):,} strict successes, compared with {fixed['strict']:,}
for action margin. The ranking-minus-margin conservative trajectory
contrast is \({g['point']:+.6f}\) pp
[{g['ci95'][0]:+.6f}, {g['ci95'][1]:+.6f}].
After separate calibration, the ranker rotates {r.get('count', 0):,}
views, of which {r.get('incremental', 0):,} meet risk, both reference
checks and positive incremental gain.
'''
    return source.replace(ANCHOR, ANCHOR+para), ('Train on the exact', 'strict successes')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=previous.OUT, read_inputs=inputs, revise_source=revise)
