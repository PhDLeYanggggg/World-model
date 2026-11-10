"""Compile the family and recipient controls together in a separate draft."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_family_candidate as previous

builder = previous.builder
OUT = previous.OUT.parent/'evidence_manuscript_v186_policy_anchors'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'opendd_vehicle_policy_anchor_v1'
START = r'\paragraph{Match the training path family.}'
END = r'\begin{table}[t]'+'\n'+r'\centering\small'+'\n'+r'\begin{tabular}{@{}lrr@{}}'


def inputs():
    digest = builder.main.digest
    hashes = {f: digest((STUDY/f'{f}.json').read_bytes()) for f in (
        'registration', 'freeze', 'readout', 'additive_freeze', 'topk_capacity')}
    d = json.loads((STUDY/'readout.json').read_bytes())
    if (any(d[k] != hashes[k] for k in ('registration', 'freeze')) or d['external_outcomes_read']
            or d['independent_confirmation'] or not d['original_2pct']
            or d['totals'] != dict(views=8874, successful_nominees=420) or d['fits'] != 252):
        raise ValueError('Require complete policy-anchor comparison')
    top = json.loads((STUDY/'topk_capacity.json').read_bytes())
    if (top['top_k'] != [1, 3, 6, 24] or top['external_outcomes_read'] or not top['no_policy_or_threshold_changed']
            or top['summary']['hits'][0] != 420 or top['summary']['hits'][-1] != 1393):
        raise ValueError('Require the frozen-shortlist capacity check')
    return {**d, 'topk': top['summary']}, hashes


def revise(source, d):
    if source.count(START) != 1 or source.count(END) != 1 or source.index(START) > source.index(END):
        raise ValueError('Expected one family paragraph before the unchanged table')
    arms = d['counts']
    r, p = (arms[a]['ego_count'] for a in ('random_budget', 'policy_budget'))
    u, v = (arms[a]['ego_count'] for a in ('random_pointwise', 'policy_pointwise'))
    if len({r['count'], p['count'], u['count'], v['count'], arms['margin']['ego_count']['count']}) != 1:
        raise ValueError('Held selection counts changed')
    q = d['paired']['policy_budget_minus_random_budget_ego_count']['lower_pp']
    n = d['training']['examples']
    hits, cap = d['topk']['hits'], d['topk']['caps'][0]
    edit = rf'''
\paragraph{{Match the training proposals.}}
The incumbent uses learned forecasts in 98.34\% of its slots.
Matching the donor path family increases strict successes from 40
to 44 for budget ranking at 526 held rotations, still below the
action-margin control's 54. We then use one draw per recording
and select the training actors with a nested copy of the additive
rule, fixing the query, families and radius. Across {n:,} donor
draws the actor set changes in {d['training']['changed_actor_set']:,}.
With random versus policy actors, pointwise training gives
{u.get('strict', 0):,} versus {v.get('strict', 0):,} strict successes,
and budget ranking gives {r.get('strict', 0):,} versus
{p.get('strict', 0):,}, on the identical held nominees.
The policy-minus-random ranking utility is \({q['point']:+.6f}\) pp
[{q['ci95'][0]:+.6f}, {q['ci95'][1]:+.6f}].

\paragraph{{Keep more than one nominee.}}
The first three angles in the frozen score order contain a strict
success in {hits[1]:,} views, up from {hits[0]:,} for one angle.
At the same recording quotas, their hindsight ceiling is {cap[1]:,}
instead of {cap[0]:,}. This measures room for a selector; it is not
performance of a selector trained on the shortlist.

'''
    a, b = source.split(START, 1)
    _, c = b.split(END, 1)
    return a+edit+END+c, ('Match the training proposals', 'three angles')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=previous.OUT, read_inputs=inputs, revise_source=revise)
