"""Compile the matched shortlist comparison as a separate anonymous draft."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_vehicle_policy_anchor_candidate as previous

builder = previous.builder
OUT = previous.OUT.parent/'evidence_manuscript_v187_shortlist'
RECEIPT = builder.main.PUBLIC/OUT.name/'receipt.json'
STUDY = builder.main.PUBLIC/'opendd_vehicle_shortlist_v1'
START = r'\paragraph{Keep more than one nominee.}'
END = previous.END


def inputs():
    digest = builder.main.digest
    files = {k: (STUDY/f'{k}.json').read_bytes() for k in ('registration', 'freeze', 'readout', 'fixed_mask_capacity')}
    hashes = {k: digest(v) for k, v in files.items()}
    d, cap = (json.loads(files[k]) for k in ('readout', 'fixed_mask_capacity'))
    if (any(d[k] != hashes[k] for k in ('registration', 'freeze')) or cap['freeze'] != hashes['freeze']
            or d['external_outcomes_read'] or d['independent_confirmation'] or not d['original_2pct']
            or d['totals'] != dict(views=8874, strict_top1=420, strict_top3=802)):
        raise ValueError('Require frozen shortlist comparison and exact same-mask capacity')
    return {**d, 'fixed_mask_capacity': cap['counts']}, hashes


def revise(source, d):
    if source.count(START) != 1 or source.count(END) != 1 or source.index(START) > source.index(END):
        raise ValueError('Expected the existing shortlist paragraph and untouched table')
    r = d['counts']
    cap = d['fixed_mask_capacity']['ego_count']
    fixed = ('margin', 'ols_fixed', 'strict_fixed', 'utility_fixed')
    a, b, c, e = [r[k]['ego_count'].get('strict', 0) for k in fixed]
    u, v = (r[k]['ego_count'].get('strict', 0) for k in ('utility_top1', 'utility_top3'))
    q = d['paired']['utility_fixed_minus_margin_ego_count']['lower_pp']
    r15 = d['paired']['utility_fixed_minus_margin_fixed_15pct']['lower_pp']
    t = r['utility_top3']['calibrated']
    edit = rf'''
\paragraph{{Choose from a shortlist without spending more edits.}}
The frozen top three cover a strict success in 802 views,
compared with 420 for top one. On the same 526 recording masks,
the hindsight cap rises from {cap['strict_top1']} to {cap['strict_top3']}.
In actual choice on those masks, top one, OLS4 agreement,
strict-score selection and utility-aware selection give
{a}, {b}, {c} and {e} strict successes. The last choice's
paired conservative trajectory gain against top one is
\({q['point']:+.6f}\) pp [{q['ci95'][0]:+.6f}, {q['ci95'][1]:+.6f}].
At the 15\% quota the same comparison gives
\({r15['point']:+.6f}\) pp [{r15['ci95'][0]:+.6f}, {r15['ci95'][1]:+.6f}].
Relative-motion utility, however, falls at both quotas.
Reranking recordings as well at the 526 quota gives {u} strict
successes with top one and {v} with top three. Calibrated top three
rotates {t.get('count', 0)} views, of which {t.get('strict', 0)} meet
the strict criterion. The policy stays unchanged.

'''
    a, b = source.split(START, 1)
    _, c = b.split(END, 1)
    return a+edit+END+c, ('Choose from a shortlist', 'hindsight cap')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), RECEIPT, parent_dir=previous.OUT, read_inputs=inputs, revise_source=revise)
