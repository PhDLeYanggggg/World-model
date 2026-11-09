"""Summarize the failure and paired pass counts without reopening targets."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_slate_imitation as exp

study,base,np = exp.study,exp.base,exp.np


def status(r):
    if not r['risk']['selected']:
        return 'abstained'
    a = r['risk']
    if a['completion_easy_risk_upper'] is None:
        return 'undefined_easy'
    if not a['supported']:
        return 'risk_failed'
    return 'dual_pass' if r['dual_pass'] else 'utility_failed'


def interval(v):
    x = np.asarray(v)
    a = np.random.default_rng(3183).choice(x,size=(3000,len(x))).mean(1)
    return dict(point=float(x.mean()),ci95=np.quantile(a,[.025,.975]).tolist(),sources=len(x))


def main():
    p = exp.OUT/'readout.json'
    a = study.read(p)
    f = study.read(study.checked(a['freeze']))
    reg = exp.OUT/'registration.json'
    if a['registration']!=base.sha(reg) or f['registration']!=a['registration'] or not a['original_2pct']:
        raise ValueError('Require the completed frozen readout')
    cache = study.read(study.checked(a['recording_score_cache']))
    if cache['registration']!=a['registration']:
        raise ValueError('Score cache registration differs')
    rows = [r for group in cache['location_rows'].values() for r in group]
    if len(rows)!=606 or len({(r['record'],r['seed']) for r in rows})!=606:
        raise ValueError('Expected 202 recordings times three seeds')
    sites = sorted({r['site'] for r in rows})
    contrasts, failures = {},{}
    for k in ('joint','independent','matched_independent','matched_uniform'):
        failures[k] = dict(Counter(status(r['arms'][k]) for r in rows if k in r['arms']))
        if k=='joint':
            continue
        matched = [r for r in rows if k in r['arms']]
        values = {s:[r for r in matched if r['site']==s] for s in sites}
        dual = lambda z,m: z['arms'][m]['dual_pass']
        pairs = Counter((dual(r,'joint'),dual(r,k)) for r in matched)
        per_site = {}
        for s, records in values.items():
            if not records:
                continue
            effect = [float(dual(r,'joint'))-float(dual(r,k)) for r in records]
            # Tail and relative-displacement differences stay in native source units.
            tail = [r['arms']['joint']['p95_ade_px']-r['arms'][k]['p95_ade_px'] for r in records
                    if r['arms']['joint']['p95_ade_px'] is not None and r['arms'][k]['p95_ade_px'] is not None]
            pair = [v['joint']-v[k] for r in records for v in [r['pair_relative_error_px']]
                    if v['joint'] is not None and v[k] is not None]
            per_site[s] = dict(dual_rate_delta_pp=float(100.*np.mean(effect)),paired_views=len(effect),
                median_delta_p95_native=float(np.median(tail)) if tail else None,
                median_delta_pair_relative_native=float(np.median(pair)) if pair else None,
                estimable_tail=len(tail),estimable_pair=len(pair))
        contrasts[k] = dict(paired_views=len(matched),joint_only=pairs[True,False],control_only=pairs[False,True],
            both=pairs[True,True],source_rate=interval([z['dual_rate_delta_pp'] for z in per_site.values()]),
            per_site=per_site)
    base.immutable(exp.OUT/'diagnostics.json',dict(readout=study.artifact(p),same_score_cache=True,
        original_2pct=True,no_new_target_access=True,matched_contrasts=contrasts,failures=failures))
    print({k:(r['paired_views'],r['joint_only'],r['control_only'],r['source_rate']) for k,r in contrasts.items()})


if __name__=='__main__':
    main()
