"""Explain risk-pass changes after exact adjacent-window budget reallocation."""

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_opendd_paired_window_budget as parent

base, study, np = parent.base, parent.study, parent.np
NAMES = ('completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation')


def reasons(row):
    return {f'{name}:{"undefined" if row[name] is None else "above_2pct"}'
            for name in NAMES if row[name] is None or row[name] > .02}


def run():
    key, cohorts, *_ = parent.register()
    f, r = (parent.OUT/f'{k}.json' for k in ('freeze', 'readout'))
    lock, out = map(study.read, (f, r))
    if (lock['registration'] != key or out['freeze'] != study.artifact(f)
            or not out['original_2pct'] or not out['offline_diagnostic']):
        raise ValueError('Require the bound paired-window result')
    src = parent.exp.loader(cohorts)
    arms = ('additive', 'joint')
    totals = {k: Counter() for k in arms}
    failures = {k: Counter() for k in arms}
    for site in parent.SITES:
        d = src[site]
        cut = np.quantile(np.concatenate([x['e'] for s, x in src.items() if s != site]), [.25, .75])
        for seed in parent.SEEDS:
            with np.load(study.checked(lock['banks'][f'{site}_{seed}']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'], d['q'])
                cv = z['cv']
                metrics = {k: study.policy(z[k], cv, cv, d['y'], d['v'], d['rec'], easy_cut=cut[0], hard_cut=cut[1])
                           ['recordings'] for k in parent.ARMS}
            for k in arms:
                for a, b in zip(metrics[f'fixed_{k}'].values(), metrics[f'pooled_{k}'].values(), strict=True):
                    x, y = reasons(a), reasons(b)
                    if bool(x) == a['supported'] or bool(y) == b['supported']:
                        raise ValueError('Attribution disagrees with original scorer')
                    totals[k][f'{int(a["supported"])}_to_{int(b["supported"])}'] += 1
                    if not x and y:
                        failures[k].update(y)
    for k, c in totals.items():
        if (c['1_to_1']+c['1_to_0'], c['1_to_1']+c['0_to_1']) != (
                out['counts'][f'fixed_{k}']['risk'], out['counts'][f'pooled_{k}']['risk']):
            raise ValueError('Risk transition totals do not replay')
    result = dict(result_source='fresh_run', parent=study.artifact(r), freeze=study.artifact(f),
                  code=study.artifact(Path(__file__)), transitions=totals, lost_pass_reasons=failures,
                  multiple_reasons_allowed=True, no_threshold_tuning=True, deployment_changed=False)
    base.immutable(parent.OUT/'risk_attribution.json', result)
    print(study.json.dumps(dict(transitions=totals, lost_pass_reasons=failures), indent=2), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
