"""Decompose the frozen WEPDTOF readout without reopening geometry."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_external_readout as exp

base,study,np = exp.base,exp.study,exp.np
OUT = exp.OUT


def explain(z):
    if not z:
        return 'no_online_history'
    r = z['risk']
    if r['selected']==0:
        return 'no_intervention'
    if r['completion_all_risk_upper'] is None:
        return 'undefined_all_reference'
    if r['completion_easy_risk_upper'] is None or r['known_easy_net_degradation'] is None:
        return 'undefined_easy_reference'
    if r['supported']:
        return 'risk_pass'
    return 'violates_original_risk'


def main():
    p = OUT/'external_readout.json'
    freeze = OUT/'augmented_freeze.json'
    x = study.read(p)
    if (x['forecast_freeze']!=base.sha(freeze) or not x['external_scoring_completed']
            or x['external_tuning'] or not x['original_2pct']):
        raise ValueError('Require the original once-only outcome receipt')
    per = {}
    rows = x['rows']
    for name in ('cv','ols4','spline','joint','fixed_joint','fixed_independent','fixed_uniform','fixed_no_pair'):
        a = [r['arms'].get(name) for c in rows for r in c['seeds'].values()]
        reasons = dict(Counter(explain(z) for z in a))
        valid = [z for z in a if z and z['lower_vs_cv'] is not None]
        per[name] = dict(reasons=reasons, any_measured_risk_failure=sum(
            bool(z and any(z['risk'][k] is not None and z['risk'][k]>.02 for k in
                           ('completion_all_risk_upper','completion_easy_risk_upper','known_easy_net_degradation')))
            for z in a), known_positive_cv=sum(z['known_vs_cv']>0. for z in valid),
            lower_positive_cv=sum(z['lower_vs_cv']>0. for z in valid),
            known_positive_ols4=sum(z['known_vs_ols4']>0. for z in valid),
            lower_positive_ols4=sum(z['lower_vs_ols4']>0. for z in valid),
            median_missing_debit_cv_pp=float(np.median([z['known_vs_cv']-z['lower_vs_cv'] for z in valid])) if valid else None,
            wholly_unknown_edits=sum(z['risk']['unknown_selected'] for z in a if z))
    source_counts = {}
    for s in sorted({z['source'] for z in rows}):
        g = [r for r in rows if r['source']==s]
        source_counts[s] = {k:sum(z[k] for z in g) for k in ('online','matched','full','missing','reference_histories')}
    z = dict(parent=study.artifact(p), per_arm=per, source_counts=source_counts,
        no_new_target_access=True, no_new_policy_selection=True, original_2pct=True)
    base.immutable(OUT/'failure_breakdown.json',z)
    print(per,flush=True)


if __name__=='__main__':
    main()
