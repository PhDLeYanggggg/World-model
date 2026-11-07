"""Attribute errors on fixed raw actions without producing a new policy."""
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_nested_action_calibration as study

run, np, cal = study.run, study.np, study.cal


def main():
    start = time.monotonic(); study.register(); refs = run.register(); d = run.population()
    lab = run.old.prior.arrays(('target_eval', 'valid')); cuts = run.cuts_by_source()
    score = run.read(study.base.PUBLIC/'analysis.json')
    previous = {(r['seed'], r['recording']): r['metrics']['filter']['risk'] for r in score['records']}
    records = []
    for r in run.read(run.PUBLIC/'freeze.json')['seeds']:
        seed = r['seed']; floor, paths = run.candidates(d, refs, seed)
        with np.load(run.checked(r['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids']); names, coef = z['names'], z['coefficients']
        raw = next(v for v in run.read(study.base.PUBLIC/'freeze.json')['choices'] if v['seed'] == seed)
        with np.load(run.checked(raw['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids']); choice = z['choice']
        for s, c in zip(names, coef):
            ix = np.flatnonzero(d['sites'] == s); p, cv = paths[ix], floor[ix]; rec = d['recordings'][ix]
            x, scale = run.model.descriptors(d['history'][ix], p, cv, d['crowd'][ix])
            radius = np.linalg.norm(p - cv[:, None], axis=-1).max(2)
            pred = cal.predicted_components(x, c, scale, radius)
            truth = cal.observed_components(p, cv, lab['target_eval'][ix]-d['origin'][ix, None], lab['valid'][ix], radius, cuts[s]['easy'])
            for v in cal.residuals(pred, truth, radius, choice[ix], rec):
                ph, pr, eh, er = v['prediction']; oh, rr, ah, ar = v['observed']
                def risk(num, den): return num / den if den > 0 else None
                val = risk(ah, ar); old = previous[seed, v['recording']]['completion_easy_risk_upper']
                if (old is None) != (val is None) or old is not None and not np.isclose(old, val, rtol=1e-12):
                    raise ValueError('Actual components must reproduce original easy risk')
                v['risks'] = dict(predicted=risk(eh, er), observed=val,
                    observed_harm_only=risk(ah, er), observed_reference_only=risk(eh, ar))
                v.update(seed=seed, source=str(s)); records.append(v)
        print(dict(seed=seed, complete_nonempty=len(records)), flush=True)
    if len(records) != score['nonempty_views']:
        raise ValueError('Preserve nonempty population and explicit empty count')
    summary = {}
    for key in records[0]['risks']:
        vals = [r['risks'][key] for r in records]
        summary[key] = dict(defined=sum(v is not None for v in vals), over_2=sum(v is not None and v > .02 for v in vals))
    p, y = (np.sum([r[key] for r in records], axis=0) for key in ('prediction', 'observed'))
    path = study.PUBLIC/'selected_component_audit.json'
    run.once(path, run.stable(dict(records=records, summary=summary, empty_views=102-len(records),
        sum_predicted=p.tolist(), sum_observed=y.tolist(),
        easy_excess_underestimate=dict(harm=float(y[2]-p[2]), reference=float(.02*(p[3]-y[3]))),
        diagnostic_only=True, new_actions=False, original_rules_unchanged=True)))
    run.once(study.PUBLIC/'selected_component_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
        artifact=run.artifact(path)))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
