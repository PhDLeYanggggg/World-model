"""Describe optimal margins in recordings still negative after OLS4 blending."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_ols_blend_capacity as blend
from src.evaluation.m3w_selection_forecast_readout import errors

run, prev, np = blend.run, blend.prev, blend.np
NEGATIVE = 'negative_maximin_upper_bound'


def describe(values):
    if not values:
        return None
    x = np.asarray(values)
    return dict(median=float(np.median(x)), min=float(x.min()), max=float(x.max()))


def main():
    _, _ = blend.register()
    analysis = prev.receipt(blend.PUBLIC, 'replay')
    d = run.population()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    history = d['history']
    cv = np.arange(1, 13)[None, :, None]*(history[:, -1]-history[:, -2])[:, None]
    y, mask = lab['target_eval']-d['origin'][:, None], lab['valid']
    cv_error, _ = errors(cv, y, mask)
    roles = (blend.ARMS[0], blend.ARMS[1], blend.ARMS[2])
    summaries = {b+'_vs_'+a: {'paired_optima': [], 'both_negative': 0} for a, b in zip(roles, roles[1:])}
    for r, (ix, _) in records.items():
        _, inv, size = np.unique(d['queries'][ix], return_inverse=True, return_counts=True)
        known = mask[ix].any(1)
        denominator = float((1./size[inv[known]]) @ cv_error[ix[known]])
        for seed in prev.SEEDS:
            p = blend.PRIVATE/f'{seed}_{r}.json'
            checkpoint = run.read(p)
            if checkpoint['binding'] != dict(registration=run.sha(blend.REG),
                                               native=run.sha(blend.dual.PRIVATE/p.name)):
                raise ValueError('Cached margin lineage differs')
            for a, b in zip(roles, roles[1:]):
                x, z = (checkpoint['result'][k] for k in (a, b))
                if x['status'] == z['status'] == NEGATIVE:
                    cell = summaries[b+'_vs_'+a]
                    cell['both_negative'] += 1
                    if x['solver_status'] == z['solver_status'] == 0:
                        if denominator <= 0:
                            raise ValueError('Expected positive CV reference for a risk-feasible view')
                        cell['paired_optima'].append([100.*q['upper_gain_mass']/denominator for q in (x, z)])
    report = {}
    for k, s in summaries.items():
        before = [x[0] for x in s['paired_optima']]
        after = [x[1] for x in s['paired_optima']]
        effect = [b-a for a, b in zip(before, after, strict=True)]
        if min(effect, default=0.) < -1e-4:
            raise ValueError('A nested menu cannot lower a certified exact optimum')
        report[k] = dict(both_negative=s['both_negative'], both_optimal=len(effect),
            before_optimum_pp=describe(before), after_optimum_pp=describe(after),
            difference_pp=describe(effect), gains_over_0p01_pp=sum(x > .01 for x in effect),
            after_within_1pp_of_zero=sum(x > -1. for x in after))
    out = blend.PUBLIC/'negative_margin_analysis.json'
    run.once(out, run.stable(dict(contrasts=report, views=analysis['views'],
        result_source='posthoc_paired_optimal_margins_on_both_negative_development_views',
        inference_or_gates_changed=False, independent_confirmation=False,
        primary_analysis_sha256=run.sha(blend.PUBLIC/'analysis.json'))))


if __name__ == '__main__':
    main()
