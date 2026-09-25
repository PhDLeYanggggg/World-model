"""Label-only decomposition of frozen count-thinning outcomes; never a new policy."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_incremental_joint as run
from scripts.report_m3w_european_floor_relative import dump


def accounting(neural, floor, cv, full, half):
    n, f, c = map(lambda x: np.asarray(x, float), (neural, floor, cv))
    full, half = np.asarray(full), np.asarray(half)
    if (n.ndim != 1 or f.shape != n.shape or c.shape != n.shape or not len(n)
            or full.shape != n.shape or half.shape != n.shape or full.dtype != bool or half.dtype != bool
            or not all(np.isfinite(x).all() for x in (n, f, c)) or np.any(c <= 0) or np.any(half & ~full)):
        raise ValueError('Known positive-CV subset and subset-of-full choices required')
    removed = full & ~half; old = np.where(full, n, f); new = np.where(half, n, f)
    lost = np.where(removed, np.maximum(f-n, 0), 0)
    avoided = np.where(removed, np.maximum(n-f, 0), 0)
    np.testing.assert_allclose((new-old).sum(), lost.sum()-avoided.sum(), rtol=1e-10, atol=1e-10)
    den = float(c.sum())
    return dict(rows=len(n), removed=int(removed.sum()), removed_beneficial=int((removed & (n < f)).sum()),
        removed_harmful=int((removed & (n > f)).sum()), lost_benefit_percent_CV=float(100*lost.sum()/den),
        avoided_harm_percent_CV=float(100*avoided.sum()/den),
        full_degradation_percent_CV=float(100*(old.sum()/den-1)),
        half_degradation_percent_CV=float(100*(new.sum()/den-1)),
        count_thinning_error_change_percent_CV=float(100*(new-old).sum()/den))


def main():
    run.ensure_frozen(); cfg, bcfg, ctx, bid, pid, identity, _ = run.load(); data = ctx[2]
    rows, failures = {}, []
    for g in run.parent.groups(bcfg, ctx, bid, pid):
        with np.load(run.PRIVATE/'decisions'/(g['name']+'.npz'), allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        ids = a['ids']; sites = data['sites'][ids]; cv = data['baseline_ade'][ids, 1]
        def ade(p):
            return run.parent.prior.native_errors(p.astype(float)+data['origin'][ids, None],
                data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))[0]
        floor, neural = ade(g['d'][ids]), ade(g['a']['p'][ids])
        group = {}
        for site in identity['rosters'][g['roles']['readout_fold']]:
            use = (sites == site) & (cv > 0) & (cv <= g['design']['easy_cut'])
            if not use.any(): group[site] = dict(status='no_positive_easy_labels'); continue
            r = accounting(neural[use], floor[use], cv[use], a['full_add'][use], a['half_joint'][use])
            group[site] = r
            if r['half_degradation_percent_CV'] > 2:
                failures.append(dict(group=g['name'], site=site, **r))
        rows[g['name']] = group
    dump(run.PUBLIC/'easy_thinning_accounting.json', dict(result_source='fresh_run_posthoc_arithmetic_frozen_choices',
        no_new_policy=True, groups=rows, failed_locality_views=failures,
        count_thinning_equation='half_minus_full_error = removed_lost_benefit - removed_avoided_harm'))
    print(json.dumps(dict(failures=failures, groups=len(rows)), allow_nan=False))


if __name__ == '__main__': main()
