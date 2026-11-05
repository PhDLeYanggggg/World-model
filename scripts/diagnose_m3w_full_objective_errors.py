"""Post-fit attribution using own-B only; no new selection or held prediction."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_full_objective_risk as run

api, source = run.api, run.source
np, torch = api.np, api.torch


def decompose(before, after, target, weights, allowed, known, rowmask):
    mask = allowed[..., None] & known[:, None, None] & rowmask[:, None, None]
    w = weights[:, None, None]; cost = w*abs(target)*mask
    wrong = lambda p: ((p <= 0) & (target > 0)) | ((p > 0) & (target < 0))
    a, b = wrong(before), wrong(after)
    result = dict(rows=int(rowmask.sum()), newly_wrong_cost=(cost*~a*b).sum((0, 1)).tolist(),
        repaired_wrong_cost=(cost*a*~b).sum((0, 1)).tolist(),
        persistent_wrong_cost=(cost*a*b).sum((0, 1)).tolist())
    for name, pred in (('initial', before), ('final', after)):
        known_mse = w*(pred-target)**2*mask/allowed.sum(1).clip(1)[:, None, None]
        unknown_mse = w*(pred-target)**2*allowed[..., None]*(~known & rowmask)[:, None, None]/allowed.sum(1).clip(1)[:, None, None]
        result[name] = dict(known_MSE=known_mse.sum((0, 1)).tolist(), unknown_bound_MSE=unknown_mse.sum((0, 1)).tolist(),
            false_safe_cost=(cost*(pred <= 0)*(target > 0)).sum((0, 1)).tolist(),
            false_reject_cost=(cost*(pred > 0)*(target < 0)).sum((0, 1)).tolist(),
            sign_cost=(cost*wrong(pred)).sum((0, 1)).tolist())
    np.testing.assert_allclose(np.array(result['final']['sign_cost'])-result['initial']['sign_cost'],
        np.array(result['newly_wrong_cost'])-result['repaired_wrong_cost'], rtol=1e-10, atol=1e-12)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = run.PRIVATE
        _, guard, beat = run.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), 'fitting_attribution')
        _, values = run.register(beat)
        if source.read(run.PUBLIC/'training_freeze.json')['complete_heads'] != 3:
            raise ValueError('Complete fitting before attribution')
        rows = []
        for job in values[3]:
            d, origin = run.inputs(values, job); c = d['causal']
            api.base.exact(origin['observed_mask_hash'], api.parent.digest(d['observed_future']))
            receipt = source.read(run.PUBLIC/'fits'/(job['key']+'.json'))
            s = api.base.read_checkpoint(source.checked(receipt['checkpoint']))
            prepared = api.prepare(d, origin); predictions = {}
            for name, state in (('initial', dict(s, model=s['initial_model'])), ('final', s)):
                model = api.model_from(state); prediction = np.empty(prepared['target'].shape, np.float64)
                for ix, seg in prepared['batches']:
                    guard()
                    with torch.no_grad():
                        prediction[ix] = model(prepared['z'][ix].double(), torch.from_numpy(seg), int(seg.max())+1).numpy()
                predictions[name] = prediction
            kw = dict(before=predictions['initial'], after=predictions['final'], target=prepared['target'].numpy(),
                weights=prepared['weights'].numpy(), allowed=c['allowed'], known=prepared['known'])
            total = decompose(**kw, rowmask=np.ones(len(c['x']), bool))
            for name in ('initial', 'final'):
                r = total[name]; f = s[name+'_fitting']
                np.testing.assert_allclose(np.array(r['known_MSE'])+r['unknown_bound_MSE'], f['task_MSE'], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(r['sign_cost'], f['observed_sign_regret'], rtol=1e-10, atol=1e-12)
            rows.append(dict(job=job['key'], checkpoint=receipt['checkpoint'], total=total,
                by_source={str(site): decompose(**kw, rowmask=c['sites'] == site) for site in sorted(set(c['sites']))}))
            del d, origin, s, prepared, model, predictions, kw; source.gc.collect(); guard()
        result = source.stable(dict(status='fresh_run_fitting_only', rows=rows, posthoc_diagnostic=True,
            no_new_V_or_H=True, no_model_selection=True, original_rules_unchanged=True,
            all3_observed_label_masks_match_frozen_origins=True,
            sign_weighting='inherited_query_row_weight_per_legal_action_not_action_mean',
            MSE_weighting='inherited_query_row_weight_action_mean', deployment_changed=False))
        out = run.PUBLIC/'fitting_error_attribution.json'
        if args.verify and not out.exists():
            raise ValueError('Original attribution required for replay')
        source.once(out, result)
        if args.verify:
            source.once(run.PUBLIC/'fitting_error_attribution_replay.json',
                dict(all3_decompositions_exact=True, attribution_sha256=source.sha(out)))
        beat(state='fitting_attribution_complete', replay=args.verify)


if __name__ == '__main__':
    main()
