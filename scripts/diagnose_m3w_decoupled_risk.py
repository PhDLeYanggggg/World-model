"""Frozen own-fitting error attribution; never a new tuning or held-data pass."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_decoupled_risk as run

api, source = run.api, run.source
np, torch = api.np, api.torch


def costs(prediction, target, weights, mask):
    cost = weights[:, None, None]*np.abs(target)*mask
    false_safe = (prediction <= 0) & (target > 0)
    false_reject = (prediction > 0) & (target < 0)
    return dict(false_safe_cost=(cost*false_safe).sum((0, 1)).tolist(),
        false_reject_cost=(cost*false_reject).sum((0, 1)).tolist(),
        sign_cost=(cost*(false_safe | false_reject)).sum((0, 1)).tolist(),
        false_safe_count=(mask*false_safe).sum((0, 1)).tolist(),
        false_reject_count=(mask*false_reject).sum((0, 1)).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--verify', action='store_true'); args = p.parse_args()
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = run.PRIVATE
        started, guard, beat = run.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), 'dual_attribution')
        _, values = run.register(beat)
        if source.read(run.PUBLIC/'training_freeze.json')['complete_heads'] != 9:
            raise ValueError('All fixed fits must be frozen first')
        rows = []
        for job in values[3]:
            d, _ = run.previous.inputs(values, job); c = d['causal']; z = api.kin.batch_features(d, 'kinematic')
            y = (api.kin.parent.signed(d['truth_by_arm']['bound_aware'])/
                torch.from_numpy(api.kin.parent.normalizer(d['preprocess']))).numpy()
            for arm in api.ARMS:
                r = source.read(run.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                s = api.base.read_checkpoint(source.checked(r['checkpoint'])); weights = api.query_weights(c, s['sampler']['counts'])
                api.base.exact(s['observed_mask_hash'], api.kin.digest(d['observed_future']))
                pred = {}
                for prefix, initial in (('initial', True), ('final', False)):
                    m = api.model_from(s, initial); risk, decision = (np.empty_like(y) for _ in range(2))
                    for ix, seg in api.batches(c):
                        guard()
                        with torch.no_grad():
                            rv, dv = m(z[ix], torch.from_numpy(seg), int(seg.max())+1)
                        risk[ix], decision[ix] = rv.numpy(), dv.numpy()
                    pred[prefix+'_risk'] = risk; pred[prefix+'_decision'] = decision
                    pred[prefix+'_combined'] = api.combined(risk, decision)
                known = c['allowed'][..., None] & d['observed_future'][:, None, None]
                total = {name: costs(v, y, weights, known) for name, v in pred.items()}
                for prefix in ('initial', 'final'):
                    f = r[prefix+'_fitting']
                    for suffix, key in (('risk', 'risk_only_sign_regret'), ('decision', 'decision_only_sign_regret'),
                            ('combined', 'observed_sign_regret')):
                        np.testing.assert_allclose(total[prefix+'_'+suffix]['sign_cost'], f[key], rtol=1e-10, atol=1e-12)
                rows.append(dict(job=job['key'], arm=arm, total=total, checkpoint=r['checkpoint'],
                    by_source={str(site): {name: costs(v, y, weights, known & (c['sites'] == site)[:, None, None])
                        for name, v in pred.items()} for site in sorted(set(c['sites']))}))
                del pred, m, s; guard()
            del d, z, y; source.gc.collect(); beat(state='attribution_context_complete', job=job['key'])
        out = run.PUBLIC/'fitting_error_attribution.json'
        if args.verify and not out.exists():
            raise ValueError('Original attribution required')
        source.once(out, source.stable(dict(status='fresh_run_fitting_only', rows=rows,
            all9_observed_masks_verified=True, posthoc_diagnostic=True, no_new_V_or_H=True,
            no_model_selection=True, original_rules_unchanged=True, deployment_changed=False)))
        if args.verify:
            source.once(run.PUBLIC/'fitting_error_attribution_replay.json',
                dict(all9_decompositions_exact=True, attribution_sha256=source.sha(out)))
        source.once(run.PUBLIC/('attribution_replay_execution.json' if args.verify else 'attribution_execution.json'),
            dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
                peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        beat(state='attribution_complete', replay=args.verify)


if __name__ == '__main__':
    main()
