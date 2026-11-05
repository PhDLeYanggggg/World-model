"""Attribute frozen fitting errors; no new validation or test readout."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_composed_risk as run
from scripts.diagnose_m3w_decoupled_risk import costs

source, api = run.source, run.api.parent
np, torch = api.np, api.torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = run.PRIVATE
        started, guard, beat = run.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), 'composed_attribution')
        _, values = run.register(beat)
        if source.read(run.PUBLIC/'training_freeze.json')['complete_heads'] != 6:
            raise ValueError('Freeze all six fixed fits first')
        rows = []
        for job in values[3]:
            data, _ = run.previous.previous.inputs(values, job)
            causal = data['causal']; z = api.kin.batch_features(data, 'kinematic')
            y = (api.kin.parent.signed(data['truth_by_arm']['bound_aware']) /
                torch.from_numpy(api.kin.parent.normalizer(data['preprocess']))).numpy()
            for arm in run.api.ARMS:
                receipt = source.read(run.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                state = api.base.read_checkpoint(source.checked(receipt['checkpoint']))
                weights = api.query_weights(causal, state['sampler']['counts'])
                api.base.exact(state['observed_mask_hash'], api.kin.digest(data['observed_future']))
                predictions = {}
                for prefix, initial in (('initial', True), ('final', False)):
                    model = api.model_from(state, initial)
                    risk, decision = (np.empty_like(y) for _ in range(2))
                    for ix, seg in api.batches(causal):
                        guard()
                        with torch.no_grad():
                            r, d = model(z[ix], torch.from_numpy(seg), int(seg.max())+1)
                        risk[ix], decision[ix] = r.numpy(), d.numpy()
                    predictions[prefix+'_risk'] = risk
                    predictions[prefix+'_decision'] = decision
                    predictions[prefix+'_combined'] = api.combined(risk, decision)
                known = causal['allowed'][..., None] & data['observed_future'][:, None, None]
                total = {name: costs(value, y, weights, known) for name, value in predictions.items()}
                for prefix in ('initial', 'final'):
                    for suffix, key in (('risk', 'risk_only_sign_regret'), ('decision', 'decision_only_sign_regret'),
                            ('combined', 'observed_sign_regret')):
                        np.testing.assert_allclose(total[prefix+'_'+suffix]['sign_cost'],
                            receipt[prefix+'_fitting'][key], rtol=1e-10, atol=1e-12)
                rows.append(dict(job=job['key'], arm=arm, total=total, checkpoint=receipt['checkpoint'],
                    by_source={str(site): {name: costs(value, y, weights,
                        known & (causal['sites'] == site)[:, None, None]) for name, value in predictions.items()}
                        for site in sorted(set(causal['sites']))}))
                del predictions, model, state; guard()
            del data, z, y; source.gc.collect()
            beat(state='fitting_attribution_complete', job=job['key'])
        output = run.PUBLIC/'fitting_error_attribution.json'
        if args.verify and not output.exists():
            raise ValueError('Original attribution required')
        source.once(output, source.stable(dict(status='fresh_run_fitting_only', rows=rows,
            all6_observed_masks_verified=True, posthoc_diagnostic=True, no_new_V_or_H=True,
            no_model_selection=True, original_rules_unchanged=True, deployment_changed=False)))
        if args.verify:
            source.once(run.PUBLIC/'fitting_error_attribution_replay.json', dict(
                all6_decompositions_exact=True, attribution_sha256=source.sha(output)))
        source.once(run.PUBLIC/('attribution_replay_execution.json' if args.verify else 'attribution_execution.json'),
            dict(pid=source.os.getpid(), seconds=time.monotonic()-started,
                peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss))
        beat(state='complete', replay=args.verify)


if __name__ == '__main__':
    main()
