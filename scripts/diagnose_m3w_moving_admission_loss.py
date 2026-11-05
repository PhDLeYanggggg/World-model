"""Fitting-only loss/sign decomposition, not a change to the risk contract."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_kinematic_risk as parent

source, api = parent.source, parent.api
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_admission_risk_v1'


def decompose(prediction, truth, weights, observed):
    np = source.np
    if prediction.shape != truth.shape or prediction.shape != (len(weights), 2):
        raise ValueError('Two normalized budget-excess outputs required')
    if not all(np.isfinite(v).all() for v in (prediction, truth, weights)) or (weights < 0).any():
        raise ValueError('Finite targets, predictions and nonnegative query weights required')
    supported = weights > 0
    both_admitted = (prediction <= 0).all(1)
    rows = []
    for channel, name in enumerate(('all', 'easy')):
        p, y = prediction[:, channel], truth[:, channel]
        squared = (p-y)**2/2
        gradient = np.abs(p-y)
        total = float((weights*squared).sum())
        total_gradient = float((weights*gradient).sum())
        for known, label in ((True, 'observed'), (False, 'bound_only')):
            for positive, sign in ((True, 'positive_budget_excess'), (False, 'nonpositive_budget_excess')):
                for admitted, gate in ((True, 'predicted_nonpositive'), (False, 'predicted_positive')):
                    at = supported & (observed == known) & ((y > 0) == positive) & ((p <= 0) == admitted)
                    w = weights*at
                    rows.append(dict(channel=name, supervision=label, target=sign, prediction=gate,
                        rows=int(at.sum()), query_weight=float(w.sum()),
                        weighted_MSE=float((w*squared).sum()),
                        channel_MSE_share=float((w*squared).sum()/total) if total else None,
                        weighted_absolute_output_gradient=float((w*gradient).sum()),
                        channel_gradient_share=float((w*gradient).sum()/total_gradient) if total_gradient else None,
                        actual_positive_excess=float((w*np.maximum(y, 0)).sum()),
                        predicted_excess=float((w*p).sum()), actual_excess=float((w*y).sum()),
                        both_risk_outputs_admit_rows=int((at & both_admitted).sum()),
                        both_risk_outputs_admit_weight=float((w*both_admitted).sum())))
    return rows


def main():
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    _, _, data, jobs, refs, parts, points, routers, _, fitted, _ = parent.register(lambda **_: None)
    frozen = source.read(parent.PUBLIC/'training_freeze.json')
    trained = {}
    for ref in frozen['fits']:
        r = source.read(source.checked(ref)); trained[r['job'], r['arm']] = r
    rows = []
    for job in jobs:
        d, warm = parent.fitting(data, job, refs, parts, points, routers, fitted)
        c = d['causal']; chosen = c['allowed'].argmax(1); ix = source.np.arange(len(chosen))
        weights = parent.parent.api.query_weights(c, warm['sampler']['counts'])
        scale = c['scales'][:, None]*api.parent.normalizer(d['preprocess'])
        truth = (api.parent.signed(d['truth_by_arm']['bound_aware']).numpy()/api.parent.normalizer(d['preprocess']))[ix, chosen]
        for arm in ('masked', 'kinematic'):
            ref = trained[job['key'], arm]['checkpoint']; state = api.base.read_checkpoint(source.checked(ref))
            api.base.exact(state['target_hash'], api.digest(d['truth_by_arm']['bound_aware'].numpy()))
            api.base.exact(state['fit_ids'], c['ids'])
            prediction = api.predict(state, c)[ix, chosen]/scale
            cells = decompose(prediction, truth, weights, d['observed_future'])
            rows.append(dict(job=job['key'], arm=arm, checkpoint=ref,
                fitting_sources=sorted(set(c['sites'])), observed_rows=int((d['observed_future'] & (weights > 0)).sum()),
                bound_rows=int((~d['observed_future'] & (weights > 0)).sum()), cells=cells))
        del d, warm; source.gc.collect()
    result = dict(status='fresh_run', scope='fitting_only_original_query_weights',
        normalization='original_fitting_signed_risk_RMS_and_causal_scale',
        risk_gate_only_not_full_support_gate=True, held_outcomes_used=False,
        labels_are_training_only=True, rows=rows,
        evidence=[source.artifact(parent.PUBLIC/'analysis.json'), source.artifact(parent.PUBLIC/'training_freeze.json')],
        diagnostic_source=source.artifact(Path(__file__).resolve()), original_rules_unchanged=True,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    source.once(PUBLIC/'fitting_loss_decomposition.json', source.stable(result))
    for r in rows:
        print(r['job'], r['arm'], [(c['channel'], round(c['channel_MSE_share']*100, 3),
            round(c['channel_gradient_share']*100, 3), c['rows']) for c in r['cells']
            if c['supervision'] == 'observed' and c['target'] == 'positive_budget_excess'
            and c['prediction'] == 'predicted_nonpositive'], flush=True)


if __name__ == '__main__':
    main()
