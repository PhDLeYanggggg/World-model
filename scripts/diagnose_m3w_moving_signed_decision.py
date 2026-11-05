"""Fitting-only signed-decision diagnosis; no new held-source readout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_source_error_priority as prior
from src.world_model.m3w_frozen_readout_refit import query_weights

source, api = prior.source, prior.api
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_signed_decision_v1'


def audit():
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    _, values = prior.register(lambda **_: None)
    verification = source.read(prior.PUBLIC/'verification.json')
    api.base.exact(verification['analysis_sha256'], source.sha(prior.PUBLIC/'analysis.json'))
    fit_refs = {}
    for ref in source.read(prior.PUBLIC/'training_freeze.json')['students']:
        r = source.read(source.checked(ref)); fit_refs[r['job'], r['arm']] = r
    rows = []
    for job in values[3]:
        d, warm = prior.student_data(values, job)
        c = d['causal']; ix = source.np.arange(len(c['ids'])); choice = c['allowed'].argmax(1)
        weights = query_weights(c, warm['sampler']['counts'])
        norm = api.kin.parent.normalizer(d['preprocess'])
        y = (api.kin.parent.signed(d['truth_by_arm']['bound_aware']).numpy()/norm)[ix, choice]
        known = d['observed_future'] & (weights > 0)
        models = [('warm', warm)]+[(a, api.base.read_checkpoint(source.checked(fit_refs[job['key'], a]['checkpoint']))) for a in api.ARMS]
        for name, state in models:
            score = api.predict(state, c)[ix, choice]/c['scales'][:, None]/norm
            for channel in (0, 1):
                target, pred = y[:, channel], score[:, channel]
                wrong = ((pred <= 0) & (target > 0)) | ((pred > 0) & (target < 0))
                false_safe = known & (pred <= 0) & (target > 0)
                false_reject = known & (pred > 0) & (target < 0)
                total_mse = float((weights*(pred-target)**2).sum())
                positive_mass = float((weights*source.np.maximum(target, 0)*known).sum())
                bins = []
                for lo, hi in ((0., .001), (.001, .01), (.01, .1), (.1, 1.), (1., float('inf'))):
                    at = false_safe & (abs(pred) >= lo) & (abs(pred) < hi)
                    bins.append(dict(abs_normalized_score_lower=lo,
                        upper=hi if source.np.isfinite(hi) else None, channels=int(at.sum()),
                        positive_excess_mass=float((weights*source.np.maximum(target, 0)*at).sum()),
                        mse=float((weights*(pred-target)**2*at).sum())))
                rows.append(dict(job=job['key'], model=name, channel=('all', 'easy')[channel],
                    observed_channels=int(known.sum()), false_safe_channels=int(false_safe.sum()),
                    false_reject_channels=int(false_reject.sum()), full_channel_MSE=total_mse,
                    observed_positive_excess_mass=positive_mass,
                    false_safe_excess_mass=float((weights*target*false_safe).sum()),
                    false_reject_margin_mass=float((weights*-target*false_reject).sum()),
                    observed_cost_weighted_sign_regret=float((weights*abs(target)*wrong*known).sum()),
                    false_safe_score_bins=bins))
        del d, warm, models; source.gc.collect()
    out = dict(status='fresh_run', source_result=source.artifact(prior.PUBLIC/'analysis.json'),
        rows=rows, weights='complete_fitting_empirical_query_draw_weights',
        normalized_units='own_fitting_RMS_and_causal_past_scale',
        fitting_only=True, new_validation_or_test_readout=False,
        repeated_channels_not_independent_samples=True, causal_feature_schema_unchanged=True)
    source.once(PUBLIC/'fitting_boundary_audit.json', source.stable(out))
    lines = ['# Fitting-Only Decision Boundary Audit', '',
        'Fresh recomputation from verified checkpoints. Only B fitting rows are',
        'inspected; no new V or reserved test readout. Full empirical query-draw',
        'weights retain empty proposal queries. Quantities below are normalized',
        'signed-budget costs, not native ADE or calibrated safety probabilities.', '',
        '| Context | Model | Channel | False-safe outputs | False-reject outputs | Weighted sign regret |',
        '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['job']} | {r['model']} | {r['channel']} | {r['false_safe_channels']} | {r['false_reject_channels']} | {r['observed_cost_weighted_sign_regret']:.8f} |")
    lines += ['', 'A false-safe sign has predicted score <=0 and positive realized budget',
        'excess. A false rejection has score >0 and negative realized excess.',
        'Both are retrospective supervision diagnostics, not observable labels',
        'at inference or rowwise safety guarantees. Counts overlap across fitting',
        'contexts. The original missing-reference/support issue is unchanged.', '',
        'The accompanying JSON reports fixed score bins, error mass and MSE.',
        'This audit can motivate an objective comparison; it cannot show that',
        'changing the objective will generalize or repair missing support.', '']
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC/'fitting_boundary_audit.md').write_text('\n'.join(lines))
    for r in rows:
        print(r['job'], r['model'], r['channel'], r['false_safe_channels'], r['observed_cost_weighted_sign_regret'])


if __name__ == '__main__':
    audit()
