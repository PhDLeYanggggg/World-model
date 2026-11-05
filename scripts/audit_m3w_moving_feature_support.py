"""Frozen B-to-V causal feature support audit; no future outcome readout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_signed_decision as prior

source, api = prior.source, prior.api
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_feature_support_v1'
BLOCKS = ((0, 24, 'ego_history'), (24, 32, 'ego_mask'),
    (32, 224, 'neighbor_history'), (224, 288, 'neighbor_mask'),
    (288, 306, 'rollout_summary'), (306, 330, 'floor_rollout'),
    (330, 354, 'candidate_rollout'), (354, 355, 'log_causal_scale'),
    (355, 379, 'CV_rollout'), (379, 380, 'floor_switch'), (380, 444, 'kinematics'))
NAMES = [f'common_{i}' for i in range(380)]+list(api.parent.NAMES)


def proposal_features(c, state, choices):
    if set(c) != api.base.CAUSAL_KEYS:
        raise ValueError('Only original causal schema, no outcome fields')
    choices = source.np.asarray(choices)
    if choices.shape != (len(c['ids']),) or not source.np.isin(choices, source.np.arange(-1, 7)).all():
        raise ValueError('Frozen aligned causal action choices required')
    choices = source.np.maximum(choices, 0)
    x = c['x']; raw = api.parent.motion_basis(x)
    i = source.np.arange(len(x)); p, e = state['preprocess'], state['extra_preprocess']
    z = source.np.concatenate(((x[i, choices, :380]-p['mean'][:380])/p['std'][:380],
        (raw[i, choices]-e['mean'])/e['std']), -1)
    if not source.np.isfinite(z).all():
        raise ValueError('Finite pre-clipping own-fitting normalized features required')
    return z


def reference(z):
    if z.ndim != 2 or z.shape[1] != 444 or not len(z):
        raise ValueError('Nonempty fixed fitting proposal features required')
    return dict(min=z.min(0), max=z.max(0), mean=z.mean(0),
        std=z.std(0).clip(1e-6), q01=source.np.quantile(z, .01, axis=0),
        q99=source.np.quantile(z, .99, axis=0))


def describe(z, ref):
    clip = abs(z) > 8
    out = (z < ref['min']-1e-6) | (z > ref['max']+1e-6)
    tail = (z < ref['q01']-1e-6) | (z > ref['q99']+1e-6)
    blocks = []
    for lo, hi, name in BLOCKS:
        a = z[:, lo:hi]
        blocks.append(dict(block=name, clipped_values_fraction=float(clip[:, lo:hi].mean()),
            rows_any_clipped_fraction=float(clip[:, lo:hi].any(1).mean()),
            outside_fitting_range_values_fraction=float(out[:, lo:hi].mean()),
            outside_fitting_01_99_fraction=float(tail[:, lo:hi].mean()),
            RMS_mean_shift_fitting_std=float(source.np.sqrt(source.np.mean(
                ((a.mean(0)-ref['mean'][lo:hi])/ref['std'][lo:hi])**2)))))
    per_feature = clip.mean(0)
    order = source.np.argsort(-per_feature, kind='stable')[:12]
    return dict(rows=len(z), blocks=blocks, rows_any_clipped_fraction=float(clip.any(1).mean()),
        rows_any_outside_fitting_range_fraction=float(out.any(1).mean()),
        top_clipped_features=[dict(name=NAMES[i], clipped_fraction=float(per_feature[i]),
            fitting_min=float(ref['min'][i]), fitting_max=float(ref['max'][i]),
            min=float(z[:, i].min()), max=float(z[:, i].max())) for i in order if per_feature[i] > 0])


def main():
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    _, values = prior.register(lambda **_: None)
    api.base.exact(source.read(prior.PUBLIC/'verification.json')['analysis_sha256'], source.sha(prior.PUBLIC/'analysis.json'))
    rows = []
    for job in values[3]:
        fitting = prior.old.packet(values[2], job, values[4], values[5][job['key']][0], training=False)
        warm = api.base.read_checkpoint(source.checked(values[11][job['key'], 'kinematic']['checkpoint']))
        api.base.exact(warm['fit_ids'], fitting['ids'])
        proposer = api.base.read_checkpoint(source.checked(values[7][job['key'], 1]['checkpoint']))
        router_api = api.parent.parent.parent
        router_api.assert_excluded(proposer, fitting['causal'], fitting['sites'])
        fit_choice, _ = router_api.router.choose(router_api.router.predict(proposer, fitting['causal']),
            fitting['causal']['allowed'])
        fit_active = fit_choice >= 0
        mask = source.np.zeros_like(fitting['causal']['allowed'])
        mask[source.np.flatnonzero(fit_active), fit_choice[fit_active]] = True
        api.base.exact(api.parent.digest(mask), warm['proposal_hash'])
        fit_z = proposal_features(fitting['causal'], warm, fit_choice)
        ref = reference(fit_z[fit_active])
        for site in sorted(set(fitting['sites'])):
            at = (fitting['sites'] == site) & fit_active
            rows.append(dict(job=job['key'], role='B_fitting', site=str(site),
                **describe(fit_z[at], ref), causal_scales_quantiles=source.np.quantile(
                    fitting['causal']['scales'][at], [0, .5, .95, 1]).tolist()))
        part = values[5][job['key']][2]
        packet = prior.old.packet(values[2], job, values[4], part, training=False)
        prior.prior.api.assert_excluded(warm, packet['causal'], part['names'])
        saved = source.read(prior.PUBLIC/'groups'/(job['key']+'_val2_propose1.json'))
        with source.np.load(source.checked(saved['cache']), allow_pickle=False) as cache:
            source.np.testing.assert_array_equal(packet['ids'], cache['ids'])
            choices = cache['public_router'].copy()
        api.base.exact(source.array_hash(choices), saved['choice_hashes']['public_router'])
        z = proposal_features(packet['causal'], warm, choices)
        for site in sorted(set(packet['sites'])):
            at = (packet['sites'] == site) & (choices >= 0)
            rows.append(dict(job=job['key'], role='V_exposed_development', site=str(site),
                full_roster_rows=int((packet['sites'] == site).sum()),
                **describe(z[at], ref), causal_scales_quantiles=source.np.quantile(
                    packet['causal']['scales'][at], [0, .5, .95, 1]).tolist()))
        del fitting, warm, proposer, fit_z, z, packet; source.gc.collect()
    source.once(PUBLIC/'causal_feature_support.json', source.stable(dict(status='fresh_run', rows=rows,
        parent_analysis=source.artifact(prior.PUBLIC/'analysis.json'), future_outcomes_used=False,
        population='frozen_foreign_proposer_active_rows_only_descriptive_not_policy_reweighting',
        mean_std_fitted_on_B_only=True, new_H=False, evaluation_rules_changed=False)))
    lines = ['# Frozen Causal Feature Support', '',
        'Fresh B-to-V descriptive input audit, using fixed foreign proposals and',
        'own-B normalization. No future outcomes are consumed by the audit.',
        'Only proposed rows enter these feature summaries; the complete original',
        'policy evaluation and all support requirements remain unchanged.', '',
        '| Context | Role | Source | Proposed rows | Any clipped % | Any outside B range % |',
        '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['job']} | {r['role']} | {r['site']} | {r['rows']} | {100*r['rows_any_clipped_fraction']:.3f} | {100*r['rows_any_outside_fitting_range_fraction']:.3f} |")
    lines += ['', 'Per-block and named clipped-feature details are in JSON. Feature-range',
        'extrapolation is not by itself proof of predictive failure or a safe',
        'rejection rule. These are previously exposed development sources, not',
        'independent test. No threshold selection, new predictions or deployment.', '']
    PUBLIC.mkdir(parents=True, exist_ok=True); (PUBLIC/'README.md').write_text('\n'.join(lines))
    for r in rows:
        print(r['role'], r['site'], r['rows'], r['rows_any_clipped_fraction'], r['top_clipped_features'][:2])


if __name__ == '__main__':
    main()
