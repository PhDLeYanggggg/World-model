"""Summarize frozen fitting diagnostics without opening any held data."""
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.manage_m3w_easy_gradient_diagnostic import PUBLIC, PRIVATE, registration, immutable, digest


def distribution(values):
    known = np.array([v for v in values if v is not None], dtype=float)
    assert np.isfinite(known).all()
    return dict(defined=len(known), undefined=len(values)-len(known),
        median=float(np.median(known)) if len(known) else None,
        q10=float(np.quantile(known, .1)) if len(known) else None,
        q90=float(np.quantile(known, .9)) if len(known) else None,
        negative=int((known < 0).sum()))


def check_scalar_geometry(g):
    n = g['norms']; c = g['risk_cosine']; p = g['risk_projection']
    assert set(n) == {'marginal', 'occurrence', 'conditional', 'auxiliary', 'supervised'}
    assert all(np.isfinite(v) and v >= 0 for v in n.values())
    for key in c:
        if c[key] is not None:
            assert abs(c[key]) <= 1+1e-12
            np.testing.assert_allclose(p[key], c[key]*n[key]/n['marginal'], rtol=1e-9, atol=1e-9)
    if n['marginal'] > 1e-14:
        np.testing.assert_allclose(g['auxiliary_to_risk_norm'], n['auxiliary']/n['marginal'], rtol=1e-9)
        np.testing.assert_allclose(p['supervised'], 1+p['auxiliary'], rtol=1e-9, atol=1e-9)
        expected = n['marginal']**2+n['auxiliary']**2+2*p['auxiliary']*n['marginal']**2
        np.testing.assert_allclose(n['supervised']**2, expected, rtol=1e-9, atol=1e-12)
    else:
        assert g['auxiliary_to_risk_norm'] is None


def summarize(groups):
    flat = []; signals = []; checked = 0
    for group in groups:
        assert len(group['rows']) == 16 and group['parameter_updates'] == 0 and not group['held_outcomes_used']
        roles = group['identity']['roles']
        assert set(group['fitting_signal']) == set(roles['training_sites'])
        for key in ('producer_sites', 'controller_sites', 'held_sites'):
            assert not set(roles['training_sites']) & set(roles[key])
        for row in group['rows']:
            for g in row['gradients'].values():
                check_scalar_geometry(g); checked += 1
            flat.append(dict(group=group['group'], **row))
        for site, signal in group['fitting_signal'].items():
            np.testing.assert_allclose(signal['easy_signed_risk'],
                signal['easy_harm_mass']-.02*signal['easy_reference_mass'], rtol=1e-8, atol=1e-10)
            signals.append(dict(group=group['group'], site=site, **signal))
    table = {}
    for arm in ('marginal', 'supervised'):
        for point in ('initial', 'final'):
            rows = [r for r in flat if r['arm'] == arm and r['point'] == point]
            result = dict(batches=len(rows), losses={k: distribution([r['losses'][k] for r in rows])
                                                   for k in ('marginal', 'occurrence', 'conditional')}, blocks={})
            for block in ('all', 'shared', 'output'):
                gs = [r['gradients'][block] for r in rows]
                result['blocks'][block] = dict(
                    auxiliary_to_risk_norm=distribution([g['auxiliary_to_risk_norm'] for g in gs]),
                    auxiliary_risk_cosine=distribution([g['risk_cosine']['auxiliary'] for g in gs]),
                    total_risk_cosine=distribution([g['risk_cosine']['supervised'] for g in gs]),
                    total_risk_projection=distribution([g['risk_projection']['supervised'] for g in gs]))
            table[arm+'_'+point] = result
    signal_summary = {k: distribution([r[k] for r in signals]) for k in (
        'easy_probability', 'easy_signed_risk', 'easy_realized_within_budget_fraction',
        'easy_zero_harm_fraction', 'easy_positive_harm_over_reference')}
    by_source = {site: {k: distribution([r[k] for r in signals if r['site'] == site]) for k in signal_summary}
                 for site in sorted({r['site'] for r in signals})}
    return dict(gradients=table, fitting_signal=signal_summary, by_source=by_source,
                scalar_geometry_checks=checked, gradient_batches=len(flat), fitting_source_views=len(signals))


def put(name, text):
    path = PUBLIC/name
    if path.exists():
        assert path.read_text() == text, 'Existing report differs'
    else:
        path.write_text(text)


def main():
    reg = registration(); assert reg == json.loads((PUBLIC/'registration.json').read_text())
    collected = PRIVATE/'collected.json'; evidence = json.loads(collected.read_text())
    first, replay = evidence['receipt.json'], evidence['replay_receipt.json']
    assert first['groups'] == replay['groups'] == 108 and replay['replay_exact']
    assert first['artifacts'] == replay['artifacts']
    assert first['registration_sha256'] == replay['registration_sha256'] == digest(PUBLIC/'registration.json')
    refs = {r['path']: r['sha256'] for r in first['artifacts']}
    for g in evidence['groups']:
        raw = json.dumps(g, indent=2, allow_nan=False)+'\n'
        assert hashlib.sha256(raw.encode()).hexdigest() == refs['groups/'+g['group']+'.json']
    summary = summarize(evidence['groups'])
    summary.update(result_source='fresh_run_fitting_only_gradients_and_exact_replay',
        input_source='cached_verified_fitting_packets_and_checkpoints',
        parent_seal_sha256=reg['parent_seal_sha256'], registered_before_execution=True,
        readout=first, replay=replay, groups=108, heads=216, parameter_updates=0,
        held_outcomes_used=False, independent_roles_read=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False, inferential_CI='not_applicable_dependent_fitting_diagnostic')
    immutable(PUBLIC/'summary.json', summary)
    lines = ['# Fitting-Only Gradient and Risk-Signal Results', '',
        'Result source: fresh_run diagnostic and exact same-runtime replay on cached_verified fitting-only inputs. Registration preceded execution. There were zero optimizer updates and no new held readout. All 108 paired fits are retained; 1,728 gradient batches repeat the same source-role and seed structures, not independent data.', '',
        '## Gradient Geometry', '',
        '| Arm / state | Block | Auxiliary / risk norm median [10th, 90th percentile] | Auxiliary conflict / defined batches | Total conflict / defined batches | Total / risk projection median |',
        '|---|---|---:|---:|---:|---:|']
    for name, row in summary['gradients'].items():
        for block, g in row['blocks'].items():
            r = g['auxiliary_to_risk_norm']; a = g['auxiliary_risk_cosine']; t = g['total_risk_cosine']
            lines.append(f"| {name} | {block} | {r['median']:.6g} [{r['q10']:.6g}, {r['q90']:.6g}] | {a['negative']}/{a['defined']} | {t['negative']}/{t['defined']} | {g['total_risk_projection']['median']:.6g} |")
    lines += ['', 'Conflict means a negative cosine. A negative total/risk projection means the negative total gradient is locally uphill for direct risk under a Euclidean infinitesimal step. It does not prove that the actual finite AdamW step or held policy got worse. Gradients were not applied.', '',
        '## Fitting-Source Signal', '',
        'These are source-query-balanced observed-label summaries. Realized within-budget examples are not causally identifiable safe admissions or an oracle used at inference.', '',
        '| Diagnostic across 216 repeated fitting-source views | Median | 10th percentile | 90th percentile |',
        '|---|---:|---:|---:|']
    for k, v in summary['fitting_signal'].items():
        lines.append(f"| {k} | {v['median']:.6g} | {v['q10']:.6g} | {v['q90']:.6g} |")
    lines += ['', '## Source Breakdown', '',
        '| Fitting locality | Views | Easy prevalence median | Easy harm/reference median | Easy realized within-budget fraction median |',
        '|---|---:|---:|---:|---:|']
    for site, row in summary['by_source'].items():
        lines.append(f"| {site} | {row['easy_probability']['defined']} | {row['easy_probability']['median']:.6g} | {row['easy_positive_harm_over_reference']['median']:.6g} | {row['easy_realized_within_budget_fraction']['median']:.6g} |")
    lines += ['', '## Execution and Limits', '',
        f"CREATE job {first['job_id']}: {first['seconds']:.2f} seconds, peak RSS {first['peak_RSS_KiB']:,} KiB. Replay job {replay['job_id']}: {replay['seconds']:.2f} seconds; all group results exact. Scalar consistency checks: {summary['scalar_geometry_checks']:,}.", '',
        'No primary or risk tolerance changed. Independent roles remain closed; no deployment, scientific efficacy, metric/seconds, human-gold, true3D, foundation or physical-safety claim. Stage5C/SMC disabled.']
    put('results.md', '\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt']='m3w-easy-gradient-v1'
    matplotlib.rcParams['svg.fonttype']='none'
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    names = ['marginal_initial', 'marginal_final', 'supervised_initial', 'supervised_final']
    for i, name in enumerate(names):
        g = summary['gradients'][name]['blocks']['shared']; r = g['auxiliary_to_risk_norm']
        axes[0].plot([r['q10'], r['q90']], [i, i], color='#197978', linewidth=2)
        axes[0].plot(r['median'], i, 'o', color='#197978')
        t = g['total_risk_cosine']; axes[1].barh(i, 100*t['negative']/t['defined'], color='#995366')
    for ax in axes:
        ax.set_yticks(range(4), [n.replace('_', ' ') for n in names]); ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    axes[0].set_xscale('log'); axes[0].set_title('Shared-layer auxiliary/risk norm ratio', fontsize=10)
    axes[0].set_xlabel('Median and 10th-90th percentiles')
    axes[1].set_title('Total gradient opposed to direct risk', fontsize=10); axes[1].set_xlabel('Repeated fitting batches (%)'); axes[1].set_xlim(0, 100)
    fig.suptitle('Frozen fitting-source diagnostic; no parameter updates', fontsize=11)
    buf = io.BytesIO(); fig.savefig(buf, format='svg', metadata={'Date': None, 'Creator': 'M3W deterministic report'})
    path = PUBLIC/'gradient_geometry.svg'
    if path.exists(): assert path.read_bytes() == buf.getvalue()
    else: path.write_bytes(buf.getvalue())
    fig.savefig(PRIVATE/'gradient_geometry.png', dpi=150, metadata={'Software': 'M3W deterministic report'}); plt.close(fig)
    print(json.dumps(dict(groups=108, replay_exact=True, scalar_checks=summary['scalar_geometry_checks'])))


if __name__ == '__main__':
    main()
