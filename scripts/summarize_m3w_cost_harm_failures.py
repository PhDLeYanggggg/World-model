"""Frozen aggregate diagnosis after the registered 72-head cost experiment."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_cost_harm_newton_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = json.loads((PUBLIC/'verification.json').read_text())
    for name in ('summary', 'complete'):
        assert sha(PUBLIC/(name+'.json')) == receipt[name+'_sha256']
    summary = json.loads((PUBLIC/'summary.json').read_text())
    complete = json.loads((PUBLIC/'complete.json').read_text())
    rows = []
    for ref in complete['groups']:
        path = ROOT/ref['path']
        assert sha(path) == ref['sha256']
        rows.append(json.loads(path.read_text()))
    localities, failures = [], []
    for source in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['source'] == source]
        localities.append(dict(source=source, heads=len(group), **{
            name: statistics.mean(r['result']['contrasts'][name] for r in group)
            for name in rows[0]['result']['contrasts']}))
    assert len(rows) == 72 and len(localities) == 12
    for name in rows[0]['result']['contrasts']:
        assert abs(statistics.mean(r[name] for r in localities)-summary[name]['mean']) < 1e-12
    for r in rows:
        p = r['result']['policies']['cost']
        upper = p['easy_selected_risk_upper']
        if upper is None or upper <= .02+1e-12:
            continue
        denominator = p['selected_known_easy_reference_mass']
        known = p['selected_known_easy_harm_mass']/denominator if denominator > 0 else None
        failures.append(dict(group=r['group'], source=r['source'], head_seed=r['head_seed'],
            selected=p['selected_count'], selected_unknown=p['selected_unknown'],
            known_easy_risk=known, easy_upper=upper,
            unknown_envelope_mass=p['selected_unknown_envelope_mass'],
            known_easy_harm_mass=p['selected_known_easy_harm_mass'],
            known_easy_reference_mass=denominator,
            known_label_failure=known is not None and known > .02+1e-12))
    assert len(failures) == summary['cost']['violations']
    assert sum(r['known_label_failure'] for r in failures) == summary['cost']['known_label_violations']
    trains = [r['training'] for r in rows]
    events = [json.loads(line) for line in
        (ROOT/'data/stage_cvpr2027_experiments/european_cost_harm_newton_v1/events.jsonl').read_text().splitlines()]
    first36 = next(e for e in events if e['pid'] == 49738 and e.get('groups') == 36)
    out = dict(result_source='fresh_run_aggregate_diagnosis_only', inputs='cached_verified_72_group_reports',
        verification_sha256=sha(PUBLIC/'verification.json'), rows_are_independent=False,
        independent_confirmation=False, posthoc_failure_slices=True, new_fit=False,
        localities=localities, upper_risk_failures=failures,
        maximum_gradient=max(t['maximum_gradient'] for t in trains),
        maximum_iterations=max(t['max_iterations'] for t in trains),
        maximum_train_mean_preservation_error=max(t['train_relative_mean_error'] for t in trains),
        head_channel_losses_decreased=sum(b < a for t in trains
            for a, b in zip(t['training_loss']['before'], t['training_loss']['after'])),
        completed_head_fit_and_exact_refit_seconds=first36['fit_refit_seconds']+complete['fit_refit_seconds'],
        excludes_pilot_and_unsaved_head_attempt=True,
        resumed_invocation_wall_seconds=complete['seconds'],
        first_invocation_blocked_elapsed='20h21m observed; retain transport_recovery.md, not active compute',
        advance_to_transfer=summary['advance_to_transfer'])
    lines = ['# Frozen Cost-Control Failure Slices', '',
        'Fresh aggregate diagnosis of hash-verified reports; no refitting or selection. These are',
        'post-hoc exposed-development slices, not independent hypothesis tests.', '',
        '| Locality | Heads | Cost-original signed MSE | Full utility % | Matched utility % |',
        '|---|---:|---:|---:|---:|']
    for r in localities:
        lines.append(f"| {r['source']} | {r['heads']} | {r['cost_minus_original_signed_MSE']:+.6f} | "
                     f"{r['cost_minus_original_full_utility_percent']:+.6f} | {r['cost_minus_original_matched_utility_percent']:+.6f} |")
    lines += ['', 'Utility is percent of full known reference error mass, not ADE/FDE improvement.', '',
        '| Source / seed / producer | Selected | Unknown | Known easy risk % | Completion upper % |',
        '|---|---:|---:|---:|---:|']
    for r in failures:
        lines.append(f"| {r['group']} / {r['head_seed']} | {r['selected']} | {r['selected_unknown']} | "
                     f"{100*r['known_easy_risk']:.4f} | {100*r['easy_upper']:.4f} |")
    lines += ['', 'Three upper failures in locality067 require unknown completion; four failures',
        'in112/124 already exist on known labels. Unknown upper bounds are not observed harm.',
        'All144 head-channel training losses decrease; maximum iterations20. The completed',
        'experiment is therefore not explained by the earlier solver nonconvergence.', '',
        'No data role, forecast, policy threshold or deployment is changed.']
    for name, content in [('failure_slices.json', json.dumps(out, indent=2, allow_nan=False)+'\n'),
                          ('failure_slices.md', '\n'.join(lines)+'\n')]:
        path = PUBLIC/name
        if path.exists():
            assert path.read_text() == content
        else:
            path.write_text(content)
    print(json.dumps({k: v for k, v in out.items() if k not in ('localities', 'upper_risk_failures')}, indent=2))


if __name__ == '__main__':
    main()
