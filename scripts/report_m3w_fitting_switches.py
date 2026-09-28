"""Independent accounting checks and descriptive report; no model selection."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import diagnose_m3w_fitting_switches as run


def verify_node(node):
    checks = 0
    assert node['rows'] == node['known_rows']+node['unknown_rows']; checks += 1
    for arm, v in node['arms'].items():
        parts = list(v['partition'].values()); r = v['row_screen']; ranks = v['ranking']
        assert sum(p['rows'] for p in parts) == node['rows']; checks += 1
        assert sum(p['known_rows'] for p in parts) == node['known_rows']; checks += 1
        for key in ('benefit', 'harm'):
            np.testing.assert_allclose(sum(p[key] for p in parts), node['total'][key], atol=1e-12); checks += 1
            np.testing.assert_allclose(v['partition']['admitted'][key], r[key], atol=1e-12); checks += 1
        np.testing.assert_allclose(sum(p['mass'] for p in parts), 1., atol=1e-12); checks += 1
        np.testing.assert_allclose(r['benefit']-r['harm'], r['net_gain'], atol=1e-12); checks += 1
        np.testing.assert_allclose(node['total']['benefit']-node['total']['harm'], node['total']['net_gain'], atol=1e-12); checks += 1
        assert r['selected_rows'] == v['partition']['admitted']['rows']; checks += 1
        for q in v['queries'].values():
            assert q['queries'] == node['query_count'] == q['defined']+q['undefined']+q['unknown_selected']; checks += 1
            assert 0 <= q['violating'] <= q['defined']; checks += 1
            assert 0 <= q['predicted_safe_realized_positive'] <= q['defined']+q['undefined']; checks += 1
        assert ranks['queries'] == node['query_count'] == sum(ranks[k] for k in (
            'queries_incomplete', 'queries_zero_count', 'queries_no_choice', 'queries_informative')); checks += 1
        assert ranks['oracle_minus_utility_sum'] >= -1e-8 and ranks['oracle_minus_screen_sum'] >= -1e-8; checks += 1
        np.testing.assert_allclose(ranks['oracle_minus_utility_sum']+ranks['utility_minus_screen_sum'],
                                   ranks['oracle_minus_screen_sum'], rtol=1e-10, atol=1e-8); checks += 1
    return checks


def aggregate(nodes):
    def mean(path):
        values = []
        for node in nodes:
            v = node
            for key in path: v = v[key]
            if v is not None: values.append(v)
        return dict(mean=float(np.mean(values)) if values else None, defined=len(values), total=len(nodes))
    out = dict(groups=len(nodes), rows=sum(n['rows'] for n in nodes),
        known_rows=sum(n['known_rows'] for n in nodes), unknown_rows=sum(n['unknown_rows'] for n in nodes),
        query_occurrences=sum(n['query_count'] for n in nodes),
        total={k: mean(('total', k)) for k in nodes[0]['total']},
        utility={k: mean(('utility', k)) for k in nodes[0]['utility']}, arms={})
    for arm in nodes[0]['arms']:
        x = nodes[0]['arms'][arm]
        out['arms'][arm] = dict(
            row_screen={k: mean(('arms', arm, 'row_screen', k)) for k in x['row_screen'] if k != 'risk_MSE'},
            partition={k: {m: mean(('arms', arm, 'partition', k, m)) for m in ('mass', 'benefit', 'harm')} for k in x['partition']},
            query_statuses={k: {m: sum(n['arms'][arm]['queries'][k][m] for n in nodes)
                for m in x['queries'][k]} for k in ('all', 'easy')},
            ranking={m: sum(n['arms'][arm]['ranking'][m] for n in nodes) for m in x['ranking']})
    return out


def main():
    started = time.monotonic(); cfg, reg = run.registration()
    assert reg == json.loads((run.PUBLIC/'registration.json').read_text())
    done = json.loads((run.PUBLIC/'completion.json').read_text()); replay = json.loads((run.PUBLIC/'replay.json').read_text())
    assert done['groups'] == replay['groups'] and replay['exact'] and len(done['groups']) == 108
    records = []; checks = 0; seen = set(); unique_ids = set(); role_rows = 0
    for ref in done['groups']:
        doc = run.checked(ref); assert doc['group'] not in seen; seen.add(doc['group'])
        assert doc['registration_sha256'] == run.digest(run.PUBLIC/'registration.json')
        assert not doc['held_outcomes_used'] and not doc['independent_roles_read'] and not doc['policy_changed']
        roles = doc['source_identity']['roles']; actual = set(doc['result']['by_site'])
        assert actual == set(roles['training_sites']) and not actual & set(roles['held_sites'])
        for p in doc['upstream_refs']: assert run.base.artifact(ROOT/p['path']) == p
        label_ref = doc['upstream_refs'][2]
        with np.load(ROOT/label_ref['path'], allow_pickle=False) as z:
            assert run.base.inter.array_hash(z['ids']) == doc['source_identity']['fitting_ids_hash']
            assert len(z['ids']) == doc['result']['rows'] and int(z['known'].sum()) == doc['result']['known_rows']
            np.testing.assert_allclose(z['positive_benefit']-z['positive_harm'], z['signed_gain'], equal_nan=True)
            unique_ids.update(z['ids'].tolist()); role_rows += len(z['ids'])
        checks += verify_node(doc['result'])
        for node in doc['result']['by_site'].values(): checks += verify_node(node)
        for key in doc['result']['total']:
            np.testing.assert_allclose(np.mean([n['total'][key] for n in doc['result']['by_site'].values()]),
                                       doc['result']['total'][key], atol=1e-12); checks += 1
        records.append(doc)
    nodes = [d['result'] for d in records]; sites = sorted({s for n in nodes for s in n['by_site']})
    summary = dict(result_source='fresh_run_fitting_only_diagnostic_with_exact_replay',
        input_source='cached_verified', overall=aggregate(nodes),
        by_site={s: aggregate([n['by_site'][s] for n in nodes if s in n['by_site']]) for s in sites},
        unique_source_row_ids=len(unique_ids), unique_rows_are_not_independent=True,
        repeated_fitting_rows=role_rows, new_training='not_run_diagnostic_only',
        held_evaluation='not_run', independent_confirmation='not_run_closed',
        no_deployment_change=True, formal_primary_replaced=False)
    run.immutable(run.PUBLIC/'summary.json', summary)
    a = summary['overall']['arms']; total = summary['overall']['total']['benefit']['mean']
    table = ['| Frozen arm | Retained benefit | Positive harm | Net gain | All risk violations/defined | Easy violations/defined |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, z in a.items():
        r = z['row_screen']; qa, qe = z['query_statuses']['all'], z['query_statuses']['easy']
        table.append(f"| {arm} | {r['benefit']['mean']:.8f} | {r['harm']['mean']:.8f} | {r['net_gain']['mean']:.8f} | {qa['violating']}/{qa['defined']} | {qe['violating']}/{qe['defined']} |")
    partition = ['| Exclusive rejection reason | Raw missed benefit share | Trainable | Fixed |', '|---|---:|---:|---:|']
    for reason in a['raw']['partition']:
        values = [100*a[arm]['partition'][reason]['benefit']['mean']/total for arm in a]
        partition.append(f"| {reason} | "+' | '.join(f'{x:.3f}%' for x in values)+' |')
    rejected = ['| Raw-screen partition | Benefit | Harm | Net |', '|---|---:|---:|---:|']
    for reason, v in a['raw']['partition'].items():
        b, h = v['benefit']['mean'], v['harm']['mean']
        rejected.append(f'| {reason} | {b:.8f} | {h:.8f} | {b-h:.8f} |')
    ranks = ['| Arm | Informative query occurrences | Utility ranking misses oracle | Utility-only worsens screen | Incomplete | Zero count | No choice |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm, z in a.items():
        r = z['ranking']; ranks.append(f"| {arm} | {r['queries_informative']} | {r['oracle_better_than_utility_queries']} | {r['utility_worse_than_screen_queries']} | {r['queries_incomplete']} | {r['queries_zero_count']} | {r['queries_no_choice']} |")
    text = ('# Fitting-Only Utility and Risk Diagnosis\n\n'
        f"fresh_run: all108 groups, {role_rows:,} repeated fitting rows, {len(unique_ids):,} unique row IDs. "
        'Unique rows still overlap in recordings and are not independent units. Inputs and checkpoints cached_verified. '
        f"Full pass {done['seconds']:.2f}s; exact full replay {replay['seconds']:.2f}s.\n\n"
        'No new training, joint optimization, threshold selection or held/independent readout. '
        'The frozen independent sign screen is a diagnostic, not a replacement deployment policy.\n\n'
        '## Source/Query-Balanced Fitting Costs\n\n'+ '\n'.join(table)+'\n\n'
        'Costs are normalized by each head\'s original fitting cost scale; entries are equal-context means. '
        'They are not ADE percentage improvements. Query counts are dependent occurrences, not independent trials.\n\n'
        '## Missed Benefit Accounting\n\n'+'\n'.join(partition)+'\n\n'
        'Shares use all available fitting benefit as denominator and sum to100%, including admitted benefit. '
        'The exclusive priority order is registered; overlapping causes must not be interpreted causally.\n\n'
        +'\n'.join(rejected)+'\n\n'
        'Missed benefit is not recoverable gain: an excluded population can contain some beneficial '
        'switches and still have negative net gain or unacceptable positive harm. Removing a gate '
        'requires its own evidence and is not licensed by the oracle.\n\n'
        '## Within-Query Utility Ordering\n\n'+'\n'.join(ranks)+'\n\n'
        'At the sign screen\'s own count, ranking the same eligible pool by realized signed gain gives an '
        'unconstrained oracle upper bound. It ignores risk constraints and is never an inference input. '
        'Only complete-label queries contribute. Utility-only ordering is not assumed safe.\n\n'
        '## Boundaries\n\n'
        'These scores were fitted on the diagnosed sources: this is capacity/objective debugging, not '
        'validation, calibration or evidence of generalization. Per-locality entries are in summary.json. '
        'Unknown-selected and zero-denominator query counts remain explicit. No formal risk certificate, '
        'new world-dynamics lift or deployment upgrade. Image-local detector silver, obs8/pred12, stride12 '
        'raw frames; no metric/seconds/true-3D/foundation claim. Stage5C and SMC remain off.\n')
    (run.PUBLIC/'results.md').write_text(text)
    analysis = ('# Failure Analysis and Next Research Boundary\n\n'
        'The readout locates observed opportunity loss, not a unique causal explanation. The fitted '
        'ridge utility, raw all-risk head and updated easy-risk heads still leave beneficial examples '
        'unselected. But realized benefit also exists in populations with larger positive harm. '
        'This is why missed oracle benefit alone cannot justify loosening thresholds.\n\n'
        '## Measured Fitting Bottleneck\n\n'
        f"The unchanged all-risk screen excludes {100*a['raw']['partition']['all_risk_rejected']['benefit']['mean']/total:.3f}% "
        'of available positive benefit under the registered exclusive accounting. '
        f"Nonpositive utility excludes {100*a['raw']['partition']['nonpositive_utility']['benefit']['mean']/total:.3f}%. "
        'The latter partition has normalized net gain '
        f"{a['raw']['partition']['nonpositive_utility']['benefit']['mean']-a['raw']['partition']['nonpositive_utility']['harm']['mean']:.8f}; "
        'its oracle-positive members are mixed with larger harmful outcomes. The all-risk-rejected '
        'partition has positive net gain but also nonzero positive harm, so it is not a safe pool '
        'to admit indiscriminately. No measured result here licenses raising the2% budget.\n\n'
        f"Raw utility ordering falls below the realized same-count oracle in {a['raw']['ranking']['oracle_better_than_utility_queries']:,} "
        f"of {a['raw']['ranking']['queries_informative']:,} informative fitting query occurrences. "
        f"Removing the risk preference and using utility alone worsens the sign screen in {a['raw']['ranking']['utility_worse_than_screen_queries']:,} "
        'of those occurrences. Neither oracle headroom nor average ordering gain implies '
        'safe query-level intervention.\n\n'
        '## What This Separates\n\n'
        '- Causal eligibility and utility-sign rejection occur before either risk screen.\n'
        '- The all-risk screen is identical in all three arms. Differences after it reflect the '
        'easy-risk branch and different admitted counts, not a matched-count deployment comparison.\n'
        '- Utility misordering is visible only against a realized, risk-unconstrained oracle. Its '
        'regret does not prove that the missing information is learnable from causal features.\n'
        '- Observed query violations on fitting rows indicate unresolved risk prediction/selection '
        'errors, not just held-scene distribution shift. Individual stochastic outcomes can violate '
        'an expected-risk prediction; this is not proof that every such prediction is miscalibrated.\n'
        '- Undefined and unknown-selected queries stay in the accounting. Neither is a safety pass.\n\n'
        '## What Not to Repeat\n\n'
        'Do not repeat the failed occurrence-freezing or gradient-cap search. Utility-only ordering '
        'already showed an accuracy/safety tradeoff in the earlier query-utility experiment. '
        'Do not relabel that old control as a new remedy or treat this diagnostic as a model gain.\n\n'
        '## Next Test\n\n'
        'Use the fitting-only partition net costs and source-wise ranking errors to decide whether '
        'the next controlled change belongs to utility representation or conditional risk learning. '
        'Before another fit, compare against earlier gain/harm and descriptor-capacity experiments '
        'and register one genuinely different factor. Keep forecasts, source roles, risk estimand '
        'and budgets fixed. The dominant all-risk partition makes another easy-occurrence-only '
        'repair poorly targeted. A useful next check should distinguish learnable causal separation from '
        'irreducible future ambiguity. A fitting-locality-held internal control can test whether '
        'gain/all-risk separation survives transfer between the two fitting sources, without '
        'opening the outer held roles. Any preprocessing and new head must be fit only on its '
        'internal training source; inherited two-source learned scores cannot be called an '
        'independent internal test. This remains development, not independent confirmation. '
        'A threshold sweep on observed test outcomes cannot answer that question.\n\n'
        'No new predictor was trained, no independent outcome was read, and no deployment changed. '
        'These are image-local detector-silver raw-frame development findings, not metric/seconds, '
        'true-3D, foundation or independently calibrated safety evidence. Stage5C/SMC stay off.\n')
    (run.PUBLIC/'failure_analysis.md').write_text(analysis)
    operation = ('# Run and Recovery Guide\n\n'
        'All commands use the native arm64 environment from the repository root.\n\n'
        '```bash\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_fitting_switches.py register\n'
        '# Commit the registration before reading fitting outcomes.\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_fitting_switches.py pilot\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_fitting_switches.py run --resume\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_fitting_switches.py replay\n'
        'PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_fitting_switches.py\n'
        '```\n\n'
        'The completed receipts are immutable: do not overwrite them to manufacture a new run. '
        'The resume option verifies completed group inputs and continues missing groups after an '
        'interruption; a completed run is not restarted. A replay reconstructs every group and '
        'compares the results exactly without rewriting the group receipts.\n\n'
        'Heartbeat and group receipts are under data/stage_cvpr2027_experiments/'
        'european_fitting_switch_diagnostic_v1/. Check the recorded PID and completed group count; '
        'slow progress is not a hang. CPU4, interop1 and workers0 are fixed. The runner stops before '
        'crossing the10GiB reserve plus32MiB temporary margin and preserves completed work.\n\n'
        'Only code, configuration, aggregate reports and light receipts belong in Git. Do not '
        'commit private fitting data, checkpoints or unrelated staged files. Full legacy tests '
        'and independent confirmation are not run by this diagnostic.\n')
    (run.PUBLIC/'operation.md').write_text(operation)
    tests = ['tests/test_m3w_fitting_switch_diagnostic.py', 'tests/test_m3w_fitting_switch_report.py', 'tests/test_m3w_fitting_gain_labels.py',
             'tests/test_m3w_easy_component_diagnostic.py']
    proc = subprocess.run([sys.executable, '-m', 'pytest', *tests, '-q'], cwd=ROOT, text=True, capture_output=True)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    (run.PUBLIC/'scoped_tests.txt').write_text(proc.stdout+proc.stderr)
    verification = dict(status='verified_fitting_diagnostic_not_generalization',
        groups=108, source_roles_checked=108, arithmetic_checks=checks,
        label_alignment_rows=role_rows, full_exact_replay=True,
        same_code_replay_is_not_independent_retraining=True, independent_accounting_checks=True,
        full_legacy_suite='not_run_scoped_tests_only',
        source_bindings={**reg['bindings'], str(Path(__file__).relative_to(ROOT)): run.digest(Path(__file__)),
                         'tests/test_m3w_fitting_switch_report.py': run.digest(ROOT/'tests/test_m3w_fitting_switch_report.py')},
        artifacts={p.name: run.digest(p) for p in sorted(run.PUBLIC.iterdir()) if p.is_file() and p.name != 'verification.json'},
        seconds=time.monotonic()-started)
    run.immutable(run.PUBLIC/'verification.json', verification)
    print(json.dumps(dict(verified=True, groups=108, arithmetic_checks=checks,
                         seconds=verification['seconds'], summary=summary['overall'])), flush=True)


if __name__ == '__main__': main()
