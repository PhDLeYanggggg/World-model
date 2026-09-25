"""Publish every frozen attribution, not a selected policy or oracle forecast."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_opportunity_diagnosis import REASONS

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_opportunity_diagnosis_v1'


def number(x):
    return 'undefined' if x is None else f'{x:.6f}'


def interval(x):
    return 'undefined' if x is None else '['+', '.join(number(v) for v in x)+']'


def main():
    raw = (PUBLIC/'analysis.json').read_bytes()
    result = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    verify = json.loads((PUBLIC/'verification.json').read_text())
    if verify['analysis_sha256'] != sha or not verify['exact_replay']:
        raise ValueError('Complete deterministic replay required')
    for path, expected in result['identity']['bindings'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Diagnostic version changed')
    previous = ROOT/'outputs/publication_readiness_2026_09/european_protected_motion_v1/analysis.json'
    if hashlib.sha256(previous.read_bytes()).hexdigest() != result['identity']['previous_analysis_sha256']:
        raise ValueError('Protected-motion evidence changed')
    original = json.loads(previous.read_text())
    policy_count, subset_count, locality_count = 0, 0, 0
    for entry in result['seeds'].values():
        for lineage in entry['producer_lineage']:
            assert not set(lineage['training_sites']) & set(lineage['evaluated_sites'])
        for name, p in entry['policies'].items():
            policy_count += 1
            assert p['causal_reasons_match_frozen_decisions']
            led = p['ledgers']['all']
            assert led['indexed_rows'] == original['source_rows'] == 318969
            assert led['supported_rows'] == 311922 and led['unknown_rows'] == 7047
            count = sum(s['counts']['switch'] for s in led['by_scene'].values())
            np.testing.assert_allclose(count/318969, original['policies'][name]['full']['switch_rate'], atol=0, rtol=0)
            for l in p['ledgers'].values():
                subset_count += 1
                for site in l['by_scene'].values():
                    locality_count += 1
                    assert sum(site['counts'].values()) == site['indexed_rows']
                    assert site['supported_rows']+site['unknown_rows'] == site['indexed_rows']
                    s = site['contributions_percent']
                    if s['oracle_gain'] is not None:
                        np.testing.assert_allclose(s['oracle_gain'], s['captured_gain']+
                            sum(s['missed_'+k] for k in REASONS[:-1]), rtol=1e-10, atol=1e-10)
                        np.testing.assert_allclose(s['net_gain'], s['captured_gain']-s['switched_harm'], atol=1e-10)
                        np.testing.assert_allclose(s['oracle_regret'], s['oracle_gain']-s['net_gain'], atol=1e-10)
    assert (policy_count, subset_count, locality_count) == (48, 144, 1728)
    summary = dict(result_source=result['result_source'], analysis_sha256=sha, new_training=False,
        uncertainty='conditional_source_locality_bootstrap_not_independent_confirmation',
        oracle_is_diagnostic_only=True, reserved_roles_opened=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False, seeds={})
    for seed, x in result['seeds'].items():
        summary['seeds'][seed] = dict(candidates=x['candidates'], producer_comparisons=x['producer_comparisons'],
            policies={n: dict(ledgers=p['ledgers'], risk_bands=p['risk_bands']) for n, p in x['policies'].items()})
    (PUBLIC/'summary_metrics.json').write_text(json.dumps(summary, separators=(',', ':'), allow_nan=False)+'\n')
    audit = dict(result_source='fresh_independent_arithmetic_checks_on_verified_diagnostic',
        analysis_sha256=sha, summary_sha256=hashlib.sha256((PUBLIC/'summary_metrics.json').read_bytes()).hexdigest(),
        policy_checks=policy_count, subset_checks=subset_count, locality_conservation_checks=locality_count,
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), all_passed=True)
    (PUBLIC/'accounting_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    lines = ['# Candidate Opportunity and Rejection Tables', '',
        'Fresh arithmetic on cached_verified source models. Oracle uses future labels only for diagnosis.',
        'No new fitting, threshold choice, reserved readout or deployment.', '',
        '## Candidate Opportunity', '',
        '| Seed | Candidate | Raw ADE gain vs CV (%) | Hindsight oracle gain (%) | Oracle conditional locality CI |',
        '|---|---|---:|---:|---|']
    for seed, x in result['seeds'].items():
        for name, c in x['candidates'].items():
            o = c['hindsight_oracle_vs_CV']
            lines.append(f"| {seed} | {name} | {number(c['raw_vs_CV']['equal_scene_gain_percent'])} | {number(o['equal_scene_gain_percent'])} | {interval(o['scene_bootstrap_ci95'])} |")
    lines += ['', '## Conserved Policy Decomposition', '',
        'Contributions are percentage points of the same locality CV error denominator. Oracle is not a learned result.',
        'Gate priority: support, motion, utility, event mass, risk. Sequential attribution is not an intervention effect.',
        '| Policy | Subset | Oracle | Captured | Switched harm | Net | Support missed | Motion missed | Utility missed | Zero-mass missed | Risk missed | Gross capture (%) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields = ('oracle_gain', 'captured_gain', 'switched_harm', 'net_gain',
              *('missed_'+k for k in REASONS[:-1]))
    for x in result['seeds'].values():
        for name, p in x['policies'].items():
            for subset, ledger in p['ledgers'].items():
                values = [number(ledger['summary'][k]['equal_locality']) for k in fields]
                fraction = ledger['gain_capture_fraction']
                values.append(number(None if fraction is None else 100*fraction))
                lines.append('| '+name+' | '+subset+' | '+' | '.join(values)+' |')
    lines += ['', '## Producer Population Comparison', '',
        'Positive means the eight-locality producer beats the named four-locality producer on the same held rows.',
        'Both ordered controls are retained; no better-of-two selection. This is not isolated randomized sample-size evidence.',
        '| Seed | Four-locality control | Eight-locality gain (%) | Conditional locality CI |',
        '|---|---|---:|---|']
    for seed, x in result['seeds'].items():
        for name, m in x['producer_comparisons'].items():
            lines.append(f"| {seed} | {name} | {number(m['equal_scene_gain_percent'])} | {interval(m['scene_bootstrap_ci95'])} |")
    lines += ['', '## Fixed Risk Bands', '',
        'Counts retain unsupported future rows; precision denominators use supported rows only. Not a threshold search.',
        '| Policy | Risk band | Indexed | Known | Beneficial | Harmful | Positive utility | Positive-utility beneficial | Switched (all rows) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in result['seeds'].values():
        for name, p in x['policies'].items():
            for band, v in p['risk_bands'].items():
                lines.append('| '+name+' | '+band+' | '+' | '.join(str(v[k]) for k in
                    ('indexed_rows', 'supported_rows', 'beneficial_rows', 'harmful_rows',
                     'utility_positive_rows', 'utility_positive_beneficial_rows', 'switched_rows'))+' |')
    lines += ['', 'Per-locality contributions and conditional intervals are retained in summary_metrics.json.',
        'Unknown and zero-reference cases remain explicit. These are source-only image-pixel obs8/pred12 raw-stride12 results;',
        'not t50, seconds, metric, true 3D, foundation, physical safety or submission-ready evidence. Stage5C/SMC off.', '']
    (PUBLIC/'tables.md').write_text('\n'.join(lines))
    print(json.dumps(audit))


if __name__ == '__main__': main()
