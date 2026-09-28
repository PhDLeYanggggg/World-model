"""Independent arithmetic and descriptive reporting of immutable fitting diagnostics."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import manage_m3w_easy_component_diagnostic as manager

KEYS = ('row_MSE', 'query_mean_MSE', 'marginal')
ARMS = ('uncapped', 'risk_priority')
PARTITIONS = ('not_easy', 'easy_zero_harm', 'easy_positive_harm')
COMPONENTS = ('occurrence', 'reference', 'harm')


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)


def check_node(node):
    checks = 0
    combos = node['transplants']; assert set(combos) == {str(i) for i in range(8)}
    for row in combos.values():
        assert row['row_MSE'] >= 0 and row['query_mean_MSE'] >= 0
        assert row['query_mean_MSE'] <= row['row_MSE']+1e-10
        close(row['marginal'], (row['row_MSE']+row['query_mean_MSE'])/2); checks += 1
    for key in KEYS:
        v = {int(k): row[key] for k, row in combos.items()}; independent = []
        # Closed subset formula, independent of the runner's permutation loop.
        for bit in range(3):
            other = [i for i in range(3) if i != bit]; mask = sum(1 << i for i in other)
            delta = (v[1 << bit]-v[0])/3 + (v[7]-v[mask])/3
            delta += sum((v[(1 << j) | (1 << bit)]-v[1 << j])/6 for j in other)
            independent.append(delta)
        close(independent, node['attribution'][key]); close(sum(independent), v[7]-v[0]); checks += 2
    for arm, mask in [('uncapped', '0'), ('risk_priority', '7')]:
        row = node['arms'][arm]
        for key in KEYS: close(row['objective'][key], combos[mask][key]); checks += 1
        p = row['partitions']; assert set(p) == {'all', *PARTITIONS}
        close(p['all']['weight_mass'], 1)
        close(sum(p[k]['weight_mass'] for k in PARTITIONS), 1)
        assert sum(p[k]['rows'] for k in PARTITIONS) == p['all']['rows'] == node['known_rows']
        close(sum(p[k]['MSE_contribution'] for k in PARTITIONS), row['objective']['row_MSE'])
        close(p['all']['MSE_contribution'], row['objective']['row_MSE']); checks += 5
        for r in p.values():
            assert r['rows'] >= 0 and 0 <= r['weight_mass'] <= 1+1e-12
            close(sum(r['squared_components'])+sum(r['cross_components']), r['MSE_contribution'])
            if r['weight_mass'] == 0:
                assert r['conditional_MSE'] is None and r['rows'] == 0
            else: close(r['conditional_MSE']*r['weight_mass'], r['MSE_contribution'])
            checks += 2
    return checks


def main():
    start = time.monotonic(); reg = manager.registration()
    assert reg == json.loads((manager.PUBLIC/'registration.json').read_text())
    collected = json.loads((manager.PRIVATE/'collected.json').read_text())
    receipt, replay = (collected[k] for k in ('receipt.json', 'replay_receipt.json'))
    assert receipt['artifacts'] == replay['artifacts'] and replay['replay_exact']
    for doc in (receipt, replay):
        assert doc['groups'] == 108 and doc['parameter_updates'] == 0
        assert doc['registration_sha256'] == manager.digest(manager.PUBLIC/'registration.json')
        assert not any(doc[k] for k in ('held_outcomes_used', 'independent_roles_read', 'policy_actions_computed'))
    groups = collected['groups']; assert len(groups) == 108
    refs = {Path(r['path']).stem: r['sha256'] for r in receipt['artifacts']}
    training = json.loads((manager.PARENT/'create_training_freeze.json').read_text())['receipt']
    old_refs = {(r['group'], r['arm']): r['sha256'] for r in training['artifacts']}
    nodes, checks = [], 0
    for row in groups:
        raw = json.dumps(row, indent=2, allow_nan=False)+'\n'
        assert hashlib.sha256(raw.encode()).hexdigest() == refs[row['group']]
        assert row['checkpoints'] == {arm: old_refs[row['group'], arm] for arm in ARMS}
        assert not row['policy_actions_computed'] and row['parameter_updates'] == 0
        roles = row['identity']['roles']; sources = row['sources']
        assert set(sources) == set(roles['training_sites']) and len(sources) == 2
        for k in ('producer_sites', 'controller_sites', 'held_sites'): assert not set(sources) & set(roles[k])
        assert sum(s['known_rows'] for s in sources.values()) == row['known_rows']
        for n in [row['source_balanced'], *sources.values()]: checks += check_node(n)
        for arm in ARMS:
            for key in KEYS:
                close(row['source_balanced']['arms'][arm]['objective'][key],
                      np.mean([s['arms'][arm]['objective'][key] for s in sources.values()]))
                checks += 1
        nodes.append(row['source_balanced'])
    loss = {arm: {key: float(np.mean([n['arms'][arm]['objective'][key] for n in nodes])) for key in KEYS} for arm in ARMS}
    att = {key: {component: float(np.mean([n['attribution'][key][j] for n in nodes]))
                 for j, component in enumerate(COMPONENTS)} for key in KEYS}
    quality = {arm: {key: float(np.mean([n['arms'][arm][key] for n in nodes])) for key in
                   ('Brier', 'signed_bias', 'conditional_reference_MSE', 'conditional_harm_MSE')} for arm in ARMS}
    partitions = {arm: {key: float(np.mean([n['arms'][arm]['partitions'][key]['MSE_contribution'] for n in nodes]))
                       for key in PARTITIONS} for arm in ARMS}
    components = {arm: dict(
        squared_mean=np.mean([n['arms'][arm]['partitions']['all']['squared_components'] for n in nodes], 0).tolist(),
        cross_mean=np.mean([n['arms'][arm]['partitions']['all']['cross_components'] for n in nodes], 0).tolist()) for arm in ARMS}
    result = dict(result_source='fresh_run_fitting_only_diagnostic_exact_replay', input_source='cached_verified',
        groups=108, source_views=216, repeated_views_not_independent=True, loss=loss,
        quality=quality, loss_change_attribution=att, partition_MSE_contribution=partitions, residual_components=components,
        improved_groups={key: sum(n['arms']['risk_priority']['objective'][key] < n['arms']['uncapped']['objective'][key] for n in nodes) for key in KEYS},
        contribution_increasing_groups={key: {c: sum(n['attribution'][key][j] > 0 for n in nodes)
                                      for j, c in enumerate(COMPONENTS)} for key in KEYS},
        arithmetic_checks=checks, parameter_updates=0, held_outcomes_used=False,
        useful_switch_ranking='not_run_signed_benefit_not_in_packet', independent_roles_read=False,
        policy_actions_computed=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    manager.immutable(manager.PUBLIC/'analysis.json', result)
    manager.immutable(manager.PUBLIC/'compute_receipt.json', dict(first=receipt, replay=replay,
        collected_sha256=manager.digest(manager.PRIVATE/'collected.json')))
    lines = ['# Conditional Risk Components: Fitting-Only Results', '', '## Material Passport', '',
        'Fresh component accounting and full exact replay; cached_verified packets and216 frozen heads. '
        'No parameter updates, new actions, held readout, independent calibration or deployment. '
        'All108 groups retained. Source views repeat localities; these are not216 independent samples.', '',
        'The table uses the exact expected source/query-balanced marginal fitting objective over all '
        'known rows, not the128-query monitor subset used during training.', '',
        '| Arm | Row risk MSE | Query-mean risk MSE | Half-sum objective |', '|---|---:|---:|---:|']
    for arm in ARMS: lines.append('| '+arm+' | '+' | '.join(f'{loss[arm][k]:.9g}' for k in KEYS)+' |')
    lines += ['', f"Repaired heads improve full-fitting marginal objective in{result['improved_groups']['marginal']}/108 groups.", '',
        '## Predicted-Factor Replacement Accounting', '',
        'All8 combinations are scored without constructing actions. Entries average marginal loss '
        'changes over all6 replacement orders, then108 repeated groups. Positive means worse fitting. '
        'They sum to risk-priority minus uncapped loss, not to a causal explanation of held error.', '',
        '| Quantity | Occurrence replacement | Reference replacement | Harm replacement |', '|---|---:|---:|---:|']
    for key in KEYS: lines.append('| '+key+' | '+' | '.join(f'{att[key][c]:.9g}' for c in COMPONENTS)+' |')
    lines += ['', '## Label Partitions', '',
        'Partitions share the full known-query weights and add to row MSE; they are not individually '
        'renormalized here. A non-easy row has zero easy-risk target, not necessarily zero overall harm.', '',
        '| Arm | Non-easy contribution | Easy zero-harm contribution | Easy positive-harm contribution |', '|---|---:|---:|---:|']
    for arm in ARMS: lines.append('| '+arm+' | '+' | '.join(f'{partitions[arm][k]:.9g}' for k in PARTITIONS)+' |')
    lines += ['', '## Exact Residual Identity', '',
        'Order: occurrence/reference/harm. Cross order: occurrence-reference, occurrence-harm, reference-harm. '
        'Squared terms plus doubled cross terms equal MSE; negative cross terms represent cancellation. '
        'Attribution is order-dependent arithmetic, not an oracle deployable model.', '']
    for arm in ARMS:
        lines += [f"- {arm}: squared={components[arm]['squared_mean']}; doubled cross={components[arm]['cross_mean']}."]
    lines += ['', '## Boundaries', '',
        'Fitting loss is not transport performance. This diagnostic does not evaluate useful-switch '
        'ranking because signed benefit is absent from its fitting packets. No selected-risk gate '
        'is repaired by these calculations. Unknown labels remain excluded, not zero. Only already-opened '
        'development fitting roles are used. Image-local detector silver, observation8/prediction12, '
        'raw stride12. No metric, seconds, human-gold, physical-safety, true3D, foundation or submission '
        'claim. Stage5C/SMC remain disabled.']
    put('results.md', '\n'.join(lines)+'\n')
    tests = ['tests/test_m3w_easy_component_diagnostic.py', 'tests/test_m3w_easy_component_report.py',
             'tests/test_m3w_easy_risk_priority.py', 'tests/test_m3w_easy_hurdle_accounting.py']
    p = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=ROOT, capture_output=True, text=True)
    (manager.PRIVATE/'pytest.txt').write_text(p.stdout+p.stderr)
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    count = int(re.search(r'(\d+) passed', p.stdout).group(1))
    bindings = {**reg['code_bindings'], **reg['control_bindings'],
        **{p: manager.digest(ROOT/p) for p in [str(Path(__file__).relative_to(ROOT)), *tests]}}
    put('verification_report.md', f'# Verification Scope\n\n{checks:,} independent arithmetic checks; '
        f'{count} tests in{len(tests)} files. All108 remote group records match exact replays and hashes. '
        'Both frozen checkpoint hashes and fitting roles are checked for every group. '
        'Row/query/source weighting and factor attribution are checked independently.\n\n'
        'No local4.91GB packet copy, no cold raw rebuild, no full legacy integration, no new model '
        'training or held readout. These checks verify descriptive accounting, not a research success gate.\n')
    artifacts = {p.name: manager.digest(p) for p in manager.PUBLIC.iterdir()
                 if p.suffix in ('.json', '.md') and p.name not in ('verification.json', 'execution_status.json', 'operation_zh.md')}
    manager.immutable(manager.PUBLIC/'verification.json', dict(source_bindings=bindings, artifacts=artifacts,
        groups=108, checks=checks, tests=count, test_files=len(tests), seconds=time.monotonic()-start,
        full_legacy_suite='not_run', cold_raw_rebuild='not_run', deployment_changed=False))
    print(json.dumps(dict(verified=True, checks=checks, tests=count, loss_change_attribution=att)))


def put(name, text):
    p = manager.PUBLIC/name
    if p.exists(): assert p.read_text() == text, 'Immutable report differs'
    else: p.write_text(text)


if __name__ == '__main__': main()
