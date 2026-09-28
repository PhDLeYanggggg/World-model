"""Independently account for decisions, costs and reductions before sealing results."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_risk_priority_policy as run
from scripts.verify_m3w_easy_hurdle import check_quality
from scripts.verify_m3w_query_excess_refit import check_selected_costs
from scripts.verify_m3w_subset_excess import check_fixed_denominator
from scripts.verify_m3w_fixed_floor_tail import reduce_check

TESTS = ['tests/test_m3w_easy_risk_priority.py', 'tests/test_m3w_easy_risk_priority_policy.py',
         'tests/test_m3w_easy_risk_priority_report.py', 'tests/test_m3w_easy_risk_priority_readout.py',
         'tests/test_m3w_easy_hurdle_verification.py', 'tests/test_m3w_query_excess_verification.py',
         'tests/test_m3w_easy_risk_priority_recovery.py', 'tests/test_m3w_easy_hurdle_accounting.py']


def check_actions(a, sites, recordings, frames, utility):
    risk = {'raw': a['raw_scores']}
    assert np.isfinite(risk['raw']).all() and np.isfinite(utility).all()
    for arm in run.repair.ARMS:
        p = a[arm+'_scores']
        assert np.isfinite(p).all() and ((p[:, 0] >= 0) & (p[:, 0] <= 1)).all()
        assert (p[:, 1:3] >= 0).all()
        np.testing.assert_allclose(p[:, 3], p[:, 0]*(p[:, 2]-.02*p[:, 1]), rtol=1e-5, atol=1e-7)
        risk[arm] = np.column_stack((risk['raw'][:, 0], p[:, 3]))
    anchors = {arm: a['eligible'] & (q <= 0).all(1) for arm, q in risk.items()}
    np.testing.assert_array_equal(a['common_anchor'], np.logical_and.reduce(list(anchors.values())))
    groups = {}
    for i, key in enumerate(zip(sites, recordings, frames)):
        groups.setdefault(tuple(map(str, key)), []).append(i)
    checks = changed = 0
    for positions in groups.values():
        ix = np.asarray(positions)
        for arm, q in risk.items():
            np.testing.assert_array_equal(a[arm+'_independent'][ix], anchors[arm][ix])
            for suffix, anchor in [('joint', anchors[arm]), ('matched', a['common_anchor'])]:
                take = a[arm+'_'+suffix][ix]
                assert take.dtype == bool and not (take & ~a['eligible'][ix]).any()
                assert int(take.sum()) == int(anchor[ix].sum())
                scale = np.maximum(abs(q[ix]).max(0), 1e-12)
                assert all(sum(float(q[i, k]/scale[k]) for i in ix[take]) <= 1e-10 for k in (0, 1))
                assert utility[ix][take].sum()+1e-10 >= utility[ix][anchor[ix]].sum()
                checks += 1
        changed += not np.array_equal(a['risk_priority_matched'][ix], a['uncapped_matched'][ix])
    return checks, changed


def check_contrasts(reported, left, right):
    expected = {
        'ADE_gain_percent': 100*(right['error_sum']-left['error_sum'])/right['error_sum'] if right['error_sum'] > 0 else None,
        'intervention_difference_pp': 100*(left['intervention_rate']-right['intervention_rate']),
        'all_reference_harm_reduction_pp': 100*(right['positive_harm_over_all_floor']-left['positive_harm_over_all_floor'])}
    for output, key, multiplier in [('easy_gain_floor_difference_pp', 'easy_gain_floor', 1),
                                   ('hard_gain_floor_difference_pp', 'hard_gain_floor', 1),
                                   ('selected_harm_reduction_pp', 'selected_positive_harm_ratio', -100)]:
        expected[output] = None if left[key] is None or right[key] is None else multiplier*(left[key]-right[key])
    assert set(expected) == set(reported)
    for key, v in expected.items():
        if v is None:
            assert reported[key] is None
        else:
            np.testing.assert_allclose(reported[key], v, rtol=1e-10, atol=1e-10)


def main():
    start = time.monotonic()
    run.repair.torch.set_num_threads(4); run.repair.torch.set_num_interop_threads(1)
    cfg, ident, data, jobs, oid, pid, fits, actions, old_easy = run.load()
    trained = json.loads((run.PUBLIC/'create_training_freeze.json').read_text())
    restored = json.loads((run.PUBLIC/'local_restore.json').read_text())
    assert restored['heads'] == 216 and not restored['new_training']
    assert restored['restore_code_sha256'] == run.digest(ROOT/'scripts/restore_m3w_easy_risk_priority_heads.py')
    for ref in trained['receipt']['artifacts']:
        assert run.digest(run.PRIVATE/ref['path']) == ref['sha256']
    assert trained['receipt']['control_parent_states_exact'] == 108
    assert trained['replay_receipt']['repair_first_pair_replay_exact']
    details = json.loads((run.PRIVATE/'details.json').read_text())
    metric = {(r['group'], r['site'], r['policy']): r['metric'] for r in details['rows']}
    quality = {(r['group'], r['site'], r['policy']): r['metric'] for r in details['qualities']}
    assert len(metric) == len(details['rows']) == 108*2*len(run.POLICIES)
    assert len(quality) == len(details['qualities']) == 432
    frozen = json.loads((run.PUBLIC/'decision_freeze.json').read_text())
    assert frozen['identity'] == ident
    refs = {Path(ref['path']).stem: ref for ref in frozen['groups']}
    assert len(refs) == 108
    counts = dict(groups=0, query_constraints=0, changed_matched_queries=0, cost_views=0, quality_views=0, reductions=0)
    for c in run.base.floor_api.contexts(run.causal_view(data), jobs, oid):
        cv, _, (floor, _), (neural, _) = run.base.floor_api.costs(c, data, np.arange(len(c['ids'])))
        for pair in range(6):
            name = c['name']+f'_pair{pair}'
            doc = run.checked(refs[name]); fit = run.checked(doc['fit'])
            assert doc['future_fields_removed'] and not doc['held_outcomes_used'] and not fit['held_outcomes_used']
            roles = fit['identity']['source_identity']['roles']
            run.base.floor_api.api.assert_roles(roles['producer_sites'], roles['controller_sites'], roles['training_sites'], roles['held_sites'])
            states = {}
            for arm, ref in fit['artifacts'].items():
                assert run.base.artifact(ROOT/ref['path']) == ref
                state = run.api.head.read_checkpoint(ROOT/ref['path']); states[arm] = state
                assert state['step'] == 2000 and state['unknown_rows_sampled'] == 0
                assert state['identity'] == fit['identity']
            run.repair.assert_matched(states['uncapped'], states['risk_priority'])
            old = run.api.head.read_checkpoint(run.original.PRIVATE/'heads'/name/'supervised'/'checkpoint.pt.gz')
            run.repair.assert_parent_control(old, states['uncapped'])
            assert run.base.artifact(ROOT/doc['arrays']['path']) == doc['arrays']
            with np.load(ROOT/doc['arrays']['path'], allow_pickle=False) as z:
                a = {k: z[k].copy() for k in z.files}
            held = np.flatnonzero(np.isin(data['sites'][c['ids']], roles['held_sites']))
            np.testing.assert_array_equal(a['ids'], c['ids'][held])
            scale = states['uncapped']['norm']['cost_scale']
            ridge = json.loads((run.base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
            ref = ridge['artifacts']['scores']; assert run.base.artifact(ROOT/ref['path']) == ref
            with np.load(ROOT/ref['path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(a['ids'], z['ids'])
                utility = (z['scores'][:, 5].astype(float)-z['scores'][:, 6].astype(float))/scale
            n, ch = check_actions(a, data['sites'][a['ids']], data['recordings'][a['ids']], data['frames'][a['ids']], utility)
            counts['query_constraints'] += n; counts['changed_matched_queries'] += ch
            for site in roles['held_sites']:
                at = data['sites'][a['ids']] == site; ix = held[at]
                for policy in run.POLICIES:
                    take = np.zeros(len(ix), bool) if policy == 'floor' else a[policy][at]
                    m = metric[name, site, policy]
                    check_selected_costs(m, cv[ix], floor[ix], neural[ix], take)
                    check_fixed_denominator(m, floor[ix], neural[ix], np.isfinite(cv[ix]), take)
                    counts['cost_views'] += 1
                truth = np.column_stack(((cv[ix] > 0) & (cv[ix] <= c['job']['design']['easy_cut']),
                    floor[ix]/scale, np.maximum(neural[ix]-floor[ix], 0)/scale))
                truth[~np.isfinite(cv[ix])] = np.nan
                for arm in run.repair.ARMS:
                    check_quality(quality[name, site, arm], a[arm+'_scores'][at], truth,
                        data['recordings'][c['ids'][ix]], data['frames'][c['ids'][ix]])
                    counts['quality_views'] += 1
            counts['groups'] += 1
        run.beat('independent_arithmetic_checked', **counts)
    assert len(details['contrasts']) == 216*len(run.CONTRASTS)
    for row in details['contrasts']:
        left, right = row['policy'].split('_vs_')
        check_contrasts(row['metric'], metric[row['group'], row['site'], left], metric[row['group'], row['site'], right])
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    for section, key in [('summary', 'rows'), ('paired', 'contrasts'), ('quality', 'qualities')]:
        for policy, metrics in summary[section].items():
            rows = [r for r in details[key] if r['policy'] == policy]
            for key_, record in metrics.items():
                counts['reductions'] += reduce_check(record, rows, key_, cfg['bootstrap_seed'], cfg['bootstrap_resamples'])
    for seed, policies in summary['by_seed'].items():
        for policy, metrics in policies.items():
            rows = [r for r in details['rows'] if r['policy'] == policy and r['seed'] == int(seed)]
            for key, record in metrics.items():
                counts['reductions'] += reduce_check(record, rows, key, cfg['bootstrap_seed'], cfg['bootstrap_resamples'])
    assert summary['gates'] == run.gates_for(summary['paired'], details['rows'], details['contrasts'])
    for name in ('prediction_replay', 'evaluation_replay'):
        replay = json.loads((run.PUBLIC/(name+'.json')).read_text()); assert replay['exact']
        if name == 'evaluation_replay':
            for key in ('summary', 'details'):
                assert run.base.artifact(ROOT/replay[key]['path']) == replay[key]
    from scripts.report_m3w_easy_risk_priority import put, main as report, structural_support
    support_doc = json.loads((run.PUBLIC/'structural_support.json').read_text())
    ref = support_doc['source_details']; assert run.base.artifact(ROOT/ref['path']) == ref
    assert support_doc['support'] == structural_support(json.loads((ROOT/ref['path']).read_text())['rows'])
    report()
    from scripts.analyze_m3w_easy_risk_priority import main as analyze, exchange
    analyze()
    posthoc = json.loads((run.PUBLIC/'posthoc_accounting.json').read_text())
    for name, metrics in posthoc['comparisons'].items():
        left, right = name.split('_vs_')
        rows = [dict(site=r['site'], metric=exchange(r['metric'], metric[r['group'], r['site'], right]))
                for r in details['rows'] if r['policy'] == left]
        for key, record in metrics.items():
            counts['reductions'] += reduce_check(record, rows, key, cfg['bootstrap_seed'], cfg['bootstrap_resamples'])
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS], cwd=ROOT, capture_output=True, text=True)
    log = run.PRIVATE/'readout_pytest.txt'; log.write_text(proc.stdout+proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stdout+proc.stderr)
    tests = int(re.search(r'(\d+) passed', proc.stdout).group(1))
    put('verification_report.md', '# Verification Scope\n\n'+
        f"Checked {counts['groups']} groups, {counts['query_constraints']:,} query constraints, "
        f"{counts['cost_views']:,} cost views, {counts['quality_views']:,} quality views and "
        f"{counts['reductions']:,} locality reductions. All contrasts use independently checked "
        'signs and undefined-denominator rules. All actions and the complete readout replay exactly. '
        'These are repeated-role checks, not independent sample counts.\n\n'+
        f"All216 checkpoint hashes match; all108 uncapped controls reproduce old training. "
        f"Only the first repaired pair was retrained for exact replay, not all108. {tests} scoped tests "
        f"in{len(TESTS)} files pass. Full legacy suite and cold raw rebuild: not_run.\n\n"+
        'Engineering verification does not establish efficacy, calibration or deployment safety. '
        'Independent confirmation stays closed; original primary not replaced; Stage5C/SMC disabled.\n')
    parent_seal = json.loads((run.original.PUBLIC/'verification.json').read_text())
    bindings = {**parent_seal['source_bindings'], **ident['bindings']}
    extras = ['scripts/verify_m3w_easy_risk_priority.py', 'scripts/report_m3w_easy_risk_priority.py',
              'scripts/report_m3w_easy_risk_priority_training.py', 'scripts/audit_m3w_easy_risk_priority_support.py',
              'scripts/recover_m3w_easy_risk_priority_replay.py', 'scripts/analyze_m3w_easy_risk_priority.py', *TESTS]
    bindings.update({p: run.digest(ROOT/p) for p in extras})
    for rel, sha in bindings.items():
        assert run.digest(ROOT/rel) == sha
    artifacts = {p.name: run.digest(p) for p in run.PUBLIC.iterdir()
        if p.suffix in ('.md', '.json', '.svg') and p.name not in ('verification.json', 'execution_status.json', 'operation_zh.md')}
    run.immutable(run.PUBLIC/'verification.json', dict(source_bindings=bindings, artifacts=artifacts,
        **counts, tests=tests, test_files=len(TESTS), test_log=run.base.artifact(log),
        seconds=time.monotonic()-start, independent_confirmation=False, deployment_changed=False,
        full_legacy_suite='not_run', cold_raw_rebuild='not_run', stage5c_executed=False, smc_enabled=False))
    print(json.dumps(dict(verified=True, **counts, tests=tests)))


if __name__ == '__main__':
    main()
