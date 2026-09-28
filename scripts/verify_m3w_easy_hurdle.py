"""Independent arithmetic and role checks for the paired easy-risk experiment."""
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_hurdle as run
from scripts.verify_m3w_query_excess_refit import check_selected_costs
from scripts.verify_m3w_subset_excess import check_fixed_denominator
from scripts.verify_m3w_fixed_floor_tail import reduce_check

TESTS = ['tests/test_m3w_easy_hurdle.py', 'tests/test_m3w_easy_hurdle_verification.py',
         'tests/test_m3w_query_utility.py', 'tests/test_m3w_query_excess_verification.py',
         'tests/test_m3w_easy_hurdle_portable.py', 'tests/test_m3w_easy_hurdle_transport.py',
         'tests/test_m3w_easy_hurdle_collection.py', 'tests/test_m3w_easy_hurdle_training_audit.py',
         'tests/test_m3w_easy_hurdle_restore.py']


def check_actions(a, sites, recordings, frames, utility):
    groups = {}
    risk = dict(raw=a['raw_scores'])
    for arm in run.api.ARMS:
        p = a[arm+'_scores']
        assert np.isfinite(p).all() and ((p[:, 0] >= 0) & (p[:, 0] <= 1)).all()
        assert (p[:, 1:3] >= 0).all()
        np.testing.assert_allclose(p[:, 3], p[:, 0]*(p[:, 2]-.02*p[:, 1]), rtol=1e-5, atol=1e-7)
        risk[arm] = np.column_stack((risk['raw'][:, 0], p[:, 3]))
    anchors = {arm: a['eligible'] & (q <= 0).all(1) for arm, q in risk.items()}
    np.testing.assert_array_equal(a['common_anchor'], np.logical_and.reduce(list(anchors.values())))
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
                assert take.sum() == anchor[ix].sum()
                scale = np.maximum(abs(q[ix]).max(0), 1e-12)
                assert all(sum(float(q[i, k]/scale[k]) for i in ix[take]) <= 1e-10 for k in (0, 1))
                assert utility[ix][take].sum()+1e-10 >= utility[ix][anchor[ix]].sum()
                checks += 1
        changed += not np.array_equal(a['supervised_matched'][ix], a['marginal_matched'][ix])
    return checks, changed


def check_quality(record, pred, truth, recordings, frames):
    groups = {}
    for i in np.flatnonzero(np.isfinite(truth).all(1)):
        groups.setdefault((str(recordings[i]), int(frames[i])), []).append(i)
    values = []
    for indices in groups.values():
        result = np.zeros(8)
        for i in indices:
            easy, reference, harm = map(float, truth[i]); pi = np.clip(float(pred[i, 0]), 1e-7, 1-1e-7)
            err = float(pred[i, 3])-easy*(harm-.02*reference)
            result += [easy, (pi-easy)**2, -easy*np.log(pi)-(1-easy)*np.log1p(-pi),
                       err**2, err, easy*(float(pred[i, 1])-reference)**2,
                       easy*(float(pred[i, 2])-harm)**2, pi]
        values.append(result/len(indices))
    v = np.mean(values, 0)
    expected = dict(easy_rate=v[0], Brier=v[1], log_loss=v[2], signed_MSE=v[3], signed_bias=v[4],
        conditional_reference_MSE=v[5]/v[0] if v[0] else None,
        conditional_harm_MSE=v[6]/v[0] if v[0] else None, predicted_easy_probability=v[7], evaluable_queries=len(groups))
    for k, value in expected.items():
        if value is None:
            assert record[k] is None
        else:
            np.testing.assert_allclose(record[k], value, rtol=1e-7, atol=1e-10)


def main():
    start = time.monotonic(); run.base.torch.set_num_threads(4); run.base.torch.set_num_interop_threads(1)
    cfg, ident, data, jobs, oid, pid, fits, actions = run.load()
    restored = json.loads((run.PUBLIC/'local_restore.json').read_text())
    assert restored['remote_training'] == run.base.artifact(run.PUBLIC/'create_training_freeze.json')
    assert restored['remote_verification'] == run.base.artifact(run.PUBLIC/'create_training_verification.json')
    assert restored['adapter'] == run.base.artifact(ROOT/'scripts/restore_m3w_easy_hurdle_heads.py')
    imported = json.loads((run.PUBLIC/'create_training_freeze.json').read_text())
    for ref in imported['receipt']['artifacts']:
        assert run.base.digest(run.PRIVATE/ref['path']) == ref['sha256']
    replay = json.loads((run.PUBLIC/'fit_replay.json').read_text())
    assert replay['evidence'] == restored['remote_verification']
    assert replay['execution_location'] == 'CREATE_same_runtime' and not replay['cross_architecture_training_replay']
    details = json.loads((run.PRIVATE/'details.json').read_text())
    metrics = {(r['group'], r['site'], r['policy']): r['metric'] for r in details['rows']}
    quality = {(r['group'], r['site'], r['policy']): r['metric'] for r in details['qualities']}
    decisions = {Path(r['path']).stem: r for r in json.loads((run.PUBLIC/'decision_freeze.json').read_text())['groups']}
    checks = changes = cost_views = quality_views = groups = 0
    for c in run.base.floor_api.contexts({k: data[k] for k in run.parent.CAUSAL_KEYS}, jobs, oid):
        cv, _, (floor, _), (neural, _) = run.base.floor_api.costs(c, data, np.arange(len(c['ids'])))
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; d = run.checked(decisions[name]); f = run.checked(d['fit'])
            assert d['future_fields_removed'] and not d['held_outcomes_used'] and not f['held_outcomes_used']
            roles = f['identity']['roles']
            run.base.floor_api.api.assert_roles(roles['producer_sites'], roles['controller_sites'], roles['training_sites'], roles['held_sites'])
            states = {}
            for arm, ref in f['artifacts'].items():
                assert run.base.artifact(ROOT/ref['path']) == ref
                states[arm] = run.api.head.read_checkpoint(ROOT/ref['path'])
                assert states[arm]['step'] == 2000 and states[arm]['unknown_rows_sampled'] == 0
                assert states[arm]['identity'] == f['identity']
            run.api.assert_matched(states['marginal'], states['supervised'])
            assert run.base.artifact(ROOT/d['arrays']['path']) == d['arrays']
            with np.load(ROOT/d['arrays']['path'], allow_pickle=False) as z:
                a = {k: z[k].copy() for k in z.files}
            held = np.flatnonzero(np.isin(data['sites'][c['ids']], roles['held_sites']))
            np.testing.assert_array_equal(a['ids'], c['ids'][held])
            scale = states['marginal']['norm']['cost_scale']
            ridge = json.loads((run.base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
            r = ridge['artifacts']['scores']; assert run.base.artifact(ROOT/r['path']) == r
            with np.load(ROOT/r['path'], allow_pickle=False) as z:
                np.testing.assert_array_equal(a['ids'], z['ids'])
                utility = (z['scores'][:, 5].astype(float)-z['scores'][:, 6].astype(float))/scale
            n, ch = check_actions(a, data['sites'][a['ids']], data['recordings'][a['ids']], data['frames'][a['ids']], utility)
            checks += n; changes += ch
            for site in roles['held_sites']:
                at = data['sites'][a['ids']] == site; ix = held[at]
                for policy in run.POLICIES:
                    take = np.zeros(len(ix), bool) if policy == 'floor' else a[policy][at]
                    m = metrics[(name, site, policy)]
                    check_selected_costs(m, cv[ix], floor[ix], neural[ix], take)
                    check_fixed_denominator(m, floor[ix], neural[ix], np.isfinite(cv[ix]), take); cost_views += 1
                truth = np.column_stack(((cv[ix] > 0) & (cv[ix] <= c['job']['design']['easy_cut']),
                                         floor[ix]/scale, np.maximum(neural[ix]-floor[ix], 0)/scale))
                truth[~np.isfinite(cv[ix])] = np.nan
                for arm in run.api.ARMS:
                    check_quality(quality[(name, site, arm)], a[arm+'_scores'][at], truth,
                                  data['recordings'][c['ids'][ix]], data['frames'][c['ids'][ix]])
                    quality_views += 1
            groups += 1
        run.beat('independent_arithmetic_checked', groups=groups, query_constraints=checks)
    summary = json.loads((run.PUBLIC/'summary.json').read_text()); reductions = 0
    for section, key in [('summary', 'rows'), ('paired', 'contrasts'), ('quality', 'qualities')]:
        for policy, values in summary[section].items():
            rows = [r for r in details[key] if r['policy'] == policy]
            for k, v in values.items():
                reductions += reduce_check(v, rows, k, cfg['bootstrap_seed'], cfg['bootstrap_resamples'])
    for key in ('prediction_replay', 'evaluation_replay'):
        assert json.loads((run.PUBLIC/(key+'.json')).read_text())['exact']
    assert json.loads((run.PUBLIC/'fit_replay.json').read_text())['exact_except_elapsed']
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS], cwd=ROOT, capture_output=True, text=True)
    (run.PRIVATE/'scoped_pytest.txt').write_text(proc.stdout+proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stdout+proc.stderr)
    tests = int(re.search(r'(\d+) passed', proc.stdout).group(1))
    seal = json.loads((run.previous.PUBLIC/'verification.json').read_text())
    bindings = {**seal['source_bindings'], **ident['bindings']}
    extra = ['scripts/verify_m3w_easy_hurdle.py', 'scripts/report_m3w_easy_hurdle.py',
        'scripts/restore_m3w_easy_hurdle_heads.py', 'scripts/verify_m3w_easy_hurdle_portable_training.py',
        'scripts/collect_m3w_easy_hurdle_training.py', 'scripts/train_m3w_easy_hurdle_portable.py', *TESTS]
    bindings.update({p: run.base.digest(ROOT/p) for p in extra})
    artifact_hashes = {p.name: run.base.digest(p) for p in run.PUBLIC.iterdir()
                      if p.suffix in ('.json', '.md', '.svg') and p.name != 'verification.json'}
    run.base.immutable_json(run.PUBLIC/'verification.json', dict(source_bindings=bindings, artifacts=artifact_hashes,
        groups=groups, neural_heads=216, updates=432000, query_constraint_checks=checks, changed_matched_queries=changes,
        cost_views=cost_views, quality_views=quality_views, locality_reductions=reductions, scoped_tests=tests,
        test_log=run.base.artifact(run.PRIVATE/'scoped_pytest.txt'), pid=os.getpid(), seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, all_action_replay_exact=True,
        evaluation_replay_exact=True, first_group_training_replay_exact=True,
        training_replay_location='CREATE_same_runtime', inference_location='local_arm64',
        cross_architecture_training_equivalence=False, full_legacy_suite='not_run',
        cold_raw_rebuild='not_run', independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    run.beat('verification_complete', groups=groups, tests=tests, queries=checks)


if __name__ == '__main__':
    main()
