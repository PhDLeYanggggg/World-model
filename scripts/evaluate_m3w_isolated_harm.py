"""Fixed complete continuation readout, including frozen-cohort harm diagnosis."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import time

from scripts import run_m3w_isolated_harm as run
from src.world_model import m3w_isolated_harm as api
from src.world_model import m3w_scaled_risk_readout as metrics
from scripts.report_m3w_shared_train_risk import check_policy, equal_number

PUBLIC = run.PUBLIC/'readout'
ARMS = ('local', 'anchor', 'old_risk')+api.ARMS
COMPARATORS = ('anchor', 'utility_continue', 'joint_risk', 'local', 'old_risk')
PAIRS = tuple((a, 'isolated_risk') for a in COMPARATORS)+(
    ('utility_continue', 'joint_risk'), ('anchor', 'utility_continue'))


def evaluate(args, predictions):
    if set(predictions) != set(ARMS):
        raise ValueError('Complete frozen and continued policies required')
    x, env, y, _, sites, recs, frames, pr = args
    eligible = api.previous.prior.eligible(args)
    api.np.testing.assert_array_equal(predictions['isolated_risk'][:, [0, 2, 3]], predictions['anchor'][:, [0, 2, 3]])
    actions = {a: eligible & (api.core.signed(p)[:, 0] > 0) & (api.core.signed(p)[:, 1:] <= 0).all(1)
               for a, p in predictions.items()}
    policies = {a: metrics.policy_score(y, action, env) for a, action in actions.items()}
    scores = {a: metrics.previous.score(p, y, pr, sites, recs, frames, api.np.ones(len(x), bool))['conditional_MSE']
              for a, p in predictions.items()}
    known = api.np.isfinite(y).all(1)
    den = float(y[known, 2].sum())
    contrasts, pairs, matched = {}, {}, {}
    for left, right in PAIRS:
        key = right+'_minus_'+left
        a, b = scores[left], scores[right]
        contrasts[key+'_MSE'] = b-a if a is not None and b is not None else None
        aa, bb = metrics.previous.policy.matched(actions[left], actions[right],
            api.core.signed(predictions[left])[:, 0], api.core.signed(predictions[right])[:, 0],
            recs, frames, api.np.arange(len(x)))
        matched[key] = {left: metrics.policy_score(y, aa, env), right: metrics.policy_score(y, bb, env)}
        for mode, a, b in (('full', actions[left], actions[right]), ('matched', aa, bb)):
            pair = metrics.previous.paired_completion(y, a, b, env)
            pairs[key+'_'+mode] = pair
            contrasts[key+'_'+mode] = 100*pair['lower_mass']/den if den > 0 else None
    return dict(rows=len(x), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        full_known_reference_mass=den, scores=scores, policies=policies, contrasts=contrasts,
        pairs=pairs, matched=matched, protected_outputs_exact=True,
        diagnostics={a: metrics.selected_diagnostics(p, y, actions[a], pr['scale']) for a, p in predictions.items()},
        fixed_anchor_cohort={a: metrics.selected_diagnostics(p, y, actions['anchor'], pr['scale']) for a, p in predictions.items()},
        recordings={str(r): {a: metrics.policy_score(y[recs == r], v[recs == r], env[recs == r])
                            for a, v in actions.items()} for r in api.np.unique(recs)}, inference_label_filter=False)


def check_result(r):
    if (r['rows'] != r['known_rows']+r['unknown_rows'] or r['inference_label_filter']
            or not r['protected_outputs_exact'] or set(r['policies']) != set(ARMS)):
        raise ValueError('All causal rows and protected outputs required')
    for a in ARMS:
        check_policy(r['policies'][a])
        for rec in r['recordings'].values():
            check_policy(rec[a])
        for key in ('known_rows', 'unknown_rows', 'selected_count', 'selected_unknown',
                    'selected_known_benefit_mass', 'selected_known_harm_mass', 'selected_known_reference_mass',
                    'selected_known_easy_harm_mass', 'selected_known_easy_reference_mass',
                    'selected_known_easy_benefit_mass', 'full_known_easy_reference_mass', 'selected_unknown_envelope_mass'):
            equal_number(r['policies'][a][key], sum(v[a][key] for v in r['recordings'].values()))
        fixed = r['fixed_anchor_cohort'][a]
        if (fixed['known_selected'] != r['fixed_anchor_cohort']['anchor']['known_selected']
                or fixed['observed_moments'] != r['fixed_anchor_cohort']['anchor']['observed_moments']):
            raise ValueError('Fixed-cohort diagnosis cannot change the selected cohort')
    for left, right in PAIRS:
        key = right+'_minus_'+left
        a, b = r['scores'][left], r['scores'][right]
        equal_number(r['contrasts'][key+'_MSE'], b-a if a is not None and b is not None else None)
        for p in r['matched'][key].values():
            check_policy(p)
        if r['matched'][key][left]['selected_count'] != r['matched'][key][right]['selected_count']:
            raise ValueError('Same-count comparison must have equal intervention counts')
        for mode in ('full', 'matched'):
            pair = r['pairs'][key+'_'+mode]
            equal_number(pair['lower_mass'], pair['known_difference_mass']-pair['unknown_exchanged_envelope_mass'])
            equal_number(pair['upper_mass'], pair['known_difference_mass']+pair['unknown_exchanged_envelope_mass'])
            den = r['full_known_reference_mass']
            equal_number(r['contrasts'][key+'_'+mode], 100*pair['lower_mass']/den if den > 0 else None)


def summarize(rows, reg, cfg):
    expected = {(scope, context, i['packet']['identity']['source'], seed)
        for scope in ('fit', 'held') for context, items in reg['contexts'].items() for i in items for seed in reg['seeds']}
    actual = [(r['scope'], r['context'], r['source'], r['seed']) for r in rows]
    if len(rows) != 144 or len(set(actual)) != 144 or set(actual) != expected:
        raise ValueError('All144 fixed source/seed/scope views required')
    scopes = {}
    for scope in ('fit', 'held'):
        group = [r for r in rows if r['scope'] == scope]
        intervals = {key: metrics.previous.interval(group, key, cfg['bootstrap_draws'], cfg['bootstrap_seed'])
                     for key in group[0]['result']['contrasts']}
        for interval in intervals.values():
            if interval['CI95'] is not None:
                interval['status'] = 'nominal_exposed_TRAIN_internal_not_confirmation'
        safety = {}
        for arm in ARMS:
            policies = [r['result']['policies'][arm] for r in group]
            safety[arm] = dict(views=72, selected_occurrences=sum(p['selected_count'] for p in policies),
                known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk'] > .02+1e-12 for p in policies),
                easy_upper_violations=sum(p['easy_selected_risk_upper'] is not None and p['easy_selected_risk_upper'] > .02+1e-12 for p in policies),
                net_easy_upper_violations=sum(p['easy_degradation_upper'] is not None and p['easy_degradation_upper'] > .02+1e-12 for p in policies),
                undefined_easy=sum(p['easy_selected_risk_upper'] is None for p in policies),
                empty_actions=sum(p['selected_count'] == 0 for p in policies),
                full_completion_support=sum(p['finite_completion_supported'] for p in policies))
        scopes[scope] = dict(contrasts=intervals, safety=safety,
            rows_per_arm=sum(r['result']['rows'] for r in group), unknown_rows_per_arm=sum(r['result']['unknown_rows'] for r in group))
    failures = []
    for a in COMPARATORS:
        for mode in ('full', 'matched'):
            ci = scopes['held']['contrasts']['isolated_risk_minus_'+a+'_'+mode]['CI95']
            if ci is None or ci[0] <= 0:
                failures.append(mode+'_utility_vs_'+a+'_not_supported')
    if scopes['held']['safety']['isolated_risk']['full_completion_support'] != 72:
        failures.append('full_policy_risk_support_failed')
    if not all(r['result']['protected_outputs_exact'] for r in rows):
        failures.append('benefit_reference_protection_failed')
    return dict(scopes=scopes, views=144, fresh_fits=54, bootstrap_draws=cfg['bootstrap_draws'],
        eligible_for_later_development_design=not failures, failures=failures,
        outer_development_read=False, independent_roles_read=False, deployment_changed=False,
        new_forecaster=False, scientific_success=False)


def models(reg):
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    if (freeze['fresh_fits'] != 54 or freeze['optimizer_steps'] != 108000
            or freeze['registration_sha256'] != run.base.sha(run.REG) or not freeze['matched_sampling']
            or any(freeze[k] for k in ('inner_held_scored', 'outer_development_read', 'independent_roles_read', 'deployment_changed'))):
        raise ValueError('All54 finals must be frozen before readout')
    out = {}
    for ref in freeze['fits']:
        path = run.ROOT/ref['path']
        if (path.is_symlink() or not path.resolve().is_relative_to((run.PUBLIC/'fits').resolve())
                or run.base.sha(path) != ref['sha256']):
            raise ValueError('Changed fit receipt')
        doc = json.loads(path.read_text())
        i = doc['identity']
        key = i['context'], i['seed'], doc['arm']
        p = run.ROOT/doc['checkpoint']['path']
        if (key in out or p.is_symlink() or not p.resolve().is_relative_to((run.PRIVATE/'heads').resolve())
                or run.base.sha(p) != doc['checkpoint']['sha256'] or doc['step'] != 2000
                or doc['inner_held_scored'] or doc['independent_roles_read'] or doc['new_forecaster']
                or doc['registration_sha256'] != run.base.sha(run.REG)):
            raise ValueError('Changed model identity or boundary')
        expected_partition = {j['packet']['identity']['source']: j['partition'] for j in reg['contexts'][i['context']]}
        if i['partitions'] != expected_partition:
            raise ValueError('Changed training partition')
        out[key] = doc
    expected = {(c, seed, a) for c in reg['contexts'] for seed in reg['seeds'] for a in api.ARMS}
    if set(out) != expected:
        raise ValueError('Complete registered fit roster required')
    return freeze, out


def execute(resume=False, check=False):
    reg, cfg, ancestor = run.admission()
    freeze, documents = models(reg)
    _, previous = run.ancestor_eval.models()
    old_local = run.parent.previous_eval.frozen_models()
    binding = dict(registration_sha256=run.base.sha(run.REG), training_freeze_sha256=run.base.sha(run.PUBLIC/'training_freeze.json'))
    if check:
        complete = json.loads((PUBLIC/'complete.json').read_text())
        rows = []
        for ref in complete['groups']:
            path = run.ROOT/ref['path']
            if path.is_symlink() or not path.resolve().is_relative_to((PUBLIC/'groups').resolve()) or run.base.sha(path) != ref['sha256']:
                raise ValueError('Changed group receipt')
            row = json.loads(path.read_text())
            partition = next(i['partition'] for i in reg['contexts'][row['context']] if i['packet']['identity']['source'] == row['source'])
            if (any(row[k] != v for k, v in binding.items()) or not row['exact_replay'] or not row['prior_controls_reproduced']
                    or row['independent_roles_read'] or row['outer_development_read'] or row['deployment_changed']
                    or set(row['result']['recordings']) != set(partition[row['scope']+'_recordings'])):
                raise ValueError('Changed readout provenance or partition')
            check_result(row['result'])
            rows.append(row)
        if (any(complete[k] != v for k, v in binding.items()) or complete['summary_sha256'] != run.base.sha(PUBLIC/'summary.json')
                or summarize(rows, reg, cfg) != json.loads((PUBLIC/'summary.json').read_text())):
            raise ValueError('Complete summary does not replay')
        for context in reg['contexts']:
            for seed in reg['seeds']:
                states = [api.core.read_checkpoint(run.ROOT/documents[context, seed, a]['checkpoint']['path']) for a in api.ARMS]
                api.assert_matched(states)
                anchor = api.core.read_checkpoint(run.ROOT/previous[context, seed, 'shared_utility']['checkpoint']['path'])
                for s in states:
                    api.core.exact(s['anchor_state'], anchor)
                    doc = documents[context, seed, s['arm']]
                    api.core.exact(s['identity'], doc['identity'])
                    api.core.exact(s['anchor_sha256'], previous[context, seed, 'shared_utility']['checkpoint']['sha256'])
                    if s['experiment_sha256'] != run.base.sha(run.REG) or s['step'] != 2000:
                        raise ValueError('Changed checkpoint binding')
        print('Verified54 frozen models,144 views, protected outputs, fixed cohorts and risk arithmetic')
        return
    if (PUBLIC/'complete.json').exists():
        raise ValueError('Completed readout immutable')
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    refs, rows, started = [], [], time.monotonic()

    def beat(**kw):
        r = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), phase='readout', **kw)
        tmp = run.PRIVATE/'readout_heartbeat.tmp'
        tmp.write_text(json.dumps(r)+'\n')
        os.replace(tmp, run.PRIVATE/'readout_heartbeat.json')
        print(json.dumps(r), flush=True)

    cached = {s: run.ancestor_eval.previous_rows(s) for s in ('fit', 'held')}
    cached_anchor = {}
    complete = json.loads((run.ancestor_eval.PUBLIC/'complete.json').read_text())
    for ref in complete['groups']:
        path = run.ROOT/ref['path']
        if run.base.sha(path) != ref['sha256']:
            raise ValueError('Changed anchor control receipt')
        row = json.loads(path.read_text())
        cached_anchor[row['scope'], row['context'], row['source'], row['seed']] = row['result']
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, held = run.parent.prior.packets(items, ancestor, beat)
            for seed in reg['seeds']:
                states = [api.core.read_checkpoint(run.ROOT/documents[context, seed, a]['checkpoint']['path']) for a in api.ARMS]
                api.assert_matched(states)
                anchor, old_risk = [api.core.read_checkpoint(run.ROOT/previous[context, seed, a]['checkpoint']['path'])
                                    for a in ('shared_utility', 'shared_risk')]
                for state in states:
                    api.core.exact(state['anchor_state'], anchor)
                for source in sorted(fit):
                    local = api.core.read_checkpoint(run.ROOT/old_local[context, seed, 'local', source]['checkpoint']['path'])
                    for scope, args in (('fit', fit[source]), ('held', held[source])):
                        path = PUBLIC/'groups'/(scope+'_'+context+'_'+source+'_seed'+str(seed)+'.json')
                        if path.exists() and not resume:
                            raise ValueError('Explicit readout resume required')
                        if path.exists():
                            row = json.loads(path.read_text())
                            if any(row[k] != v for k, v in binding.items()):
                                raise ValueError('Changed interrupted receipt')
                        else:
                            x, env = args[:2]
                            api.core.exact(anchor['source_preprocess'][source], args[-1])
                            p = dict(local=api.previous.old.predict(local, x, env)[0],
                                anchor=api.previous.predict(anchor, source, x, env),
                                old_risk=api.previous.predict(old_risk, source, x, env))
                            for s in states:
                                p[s['arm']] = api.predict(s, source, x, env)
                                api.np.testing.assert_array_equal(p[s['arm']], api.predict(s, source, x, env))
                            result = evaluate(args, p)
                            check_result(result)
                            old = cached[scope][context, source, seed]['result']
                            for a, b in (('local', 'local'), ('old_risk', 'shared_source')):
                                api.core.exact(result['policies'][a], old['policies'][b])
                                api.core.exact(result['scores'][a], old['scores'][b])
                            old = cached_anchor[scope, context, source, seed]
                            api.core.exact(result['policies']['anchor'], old['policies']['shared_utility'])
                            api.core.exact(result['scores']['anchor'], old['scores']['shared_utility'])
                            row = dict(**binding, context=context, source=source, seed=seed, scope=scope, result=result,
                                exact_replay=True, prior_controls_reproduced=True,
                                prediction_hashes={a: api.previous.old.parent.forest.fingerprint(v) for a, v in p.items()},
                                result_source='fresh_run_fixed_final_'+scope+'_inference', independent_roles_read=False,
                                outer_development_read=False, deployment_changed=False)
                            run.base.once(path, row)
                        rows.append(row)
                        refs.append(dict(path=str(path.relative_to(run.ROOT)), sha256=run.base.sha(path)))
                        beat(state='view_complete', views=len(rows), total=144)
        run.base.once(PUBLIC/'summary.json', summarize(rows, reg, cfg))
        run.base.once(PUBLIC/'complete.json', dict(**binding, groups=refs, seconds=time.monotonic()-started,
            summary_sha256=run.base.sha(PUBLIC/'summary.json'), independent_roles_read=False,
            outer_development_read=False, deployment_changed=False))
        beat(state='complete', views=144)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'check'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    execute(args.resume, args.action == 'check')
