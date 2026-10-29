"""Fixed-final fit/held factorial readout; all outputs remain exploratory."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import time

from scripts import run_m3w_decoder_risk_factorial as run
from src.world_model import m3w_decoder_risk_factorial as api
from src.world_model import m3w_scaled_risk_readout as metrics
from scripts.report_m3w_shared_train_risk import check_policy, equal_number

PUBLIC = run.PUBLIC/'readout'
ARMS = ('local',)+api.ARMS
PAIRS = (('shared_risk', 'source_risk'), ('shared_utility', 'source_utility'),
         ('shared_utility', 'shared_risk'), ('source_utility', 'source_risk'),
         ('local', 'source_risk'), ('local', 'source_utility'))


def evaluate(args, predictions):
    if set(predictions) != set(ARMS):
        raise ValueError('All fixed factorial and local arms required')
    x, env, y, _, sites, recs, frames, pr = args
    eligible = api.prior.eligible(args)
    actions = {arm: eligible & (api.old.core.signed(p)[:, 0] > 0)
        & (api.old.core.signed(p)[:, 1:] <= 0).all(1) for arm, p in predictions.items()}
    policies = {arm: metrics.policy_score(y, a, env) for arm, a in actions.items()}
    scores = {arm: metrics.previous.score(p, y, pr, sites, recs, frames,
        api.np.ones(len(x), bool))['conditional_MSE'] for arm, p in predictions.items()}
    known = api.np.isfinite(y).all(1)
    den = float(y[known, 2].sum())
    contrasts, pairs, matched = {}, {}, {}
    for left, right in PAIRS:
        key = right+'_minus_'+left
        contrasts[key+'_MSE'] = scores[right]-scores[left] if scores[right] is not None and scores[left] is not None else None
        a, b = metrics.previous.policy.matched(actions[left], actions[right],
            api.old.core.signed(predictions[left])[:, 0], api.old.core.signed(predictions[right])[:, 0],
            recs, frames, api.np.arange(len(x)))
        matched[key] = {left: metrics.policy_score(y, a, env), right: metrics.policy_score(y, b, env)}
        for mode, aa, bb in (('full', actions[left], actions[right]), ('matched', a, b)):
            pair = metrics.previous.paired_completion(y, aa, bb, env)
            pairs[key+'_'+mode] = pair
            contrasts[key+'_'+mode] = 100*pair['lower_mass']/den if den > 0 else None
    a = contrasts['source_risk_minus_shared_risk_MSE']
    b = contrasts['source_utility_minus_shared_utility_MSE']
    contrasts['decoder_by_risk_MSE_interaction'] = a-b if a is not None and b is not None else None
    return dict(rows=len(x), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        full_known_reference_mass=den, scores=scores, policies=policies, contrasts=contrasts,
        pairs=pairs, matched=matched,
        diagnostics={arm: metrics.selected_diagnostics(p, y, actions[arm], pr['scale']) for arm, p in predictions.items()},
        recordings={str(r): {arm: metrics.policy_score(y[recs == r], a[recs == r], env[recs == r])
                            for arm, a in actions.items()} for r in api.np.unique(recs)},
        inference_label_filter=False), actions


def check_result(result):
    if result['rows'] != result['known_rows']+result['unknown_rows'] or result['inference_label_filter']:
        raise ValueError('All causal inference rows required')
    for arm in ARMS:
        p = result['policies'][arm]
        check_policy(p)
        for rec in result['recordings'].values():
            check_policy(rec[arm])
        for key in ('known_rows', 'unknown_rows', 'selected_count', 'selected_unknown',
                    'selected_known_benefit_mass', 'selected_known_harm_mass',
                    'selected_known_reference_mass', 'selected_known_easy_harm_mass',
                    'selected_known_easy_reference_mass', 'selected_unknown_envelope_mass'):
            equal_number(p[key], sum(r[arm][key] for r in result['recordings'].values()))
    for left, right in PAIRS:
        key = right+'_minus_'+left
        a, b = result['scores'][left], result['scores'][right]
        equal_number(result['contrasts'][key+'_MSE'], b-a if a is not None and b is not None else None)
        for p in result['matched'][key].values():
            check_policy(p)
        if result['matched'][key][left]['selected_count'] != result['matched'][key][right]['selected_count']:
            raise ValueError('Same-count contrast changed counts')
        for mode in ('full', 'matched'):
            pair = result['pairs'][key+'_'+mode]
            equal_number(pair['lower_mass'], pair['known_difference_mass']-pair['unknown_exchanged_envelope_mass'])
            equal_number(pair['upper_mass'], pair['known_difference_mass']+pair['unknown_exchanged_envelope_mass'])
            den = result['full_known_reference_mass']
            equal_number(result['contrasts'][key+'_'+mode], 100*pair['lower_mass']/den if den > 0 else None)


def summarize(rows, reg, cfg):
    expected = {(scope, context, i['packet']['identity']['source'], seed)
        for scope in ('fit', 'held') for context, items in reg['contexts'].items()
        for i in items for seed in reg['seeds']}
    actual = [(r['scope'], r['context'], r['source'], r['seed']) for r in rows]
    if len(rows) != 144 or len(set(actual)) != 144 or set(actual) != expected:
        raise ValueError('Every fixed144 scope/context/source/seed view required')
    scopes = {}
    for scope in ('fit', 'held'):
        group = [r for r in rows if r['scope'] == scope]
        intervals = {k: metrics.previous.interval(group, k, cfg['bootstrap_draws'], cfg['bootstrap_seed'])
                     for k in group[0]['result']['contrasts']}
        for value in intervals.values():
            if value['CI95'] is not None:
                value['status'] = 'nominal_exposed_TRAIN_internal_not_confirmation'
        safety = {}
        for arm in ARMS:
            policies = [r['result']['policies'][arm] for r in group]
            safety[arm] = dict(views=72, selected_occurrences=sum(p['selected_count'] for p in policies),
                known_easy_violations=sum(p['known_easy_positive_risk'] is not None
                    and p['known_easy_positive_risk'] > .02+1e-12 for p in policies),
                easy_upper_violations=sum(p['easy_selected_risk_upper'] is not None
                    and p['easy_selected_risk_upper'] > .02+1e-12 for p in policies),
                undefined_easy=sum(p['easy_selected_risk_upper'] is None for p in policies),
                full_completion_support=sum(p['finite_completion_supported'] for p in policies))
        scopes[scope] = dict(contrasts=intervals, safety=safety,
            rows_per_arm=sum(r['result']['rows'] for r in group),
            unknown_rows_per_arm=sum(r['result']['unknown_rows'] for r in group))
    routing = {}
    for arm in ('source_risk', 'source_utility'):
        control = arm.replace('source_', 'shared_')
        failures = []
        for scope in ('fit', 'held'):
            value = scopes[scope]['contrasts'][arm+'_minus_'+control+'_MSE']
            if value['CI95'] is None or value['CI95'][1] >= 0:
                failures.append(scope+'_decoder_prediction_lift_not_supported')
        for mode in ('full', 'matched'):
            for compare in (control, 'local'):
                value = scopes['held']['contrasts'][arm+'_minus_'+compare+'_'+mode]
                if value['CI95'] is None or value['CI95'][0] <= 0:
                    failures.append(mode+'_utility_vs_'+compare+'_not_supported')
        if scopes['held']['safety'][arm]['full_completion_support'] != 72:
            failures.append('full_policy_risk_support_failed')
        routing[arm] = dict(eligible_for_later_development_design=not failures, failures=failures)
    return dict(scopes=scopes, routing=routing, views=144, fits=72, exact_prior_control_fits=18,
        bootstrap_draws=cfg['bootstrap_draws'], cost_head_seeds_not_forecaster_seeds=True,
        outer_development_read=False, independent_roles_read=False,
        deployment_changed=False, scientific_success=False)


def models():
    freeze = json.loads((run.PUBLIC/'training_freeze.json').read_text())
    if freeze['fresh_fits'] != 72 or freeze['registration_sha256'] != run.base.sha(run.REG):
        raise ValueError('All72 finals must be frozen before any readout')
    result = {}
    for ref in freeze['fits']:
        path = run.ROOT/ref['path']
        if run.base.sha(path) != ref['sha256']:
            raise ValueError('Changed fit receipt')
        doc = json.loads(path.read_text())
        i = doc['identity']
        key = i['context'], i['seed'], doc['arm']
        checkpoint = run.ROOT/doc['checkpoint']['path']
        if (checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to((run.PRIVATE/'heads').resolve())
                or run.base.sha(checkpoint) != doc['checkpoint']['sha256'] or key in result
                or doc['step'] != 2000 or doc['inner_held_scored'] or doc['independent_roles_read']
                or doc['registration_sha256'] != run.base.sha(run.REG)):
            raise ValueError('Changed frozen model or evidence boundary')
        result[key] = doc
    if len(result) != 72:
        raise ValueError('Complete factorial required')
    return freeze, result


def previous_rows(scope):
    root = run.prior.PUBLIC/('readout' if scope == 'held' else 'fit_census')
    complete = json.loads((root/'complete.json').read_text())
    result = {}
    for ref in complete['groups']:
        path = run.ROOT/ref['path']
        if run.base.sha(path) != ref['sha256']:
            raise ValueError('Changed previous aggregate')
        row = json.loads(path.read_text())
        result[row['context'], row['source'], row['seed']] = row
    return result


def execute(resume=False, check=False):
    reg, cfg, ancestor = run.admission()
    freeze, documents = models()
    old_models = run.previous_eval.frozen_models()
    previous = {scope: previous_rows(scope) for scope in ('fit', 'held')}
    binding = dict(registration_sha256=run.base.sha(run.REG),
                   training_freeze_sha256=run.base.sha(run.PUBLIC/'training_freeze.json'))
    if check:
        complete = json.loads((PUBLIC/'complete.json').read_text())
        rows = []
        for ref in complete['groups']:
            path = run.ROOT/ref['path']
            if (path.is_symlink() or not path.resolve().is_relative_to((PUBLIC/'groups').resolve())
                    or run.base.sha(path) != ref['sha256']):
                raise ValueError('Changed readout group')
            row = json.loads(path.read_text())
            if any(row[k] != v for k, v in binding.items()) or not row['exact_replay'] or not row['prior_controls_reproduced']:
                raise ValueError('Changed readout provenance')
            check_result(row['result'])
            rows.append(row)
        if (complete['summary_sha256'] != run.base.sha(PUBLIC/'summary.json')
                or any(complete[k] != v for k, v in binding.items())
                or summarize(rows, reg, cfg) != json.loads((PUBLIC/'summary.json').read_text())):
            raise ValueError('Changed complete summary')
        print('Checked144 fixed fit/held views,72 weight hashes and aggregate arithmetic')
        return
    if (PUBLIC/'complete.json').exists():
        raise ValueError('Completed readout immutable')
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    rows, refs, started = [], [], time.monotonic()
    def beat(**kw):
        row = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), phase='factorial_readout', **kw)
        temp = run.PRIVATE/'readout_heartbeat.tmp'
        temp.write_text(json.dumps(row)+'\n')
        os.replace(temp, run.PRIVATE/'readout_heartbeat.json')
        print(json.dumps(row), flush=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for context, items in sorted(reg['contexts'].items()):
            fit, held = run.prior.packets(items, ancestor, beat)
            for seed in reg['seeds']:
                states = [api.old.core.read_checkpoint(run.ROOT/documents[context, seed, arm]['checkpoint']['path'])
                          for arm in api.ARMS]
                api.assert_matched(states)
                old_state = api.old.core.read_checkpoint(run.ROOT/old_models[context, seed, 'shared_source', None]['checkpoint']['path'])
                api.assert_previous_control(states[0], old_state)
                for source in sorted(fit):
                    local = api.old.core.read_checkpoint(run.ROOT/old_models[context, seed, 'local', source]['checkpoint']['path'])
                    for scope, args in (('fit', fit[source]), ('held', held[source])):
                        path = PUBLIC/'groups'/(scope+'_'+context+'_'+source+'_seed'+str(seed)+'.json')
                        if path.exists() and not resume:
                            raise ValueError('Explicit readout resume required')
                        if path.exists():
                            row = json.loads(path.read_text())
                            if any(row[k] != v for k, v in binding.items()):
                                raise ValueError('Changed receipt binding')
                        else:
                            api.old.core.exact(local['preprocess'], args[-1])
                            predictions = {'local': api.old.predict(local, args[0], args[1])[0]}
                            for state in states:
                                api.old.core.exact(state['source_preprocess'][source], args[-1])
                                p = api.predict(state, source, args[0], args[1])
                                api.np.testing.assert_array_equal(p, api.predict(state, source, args[0], args[1]))
                                predictions[state['arm']] = p
                            result, actions = evaluate(args, predictions)
                            check_result(result)
                            cached = previous[scope][context, source, seed]['result']
                            for new, old in (('local', 'local'), ('shared_risk', 'shared_source')):
                                api.old.core.exact(result['policies'][new], cached['policies'][old])
                                api.old.core.exact(result['scores'][new], cached['scores'][old])
                            row = dict(**binding, scope=scope, context=context, source=source, seed=seed, result=result,
                                prediction_hashes={k: api.old.parent.forest.fingerprint(p) for k, p in predictions.items()},
                                exact_replay=True, prior_controls_reproduced=True,
                                result_source='fresh_run_fixed_final_'+scope+'_inference',
                                independent_roles_read=False, outer_development_read=False, deployment_changed=False)
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
    execute(resume=args.resume, check=args.action == 'check')
