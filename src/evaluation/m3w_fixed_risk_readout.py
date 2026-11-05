"""Read out frozen dual-head risks on the already exposed nested-C roles only."""
from scripts import run_m3w_moving_decoupled_risk as inherited

source, old, direct = inherited.source, inherited.old, inherited.direct


def evaluate(values, models, predictor, *, names, control_public, public, private, guard, beat, verify=False):
    _, _, data, jobs, refs, parts, points, routers, critics, fitted, *_ = values
    rows, receipts = [], []
    for job in jobs:
        part = parts[job['key']][2]; p = old.packet(data, job, refs, part, training=False)
        c, ids = p['causal'], p['ids']; key = job['key']+'_val2_propose1'
        old_row = source.read(control_public/'groups'/(key+'.json'))
        _, original, _ = direct.decisions(p, job, part, points, routers, critics, fitted, True)
        with source.np.load(source.checked(old_row['cache']), allow_pickle=False) as saved:
            source.np.testing.assert_array_equal(saved['ids'], ids)
            choices = {name: saved[name].copy() for name in old_row['choice_hashes']}
        for name, choice in choices.items():
            inherited.api.base.exact(source.array_hash(choice), old_row['choice_hashes'][name])
        predictions = {}
        for name in names:
            state = models[job['key'], name]
            inherited.assert_excluded(state, c, part['names'])
            risk, decision = predictor(state, c)
            predictions[name+'_risk'], predictions[name+'_decision'] = risk, decision
            for suffix, score in (('', inherited.api.combined(risk, decision)), ('_risk_only', risk)):
                policy = name+suffix
                choices[policy+'_veto'], choices[policy+'_matched'] = inherited.api.kin.parent.choose(
                    original['router'], original['all_action'], score, c)
        paths = {name: old.rollout(p['paths'], p['floor'], choice) for name, choice in choices.items()}
        # Outcomes and validity are accessed only after every model/action is frozen.
        target = data['target_eval'][ids]-data['origin'][ids, None]; valid = data['valid'][ids]
        group_rows = []
        pairs = [('main', names[0]+'_veto', names[1]+'_veto'),
            ('vs_constant', 'constant_veto', names[1]+'_veto')]
        for name in names:
            pairs.append((name+'_decision_head', name+'_risk_only_veto', name+'_veto'))
        for site in sorted(set(p['sites'])):
            at = p['sites'] == site
            policies = {name: source.readout.policy(v[at], p['floor'][at], p['cv'][at], target[at], valid[at],
                p['recordings'][at], easy_cut=job['easy_cut'], hard_cut=job['hard_cut']) for name, v in paths.items()}
            for name, policy in next(r for r in old_row['rows'] if r['site'] == site)['policies'].items():
                inherited.api.base.exact(policies[name], policy)
            contrasts = {label: source.readout.paired(paths[left][at], paths[right][at], p['floor'][at], target[at], valid[at])
                for label, left, right in pairs}
            groups = source.query_groups(p['recordings'][at], p['frames'][at], ids[at])
            matched = {name: source.matched_utility(paths[name+'_matched'][at], paths[name+'_veto'][at], p['floor'][at],
                target[at], valid[at], choices[name+'_matched'][at], choices[name+'_veto'][at], groups) for name in names}
            group_rows.append(dict(job=job['key'], site=str(site), policies=policies, contrasts=contrasts, matched=matched))
        payload = dict(ids=ids, **choices, **{a+'_prediction': v for a, v in predictions.items()})
        cache = private/'evaluation'/(key+'.npz'); cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            with source.np.load(cache, allow_pickle=False) as saved:
                for k, v in payload.items():
                    source.np.testing.assert_array_equal(saved[k], v)
        elif verify:
            raise ValueError('Original prediction cache required')
        else:
            source.np.savez_compressed(cache, **payload)
        out = public/'groups'/(key+'.json')
        source.once(out, source.stable(dict(rows=group_rows, cache=source.artifact(cache),
            prediction_hashes={a: source.array_hash(v) for a, v in predictions.items()},
            choice_hashes={a: source.array_hash(v) for a, v in choices.items()},
            old_controls='cached_verified_actions_fresh_metric_recomputation_exact')))
        rows.extend(group_rows); receipts.append(source.artifact(out)); beat(state='readout', completed=len(receipts))
        del p, predictions, paths; source.gc.collect(); guard()
    macro = inherited.macro
    policies = {name: dict(known=macro(rows, lambda r, n=name: r['policies'][n]['known_ADE_gain_floor_percent']),
        hard=macro(rows, lambda r, n=name: r['policies'][n]['subsets']['hard']['gain_percent']),
        switch_percent=macro(rows, lambda r, n=name: 100*r['policies'][n]['switch_rate']),
        safety=direct.safety(rows, name)) for name in rows[0]['policies']}
    contrasts = {name: {k: macro(rows, lambda r, n=name, k=k: r['contrasts'][n][k])
        for k in ('known_pp', 'completion_lower_pp', 'completion_upper_pp')} for name in rows[0]['contrasts']}
    return dict(held_policy=policies, contrasts=contrasts, group_receipts=receipts)
