"""Freeze causal actions before the registered risk-priority development readout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_easy_hurdle as original
from scripts.manage_m3w_easy_risk_priority import PUBLIC, PRIVATE, CONFIG, registration, digest, immutable
from scripts.export_m3w_easy_hurdle_create import closure
from src.world_model import m3w_easy_risk_priority as repair

np, api = repair.np, repair.api
parent, base = original.parent, original.base
POLICIES = ['floor', 'common_anchor', *[arm+'_'+policy for arm in ('raw', *repair.ARMS)
                                      for policy in ('independent', 'joint', 'matched')]]
CONTRASTS = [['risk_priority_matched', 'uncapped_matched'],
             *[[arm+'_matched', 'raw_matched'] for arm in repair.ARMS],
             *[[arm+'_joint', 'raw_joint'] for arm in repair.ARMS]]


def identity():
    assert registration() == json.loads((PUBLIC/'registration.json').read_text())
    paths = closure(ROOT, ['scripts.run_m3w_easy_risk_priority_policy', 'scripts.restore_m3w_easy_risk_priority_heads'])
    paths += [ROOT/'tests/test_m3w_easy_risk_priority_policy.py', PUBLIC/'decision_protocol.md']
    return dict(training_registration=base.artifact(PUBLIC/'registration.json'),
        bindings={str(p.relative_to(ROOT)): digest(p) for p in paths},
        policies=POLICIES, contrasts=CONTRASTS, causal_keys=list(parent.CAUSAL_KEYS),
        groups=108, held_outcomes_used_for_actions=False, independent_roles_read=False,
        formal_primary_replaced=False, new_policy_hyperparameters=False)


def causal_view(data):
    # Future arrays in the archive are deliberately unavailable to all prediction code.
    return {k: data[k] for k in parent.CAUSAL_KEYS}


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'policy_heartbeat.json', row)
    with (PRIVATE/'policy_events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def reserve():
    if shutil.disk_usage(PRIVATE).free < 10*2**30:
        raise OSError('Keep10GiB and all completed artifacts; no scope reduction')


def checked(ref):
    assert base.artifact(ROOT/ref['path']) == ref
    return json.loads((ROOT/ref['path']).read_text())


def load():
    ident = identity(); cfg = json.loads(CONFIG.read_text())
    assert json.loads((PUBLIC/'decision_registration.json').read_text()) == ident
    for p in (PUBLIC/'decision_registration.json', PUBLIC/'training_freeze.json'):
        base.inter.committed(p)
    trained = json.loads((PUBLIC/'training_freeze.json').read_text())
    assert trained['registration_sha256'] == digest(PUBLIC/'registration.json')
    assert trained['heads'] == 216 and not trained['held_outcomes_used']
    _, _, data, jobs, oid, pid, _, actions = original.load()
    fits = {Path(ref['path']).parent.name: ref for ref in trained['groups']}
    old = json.loads((original.PUBLIC/'decision_freeze.json').read_text())
    old_easy_actions = {Path(ref['path']).stem: ref for ref in old['groups']}
    assert len(fits) == len(old_easy_actions) == 108
    return cfg, ident, data, jobs, oid, pid, fits, actions, old_easy_actions


def action_group(c, causal, pair, fit_ref, parent_action_ref, old_easy_ref, pid, cfg):
    if set(causal) != set(parent.CAUSAL_KEYS):
        raise ValueError('Only whitelisted causal context keys accepted')
    new, old, old_easy = checked(fit_ref), checked(parent_action_ref), checked(old_easy_ref)
    source_identity = new['identity']['source_identity']
    assert source_identity['parent_fit'] == old['fit']
    held, scores, pr = parent.predictions(c, causal, old['fit'], pid)
    ids = c['ids'][held]
    assert base.artifact(ROOT/old['arrays']['path']) == old['arrays']
    with np.load(ROOT/old['arrays']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids'])
        eligible, original_raw_joint = z['eligible'].copy(), z['subset_aggregate_joint'].copy()
    u = api.head.descriptors(causal['geometry'][ids], c['floor'][held], c['prediction'][held], c['x'][held], pr)
    risks = dict(raw=parent.api.head.parent.signed(scores['subset_aggregate'].astype(float))/pr['cost_scale'])
    arrays = dict(ids=ids, eligible=eligible, raw_scores=risks['raw']); states = {}
    for arm, ref in new['artifacts'].items():
        assert base.artifact(ROOT/ref['path']) == ref
        state = api.head.read_checkpoint(ROOT/ref['path']); states[arm] = state
        assert state['identity'] == new['identity'] and state['step'] == cfg['head_training']['steps']
        p = api.predict(state, c['x'][held], u, c['env'][held])
        arrays[arm+'_scores'] = p[:, :4]
        risks[arm] = np.column_stack((risks['raw'][:, 0], p[:, 3].astype(float)))
    repair.assert_matched(states['uncapped'], states['risk_priority'])
    assert base.artifact(ROOT/old_easy['arrays']['path']) == old_easy['arrays']
    with np.load(ROOT/old_easy['arrays']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids'])
        np.testing.assert_array_equal(arrays['uncapped_scores'], z['supervised_scores'])
        np.testing.assert_array_equal(eligible, z['eligible'])
    name = c['name']+f'_pair{pair}'
    ridge = json.loads((base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
    ref = ridge['artifacts']['scores']; assert base.artifact(ROOT/ref['path']) == ref
    with np.load(ROOT/ref['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids'])
        utility = (z['scores'][:, 5].astype(float)-z['scores'][:, 6].astype(float))/pr['cost_scale']
    keys = np.array([str(s)+'|'+str(r) for s, r in zip(causal['sites'][ids], causal['recordings'][ids])])
    actions, solver = repair.decisions(utility, risks, eligible, keys, causal['frames'][ids], ids, cfg['node_limit'])
    np.testing.assert_array_equal(actions['raw_joint'], original_raw_joint)
    with np.load(ROOT/old_easy['arrays']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(actions['uncapped_joint'], z['supervised_joint'])
        np.testing.assert_array_equal(actions['uncapped_independent'], z['supervised_independent'])
    arrays.update(actions)
    return arrays, dict(solver=solver, parent_control_scores_exact=True, parent_own_count_actions_exact=True,
                       utility_ref=ref, future_fields_removed=True, held_outcomes_used=False)


def decide(cfg, ident, data, jobs, oid, pid, fits, actions, old_easy, *, resume=False, replay=False):
    causal = causal_view(data); refs = []; start = time.monotonic()
    for c in base.floor_api.contexts(causal, jobs, oid):
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; dest = PRIVATE/'decisions'/(name+'.json'); arr = dest.with_suffix('.npz')
            reserve()
            if dest.exists() and resume and not replay:
                record = checked(base.artifact(dest))
                assert record['identity'] == ident and record['fit'] == fits[name]
                assert record['parent_action'] == actions[name] and record['old_easy_action'] == old_easy[name]
                assert base.artifact(ROOT/record['arrays']['path']) == record['arrays']
            else:
                if dest.exists() and not replay:
                    raise ValueError('Existing action groups require resume or replay')
                arrays, meta = action_group(c, causal, pair, fits[name], actions[name], old_easy[name], pid, cfg)
                parent.previous.descriptor.write_arrays(arr, arrays, replay)
                record = dict(identity=ident, fit=fits[name], parent_action=actions[name], old_easy_action=old_easy[name],
                              arrays=base.artifact(arr), **meta)
                immutable(dest, record)
            refs.append(base.artifact(dest)); beat('actions_replayed' if replay else 'actions_frozen', group=name, groups=len(refs))
    assert len(refs) == 108
    immutable(PUBLIC/'decision_freeze.json', dict(identity=ident, groups=refs, held_outcomes_used=False))
    immutable(PUBLIC/('prediction_replay.json' if replay else 'decision_runtime.json'),
        dict(groups=108, exact=replay, seconds=time.monotonic()-start, pid=os.getpid(),
             peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def reduce_metrics(rows, sites, cfg):
    return {p: {k: base.inter.paired_localities([r for r in rows if r['policy'] == p], sites, k,
        cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for k in rows[0]['metric']}
        for p in sorted({r['policy'] for r in rows})}


def gates_for(paired, rows, contrasts):
    primary = paired['risk_priority_matched_vs_uncapped_matched']
    selected = [r['metric'] for r in rows if r['policy'] == 'risk_priority_matched']
    assert selected, 'No selected-policy readout'
    positive = lambda k: primary[k]['ci95'] is not None and primary[k]['ci95'][0] > 0
    matched = [r for r in contrasts if '_matched_vs_' in r['policy']]
    assert matched, 'Missing matched-count contrasts'
    gates = dict(matched_count=all(r['metric']['intervention_difference_pp'] == 0 for r in matched),
        matched_ADE_advantage=positive('ADE_gain_percent'),
        matched_all_reference_harm_reduction=positive('all_reference_harm_reduction_pp'),
        every_view_defined_risk_within_2percent=all(m['selected_positive_harm_ratio'] is not None and m['selected_positive_harm_ratio'] <= .02 for m in selected),
        every_view_easy_preserved=all(m['easy_gain_CV'] is not None and m['easy_gain_CV'] >= -2 for m in selected),
        no_zero_CV_harm=all(m['zero_CV_harmed'] == 0 for m in selected))
    gates['exploratory_screen_pass'] = all(gates.values())
    gates.update(independent_confirmation=False, formal_primary_replaced=False, calibration_certificate=False,
                 deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    return gates


def evaluate(cfg, ident, data, jobs, oid, pid, fits, actions, old_easy, *, replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json')
    frozen = json.loads((PUBLIC/'decision_freeze.json').read_text()); assert frozen['identity'] == ident
    refs = {Path(ref['path']).stem: ref for ref in frozen['groups']}
    rows, qualities, losses = [], [], []; start = time.monotonic()
    for c in base.floor_api.contexts(causal_view(data), jobs, oid):
        cv, cf, (floor, ff), (neural, nf) = base.floor_api.costs(c, data, np.arange(len(c['ids'])))
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; doc = checked(refs[name]); fit = checked(doc['fit'])
            assert base.artifact(ROOT/doc['arrays']['path']) == doc['arrays']
            with np.load(ROOT/doc['arrays']['path'], allow_pickle=False) as z:
                a = {k: z[k].copy() for k in z.files}
            held = np.flatnonzero(np.isin(data['sites'][c['ids']], c['pairs'][pair][1]))
            np.testing.assert_array_equal(a['ids'], c['ids'][held])
            scales = []
            for arm, ref in fit['artifacts'].items():
                assert base.artifact(ROOT/ref['path']) == ref
                state = api.head.read_checkpoint(ROOT/ref['path']); scales.append(state['norm']['cost_scale'])
                losses.append(dict(group=name, arm=arm, step=state['step'], first=state['trace'][0], last=state['trace'][-1]))
            assert scales[0] == scales[1]
            for site in c['pairs'][pair][1]:
                at = data['sites'][a['ids']] == site; ix = held[at]
                common = dict(group=name, site=site, seed=c['job']['old_identity']['seed'])
                for policy in POLICIES:
                    take = np.zeros(len(ix), bool) if policy == 'floor' else a[policy][at]
                    m = base.floor_api.metric(cv[ix], floor[ix], neural[ix], cf[ix], ff[ix], nf[ix],
                        np.where(take, neural[ix], floor[ix]), np.where(take, nf[ix], ff[ix]), take,
                        data['valid'][c['ids'][ix]], c['job']['design']['easy_cut'], c['job']['design']['hard_cut'])
                    m.update(parent.harm_accounting(floor[ix], neural[ix], np.isfinite(cv[ix]), take))
                    rows.append(dict(**common, policy=policy, metric=m))
                truth = api.targets(cv[ix], floor[ix], neural[ix], c['job']['design']['easy_cut'], scales[0])
                for arm in repair.ARMS:
                    q = api.quality(a[arm+'_scores'][at], truth, data['recordings'][c['ids'][ix]], data['frames'][c['ids'][ix]])
                    qualities.append(dict(**common, policy=arm, metric=q))
        beat('held_development_scored', group=c['name'])
    sites = sorted(set(data['sites'])); index = {(r['group'], r['site'], r['policy']): r['metric'] for r in rows}
    contrasts = []
    for new, old in CONTRASTS:
        for row in (r for r in rows if r['policy'] == new):
            contrasts.append(dict(group=row['group'], site=row['site'], seed=row['seed'], policy=new+'_vs_'+old,
                metric=original.previous.contrast(row['metric'], index[row['group'], row['site'], old])))
    summary = reduce_metrics(rows, sites, cfg); paired = reduce_metrics(contrasts, sites, cfg)
    gates = gates_for(paired, rows, contrasts)
    worst = {p: dict(risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and
        r['metric']['selected_positive_harm_ratio'] > .02 for r in rows if r['policy'] == p),
        undefined_selected_risk_views=sum(r['metric']['selected_positive_harm_ratio'] is None for r in rows if r['policy'] == p),
        abstaining_views=sum(r['metric']['intervention_rate'] == 0 for r in rows if r['policy'] == p),
        worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy'] == p and r['metric']['easy_gain_CV'] is not None)) for p in POLICIES}
    old = json.loads((original.PUBLIC/'summary.json').read_text())
    for p, key in [('floor','floor'), ('raw_joint','raw_joint'), ('raw_independent','raw_independent'),
                   ('uncapped_joint','supervised_joint'), ('uncapped_independent','supervised_independent')]:
        assert summary[p] == old['summary'][key], p
    out = dict(identity=ident, result_source='fresh_run_actions_and_development_readout',
        training='fresh_run_CREATE', inputs_and_frozen_forecasts='cached_verified', independent_confirmation='not_run',
        summary=summary, paired=paired, quality=reduce_metrics(qualities, sites, cfg), worst_views=worst, gates=gates,
        by_seed={str(s): reduce_metrics([r for r in rows if r['seed'] == s], sites, cfg) for s in (17,29,43)})
    immutable(PRIVATE/'details.json', dict(rows=rows, qualities=qualities, contrasts=contrasts, losses=losses))
    immutable(PUBLIC/'summary.json', out); immutable(PUBLIC/'gates.json', gates)
    immutable(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),
        dict(summary=base.artifact(PUBLIC/'summary.json'), details=base.artifact(PRIVATE/'details.json'),
             exact=replay, seconds=time.monotonic()-start, pid=os.getpid(),
             peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register','decide','evaluate','replay_decide','replay_evaluate'])
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    if a.phase == 'register':
        ident = identity(); immutable(PUBLIC/'decision_registration.json', ident)
        print(json.dumps(dict(registered=True, bindings=len(ident['bindings']), held_readout='not_run'))); return
    repair.torch.set_num_threads(4); repair.torch.set_num_interop_threads(1)
    with (PRIVATE/'policy.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reserve(); beat('started', phase=a.phase); values = load()
        if a.phase in ('decide', 'replay_decide'):
            decide(*values, resume=a.resume, replay=a.phase == 'replay_decide')
        else:
            evaluate(*values, replay=a.phase == 'replay_evaluate')
        beat('complete', phase=a.phase)


if __name__ == '__main__':
    main()
