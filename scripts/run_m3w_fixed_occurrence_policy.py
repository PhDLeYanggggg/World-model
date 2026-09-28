"""Freeze paired causal decisions before a fixed-occurrence development readout."""
import argparse
import fcntl
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import train_m3w_fixed_occurrence as train
from scripts import run_m3w_easy_risk_priority_policy as previous
from src.world_model import m3w_easy_risk_priority as allocator

api, old = train.api, train.old
parent, base = old.parent, old.base
PUBLIC, PRIVATE = train.PUBLIC, train.PRIVATE
POLICIES = ['floor', 'common_anchor', *[a+'_'+p for a in ('raw', *api.ARMS)
                                     for p in ('independent', 'joint', 'matched')]]
CONTRASTS = [['fixed_matched', 'trainable_matched'],
             *[[a+'_matched', 'raw_matched'] for a in api.ARMS],
             *[[a+'_joint', 'raw_joint'] for a in api.ARMS]]


def identity():
    cfg, reg = train.registration()
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    paths = train.closure(ROOT, ['scripts.run_m3w_fixed_occurrence_policy'])
    paths += [ROOT/'tests/test_m3w_fixed_occurrence_policy.py', PUBLIC/'decision_protocol.md']
    return cfg, dict(training_registration=base.artifact(PUBLIC/'registration.json'),
        bindings={str(p.relative_to(ROOT)): train.digest(p) for p in paths}, policies=POLICIES,
        contrasts=CONTRASTS, causal_keys=list(parent.CAUSAL_KEYS), groups=108,
        held_outcomes_used_for_actions=False, independent_roles_read=False, formal_primary_replaced=False)


def causal_view(data):
    return {k: data[k] for k in parent.CAUSAL_KEYS}


def decisions(utility, risks, eligible, recordings, frames, ids, node_limit=256):
    if set(risks) != {'raw', *api.ARMS}: raise ValueError('Raw control and both paired arms required')
    mapping = dict(raw='raw', trainable='uncapped', fixed='risk_priority')
    result, stats = allocator.decisions(utility, {mapping[a]: q for a, q in risks.items()},
                                        eligible, recordings, frames, ids, node_limit)
    out = dict(common_anchor=result['common_anchor']); renamed = {}
    for arm, key in mapping.items():
        for policy in ('independent', 'joint', 'matched'):
            out[arm+'_'+policy] = result[key+'_'+policy]
            if key+'_'+policy in stats: renamed[arm+'_'+policy] = stats[key+'_'+policy]
    return out, renamed


def checked(ref):
    assert base.artifact(ROOT/ref['path']) == ref
    return json.loads((ROOT/ref['path']).read_text())


def load():
    cfg, ident = identity(); assert ident == json.loads((PUBLIC/'decision_registration.json').read_text())
    for p in (PUBLIC/'decision_registration.json', PUBLIC/'training_freeze.json', PUBLIC/'fit_replay.json'):
        base.inter.committed(p)
    trained = json.loads((PUBLIC/'training_freeze.json').read_text())
    assert trained['registration_sha256'] == train.digest(PUBLIC/'registration.json')
    assert trained['heads'] == 216 and not trained['held_outcomes_used']
    assert json.loads((PUBLIC/'fit_replay.json').read_text())['exact_except_elapsed']
    _, _, data, jobs, oid, pid, _, actions = old.load()
    fits = {Path(r['path']).parent.name: r for r in trained['groups']}; assert len(fits) == 108
    return cfg, ident, data, jobs, oid, pid, fits, actions


def action_group(c, causal, pair, fit_ref, parent_ref, pid, cfg):
    if set(causal) != set(parent.CAUSAL_KEYS): raise ValueError('Causal whitelist only')
    fit, anchor = checked(fit_ref), checked(parent_ref)
    assert fit['identity']['source_identity']['parent_fit'] == anchor['fit']
    held, scores, pr = parent.predictions(c, causal, anchor['fit'], pid)
    ids = c['ids'][held]; ref = anchor['arrays']; assert base.artifact(ROOT/ref['path']) == ref
    with np.load(ROOT/ref['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        eligible, old_raw_joint = z['eligible'].copy(), z['subset_aggregate_joint'].copy()
    u = old.api.head.descriptors(causal['geometry'][ids], c['floor'][held], c['prediction'][held], c['x'][held], pr)
    risks = dict(raw=parent.api.head.parent.signed(scores['subset_aggregate'].astype(float))/pr['cost_scale'])
    arrays = dict(ids=ids, eligible=eligible, raw_scores=risks['raw']); states = {}
    for arm, ref in fit['artifacts'].items():
        assert base.artifact(ROOT/ref['path']) == ref
        state = old.api.head.read_checkpoint(ROOT/ref['path']); states[arm] = state
        assert state['identity'] == fit['identity'] and state['step'] == cfg['head_training']['steps']
        z, env = old.api.inputs(c['x'][held], u, c['env'][held], state['warm_start']['norm'])
        p = api.predict(state, z.numpy(), env.numpy()); arrays[arm+'_scores'] = p[:, :4]
        risks[arm] = np.column_stack((risks['raw'][:, 0], p[:, 3].astype(float)))
        if arm == 'fixed':
            teacher = old.api.predict(state['warm_start'], c['x'][held], u, c['env'][held])
            np.testing.assert_array_equal(p[:, 0], teacher[:, 0])
    api.assert_matched(states['trainable'], states['fixed'])
    name = c['name']+f'_pair{pair}'
    ridge = json.loads((base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
    ref = ridge['artifacts']['scores']; assert base.artifact(ROOT/ref['path']) == ref
    with np.load(ROOT/ref['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids'])
        utility = (z['scores'][:, 5].astype(float)-z['scores'][:, 6].astype(float))/pr['cost_scale']
    keys = np.array([str(s)+'|'+str(r) for s, r in zip(causal['sites'][ids], causal['recordings'][ids])])
    actions, stats = decisions(utility, risks, eligible, keys, causal['frames'][ids], ids, cfg['node_limit'])
    np.testing.assert_array_equal(actions['raw_joint'], old_raw_joint); arrays.update(actions)
    return arrays, dict(solver=stats, raw_own_count_action_exact=True,
        frozen_occurrence_predictions_exact=True, utility_ref=ref,
        future_fields_removed=True, held_outcomes_used=False)


def decide(cfg, ident, data, jobs, oid, pid, fits, actions, *, resume=False, replay=False):
    start = time.monotonic(); refs = []; causal = causal_view(data)
    for c in base.floor_api.contexts(causal, jobs, oid):
        for pair in range(6):
            train.guard(); name = c['name']+f'_pair{pair}'; dest = PRIVATE/'decisions'/(name+'.json'); arr = dest.with_suffix('.npz')
            if dest.exists() and resume and not replay:
                doc = json.loads(dest.read_text()); assert doc['identity'] == ident
                assert doc['fit'] == fits[name] and doc['parent_action'] == actions[name]
                assert base.artifact(ROOT/doc['arrays']['path']) == doc['arrays']
            else:
                if dest.exists() and not replay: raise ValueError('Existing decisions require resume')
                arrays, meta = action_group(c, causal, pair, fits[name], actions[name], pid, cfg)
                parent.previous.descriptor.write_arrays(arr, arrays, replay)
                doc = dict(identity=ident, fit=fits[name], parent_action=actions[name], arrays=base.artifact(arr), **meta)
                train.immutable(dest, doc)
            refs.append(base.artifact(dest)); train.beat('actions_replayed' if replay else 'actions_frozen', groups=len(refs), group=name)
    assert len(refs) == 108
    train.immutable(PUBLIC/'decision_freeze.json', dict(identity=ident, groups=refs, held_outcomes_used=False))
    train.immutable(PUBLIC/('prediction_replay.json' if replay else 'decision_runtime.json'),
        dict(groups=108, exact=replay, seconds=time.monotonic()-start))


def gates_for(paired, rows, contrasts):
    # Preserve the existing risk screen exactly, including undefined ratios failing.
    alias = {'fixed_matched': 'risk_priority_matched', 'trainable_matched': 'uncapped_matched'}
    mapped = {'risk_priority_matched_vs_uncapped_matched': paired['fixed_matched_vs_trainable_matched']}
    mapped_rows = [dict(r, policy=alias.get(r['policy'], r['policy'])) for r in rows]
    return previous.gates_for(mapped, mapped_rows, contrasts)


def evaluate(cfg, ident, data, jobs, oid, pid, fits, actions, *, replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json')
    frozen = json.loads((PUBLIC/'decision_freeze.json').read_text()); assert frozen['identity'] == ident
    refs = {Path(r['path']).stem: r for r in frozen['groups']}; rows, qualities = [], []; started = time.monotonic()
    for c in base.floor_api.contexts(causal_view(data), jobs, oid):
        cv, cf, (floor, ff), (neural, nf) = base.floor_api.costs(c, data, np.arange(len(c['ids'])))
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; doc = checked(refs[name]); fit = checked(doc['fit'])
            ref = doc['arrays']; assert base.artifact(ROOT/ref['path']) == ref
            with np.load(ROOT/ref['path'], allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
            held = np.flatnonzero(np.isin(data['sites'][c['ids']], c['pairs'][pair][1]))
            np.testing.assert_array_equal(a['ids'], c['ids'][held]); scales = []
            for ref in fit['artifacts'].values():
                assert base.artifact(ROOT/ref['path']) == ref
                scales.append(old.api.head.read_checkpoint(ROOT/ref['path'])['warm_start']['norm']['cost_scale'])
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
                truth = old.api.targets(cv[ix], floor[ix], neural[ix], c['job']['design']['easy_cut'], scales[0])
                for arm in api.ARMS:
                    quality = old.api.quality(a[arm+'_scores'][at], truth, data['recordings'][c['ids'][ix]], data['frames'][c['ids'][ix]])
                    qualities.append(dict(**common, policy=arm, metric=quality))
        train.beat('held_development_scored', group=c['name'])
    sites = sorted(set(data['sites'])); index = {(r['group'], r['site'], r['policy']): r['metric'] for r in rows}
    contrasts = []
    for new, control in CONTRASTS:
        for row in (r for r in rows if r['policy'] == new):
            contrasts.append(dict(group=row['group'], site=row['site'], seed=row['seed'], policy=new+'_vs_'+control,
                metric=old.previous.contrast(row['metric'], index[row['group'], row['site'], control])))
    summary = previous.reduce_metrics(rows, sites, cfg); paired = previous.reduce_metrics(contrasts, sites, cfg)
    gates = gates_for(paired, rows, contrasts)
    prior = json.loads((old.PUBLIC/'summary.json').read_text())
    for p in ('floor', 'raw_joint', 'raw_independent'): assert summary[p] == prior['summary'][p]
    worst = {}
    for p in POLICIES:
        mm = [r['metric'] for r in rows if r['policy'] == p]
        worst[p] = dict(risk_violating_views=sum(m['selected_positive_harm_ratio'] is not None and m['selected_positive_harm_ratio'] > .02 for m in mm),
            undefined_selected_risk_views=sum(m['selected_positive_harm_ratio'] is None for m in mm),
            abstaining_views=sum(m['intervention_rate'] == 0 for m in mm),
            worst_easy_gain_CV=min(m['easy_gain_CV'] for m in mm if m['easy_gain_CV'] is not None))
    result = dict(identity=ident, result_source='fresh_run_local_Torch_training_actions_and_development_readout',
        inputs_and_frozen_forecasts='cached_verified', independent_confirmation='not_run',
        summary=summary, paired=paired, quality=previous.reduce_metrics(qualities, sites, cfg),
        worst_views=worst, gates=gates,
        by_seed={str(s): previous.reduce_metrics([r for r in rows if r['seed'] == s], sites, cfg) for s in (17, 29, 43)})
    train.immutable(PRIVATE/'details.json', dict(rows=rows, qualities=qualities, contrasts=contrasts))
    train.immutable(PUBLIC/'summary.json', result); train.immutable(PUBLIC/'gates.json', gates)
    train.immutable(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),
        dict(exact=replay, seconds=time.monotonic()-started,
             summary=base.artifact(PUBLIC/'summary.json'), details=base.artifact(PRIVATE/'details.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'decide', 'replay_decide', 'evaluate', 'replay_evaluate'])
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    if a.phase == 'register':
        _, ident = identity(); train.immutable(PUBLIC/'decision_registration.json', ident)
        print(json.dumps(dict(registered=True, files=len(ident['bindings'])))); return
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    with (PRIVATE/'policy.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); train.guard(); values = load()
        if a.phase in ('decide', 'replay_decide'): decide(*values, resume=a.resume, replay=a.phase == 'replay_decide')
        else: evaluate(*values, replay=a.phase == 'replay_evaluate')
        train.beat('policy_phase_complete', phase=a.phase)


if __name__ == '__main__': main()
