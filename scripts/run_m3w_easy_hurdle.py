"""Source-excluded easy-risk training, causal action freeze and development readout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 Python required before Torch import')
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_centered_risk_policy as previous
from src.world_model import m3w_easy_hurdle as api
parent, base = previous.parent, previous.base
PUBLIC = previous.PUBLIC.parent/'european_easy_hurdle_v1'
PRIVATE = previous.PRIVATE.parent/'european_easy_hurdle_v1'
CONFIG = 'configs/m3w_european_easy_hurdle_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_easy_hurdle.py', 'scripts/run_m3w_easy_hurdle.py',
         'tests/test_m3w_easy_hurdle.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]
POLICIES = ['floor', 'common_anchor', *[a+'_'+p for a in ('raw', *api.ARMS)
                                      for p in ('independent', 'joint', 'matched')]]
CONTRASTS = [['supervised_matched', 'marginal_matched'],
             *[[a+'_matched', 'raw_matched'] for a in api.ARMS],
             *[[a+'_joint', 'raw_joint'] for a in api.ARMS]]


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    base.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def identity():
    cfg = json.loads((ROOT/CONFIG).read_text()); seal_path = previous.PUBLIC/'verification.json'
    assert base.digest(seal_path) == cfg['parent_seal_sha256']
    seal = json.loads(seal_path.read_text())
    for p, h in seal['source_bindings'].items():
        assert base.digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items():
        assert base.digest(previous.PUBLIC/p) == h, p
    assert cfg['arms'] == list(api.ARMS) and cfg['groups'] == 108 and cfg['new_heads'] == 216
    assert cfg['risk_budget'] == api.BUDGET and cfg['disk_reserve_bytes'] == 10*2**30
    assert not any(cfg[k] for k in ('independent_roles_read', 'threshold_search', 'deployment_changed',
        'formal_primary_replaced', 'stage5c_executed', 'smc_enabled'))
    ident = dict(parent_seal=base.artifact(seal_path), bindings={p: base.digest(ROOT/p) for p in FILES},
                 policies=POLICIES, contrasts=CONTRASTS, causal_keys=list(parent.CAUSAL_KEYS),
                 groups=108, heads=216, independent_roles_read=False, formal_primary_replaced=False)
    return cfg, ident


def load():
    cfg, ident = identity()
    assert json.loads((PUBLIC/'registration.json').read_text()) == ident
    base.inter.committed(PUBLIC/'registration.json')
    _, data, jobs, oid, _, _, _, _, _, pid = parent.load()
    fits = {Path(r['path']).parent.name: r for r in json.loads((parent.PUBLIC/'training_freeze.json').read_text())['groups']}
    actions = {Path(r['path']).stem: r for r in json.loads((parent.PUBLIC/'decision_freeze.json').read_text())['groups']}
    return cfg, ident, data, jobs, oid, pid, fits, actions


def reserve(cfg):
    if shutil.disk_usage(PRIVATE).free < cfg['disk_reserve_bytes']:
        raise OSError('Preserve10GiB and all completed artifacts; no silent scope reduction')


def checked(ref):
    assert base.artifact(ROOT/ref['path']) == ref
    return json.loads((ROOT/ref['path']).read_text())


def fitting(c, data, pair, ref, pid):
    doc = checked(ref); assert doc['identity']['experiment'] == pid
    roles = doc['identity']['roles_and_targets']
    base.floor_api.api.assert_roles(roles['producer_sites'], roles['controller_sites'], roles['training_sites'], roles['held_sites'])
    assert set(roles['training_sites']) == set(c['pairs'][pair][0])
    assert set(roles['held_sites']) == set(c['pairs'][pair][1])
    fit = np.flatnonzero(np.isin(data['sites'][c['ids']], roles['training_sites'])); ids = c['ids'][fit]
    sref = doc['artifacts']['subset_aggregate']; assert base.artifact(ROOT/sref['path']) == sref
    state = api.head.read_checkpoint(ROOT/sref['path']); pr = state['preprocess']
    u = api.head.descriptors(data['geometry'][ids], c['floor'][fit], c['prediction'][fit], c['x'][fit], pr)
    cv, _, (floor, _), (neural, _) = base.floor_api.costs(c, data, fit)
    y = api.targets(cv, floor, neural, c['job']['design']['easy_cut'], pr['cost_scale'])
    ident = dict(registration=base.artifact(PUBLIC/'registration.json'), parent_fit=ref,
                 roles=roles, seed=state['seed'], fitting_ids_hash=base.inter.array_hash(ids),
                 input_hashes={k: base.inter.array_hash(v) for k, v in
                     dict(x=c['x'][fit], descriptors=u, envelope=c['env'][fit], targets=y).items()})
    return fit, ids, u, y, pr, ident


def audit(cfg, ident, data, jobs, oid, pid, fits, actions, **_):
    rows = []
    for c in base.floor_api.contexts({k: data[k] for k in parent.CAUSAL_KEYS}, jobs, oid):
        for pair in range(6):
            name = c['name']+f'_pair{pair}'
            fit, ids, u, y, pr, fid = fitting(c, data, pair, fits[name], pid)
            known = np.isfinite(y).all(1)
            row = dict(group=name, roles=fid['roles'], rows=len(ids), known=int(known.sum()),
                       easy=int((y[:, 0] == 1).sum()), unknown=int((~known).sum()),
                       source_easy_counts={s: int(((data['sites'][ids] == s) & (y[:, 0] == 1)).sum())
                           for s in pr['training_sites']}, fitting_ids_hash=fid['fitting_ids_hash'],
                       label_hash=fid['input_hashes']['targets'], held_outcomes_used=False)
            rows.append(row)
        beat('fitting_support_audited', group=c['name'])
    assert len(rows) == 108
    out = dict(identity=ident, result_source='fresh_run_fitting_only', groups=rows,
               all_pairs_have_easy_labels=all(r['easy'] > 0 for r in rows),
               independent_roles_read=False, held_outcomes_used=False)
    base.immutable_json(PUBLIC/'fitting_support.json', out)


def train(cfg, ident, data, jobs, oid, pid, fits, actions, *, resume=False, pilot=False, replay=False):
    support = json.loads((PUBLIC/'fitting_support.json').read_text())
    assert support['identity'] == ident and support['all_pairs_have_easy_labels']
    if not pilot and not replay:
        assert json.loads((PUBLIC/'pilot.json').read_text())['storage_sufficient']
    refs = []; start = time.monotonic()
    for c in base.floor_api.contexts({k: data[k] for k in parent.CAUSAL_KEYS}, jobs, oid):
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; home = PRIVATE/'heads'/name
            rec = home/'complete.json'; reserve(cfg)
            if rec.exists() and resume and not replay:
                record = json.loads(rec.read_text())
                assert record['identity']['registration'] == base.artifact(PUBLIC/'registration.json')
                assert record['identity']['parent_fit'] == fits[name]
                for r in record['artifacts'].values():
                    assert base.artifact(ROOT/r['path']) == r
            else:
                if rec.exists() and not replay:
                    raise ValueError('Completed heads require resume')
                fit, ids, u, y, pr, fid = fitting(c, data, pair, fits[name], pid)
                states = {}
                for arm in api.ARMS:
                    path = (PRIVATE/'fit_replay'/name if replay else home)/arm/'checkpoint.pt.gz'
                    states[arm] = api.fit(c['x'][fit], u, y, data['sites'][ids], data['recordings'][ids],
                        data['frames'][ids], c['env'][fit], pr, arm=arm, seed=fid['seed'],
                        settings=cfg['head_training'], identity=fid, path=path,
                        heartbeat=lambda **kw: beat(group=name, arm=arm, **kw),
                        resume=resume and not replay, stop_at=cfg['pilot_updates'] if pilot else None)
                    if replay:
                        old = api.head.read_checkpoint(home/arm/'checkpoint.pt.gz')
                        for key in states[arm]:
                            if key != 'seconds':
                                api.sampling.exact(old[key], states[arm][key])
                api.assert_matched(states['marginal'], states['supervised'])
                if pilot:
                    size = sum((home/a/'checkpoint.pt.gz').stat().st_size for a in api.ARMS)
                    # Include actions/scores and bounded metadata, not just model tensors.
                    estimated = size*cfg['groups']*2+80_000_000
                    free = shutil.disk_usage(PRIVATE).free
                    out = dict(identity=ident, group=name, updates_per_head=cfg['pilot_updates'],
                        seconds=time.monotonic()-start, paired_checkpoint_bytes=size,
                        projected_checkpoint_and_action_bytes=estimated, free_disk_bytes=free,
                        storage_sufficient=free-estimated > cfg['disk_reserve_bytes'],
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, pid=os.getpid())
                    base.immutable_json(PUBLIC/'pilot.json', out)
                    if not out['storage_sufficient']:
                        raise OSError('Pilot projects crossing10GiB reserve; retain artifacts')
                    return
                if replay:
                    base.immutable_json(PUBLIC/'fit_replay.json', dict(group=name, heads=2, updates=4000,
                        exact_except_elapsed=True, input_identity=states['marginal']['identity']))
                    return
                record = dict(identity=fid, arms=list(api.ARMS), matched_initialization_and_queries=True,
                    held_outcomes_used=False, artifacts={a: base.artifact(home/a/'checkpoint.pt.gz') for a in api.ARMS})
                base.immutable_json(rec, record)
            refs.append(base.artifact(rec)); beat('paired_fit_complete', group=name, complete=len(refs))
    assert len(refs) == 108
    base.immutable_json(PUBLIC/'training_freeze.json', dict(identity=ident, groups=refs, heads=216,
        updates=432000, held_outcomes_used=False, independent_roles_read=False))
    base.immutable_json(PUBLIC/'training_runtime.json', dict(seconds=time.monotonic()-start, pid=os.getpid(),
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def decide(cfg, ident, data, jobs, oid, pid, fits, actions, *, resume=False, replay=False, **_):
    base.inter.committed(PUBLIC/'training_freeze.json')
    trained = json.loads((PUBLIC/'training_freeze.json').read_text()); assert trained['identity'] == ident
    models = {Path(r['path']).parent.name: r for r in trained['groups']}
    causal = {k: data[k] for k in parent.CAUSAL_KEYS}; refs = []; start = time.monotonic()
    for c in base.floor_api.contexts(causal, jobs, oid):
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; path = PRIVATE/'decisions'/(name+'.json'); arr = path.with_suffix('.npz')
            reserve(cfg)
            if path.exists() and resume and not replay:
                doc = json.loads(path.read_text()); assert doc['identity'] == ident and doc['fit'] == models[name]
                assert base.artifact(ROOT/doc['arrays']['path']) == doc['arrays']
            else:
                if path.exists() and not replay:
                    raise ValueError('Existing decisions require resume or replay')
                old = checked(actions[name]); new = checked(models[name]); assert new['identity']['parent_fit'] == old['fit']
                held, scores, pr = parent.predictions(c, causal, old['fit'], pid)
                ids = c['ids'][held]
                assert base.artifact(ROOT/old['arrays']['path']) == old['arrays']
                with np.load(ROOT/old['arrays']['path'], allow_pickle=False) as z:
                    np.testing.assert_array_equal(ids, z['ids']); eligible = z['eligible'].copy()
                    cached_joint = z['subset_aggregate_joint'].copy()
                u = api.head.descriptors(causal['geometry'][ids], c['floor'][held], c['prediction'][held], c['x'][held], pr)
                risks = dict(raw=parent.api.head.parent.signed(scores['subset_aggregate'].astype(float))/pr['cost_scale'])
                out = dict(ids=ids, eligible=eligible, raw_scores=risks['raw']); states = {}
                for a, ref in new['artifacts'].items():
                    assert base.artifact(ROOT/ref['path']) == ref
                    state = api.head.read_checkpoint(ROOT/ref['path']); states[a] = state
                    assert state['identity'] == new['identity'] and state['step'] == cfg['head_training']['steps']
                    p = api.predict(state, c['x'][held], u, c['env'][held])
                    out[a+'_scores'] = p[:, :4]
                    risks[a] = np.column_stack((risks['raw'][:, 0], p[:, 3].astype(float)))
                api.assert_matched(states['marginal'], states['supervised'])
                ridge = json.loads((base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
                r = ridge['artifacts']['scores']; assert base.artifact(ROOT/r['path']) == r
                with np.load(ROOT/r['path'], allow_pickle=False) as z:
                    np.testing.assert_array_equal(ids, z['ids'])
                    utility = (z['scores'][:, 5].astype(float)-z['scores'][:, 6].astype(float))/pr['cost_scale']
                keys = np.array([str(s)+'|'+str(r) for s, r in zip(causal['sites'][ids], causal['recordings'][ids])])
                fresh, solver = api.decisions(utility, risks, eligible, keys, causal['frames'][ids], ids, cfg['node_limit'])
                np.testing.assert_array_equal(fresh['raw_joint'], cached_joint)
                out.update(fresh)
                parent.previous.descriptor.write_arrays(arr, out, replay)
                doc = dict(identity=ident, fit=models[name], parent_action=actions[name], arrays=base.artifact(arr),
                           solver=solver, held_outcomes_used=False, future_fields_removed=True)
                base.immutable_json(path, doc)
            refs.append(base.artifact(path)); beat('actions_replayed' if replay else 'actions_frozen', group=name, complete=len(refs))
    assert len(refs) == 108
    base.immutable_json(PUBLIC/'decision_freeze.json', dict(identity=ident, groups=refs, held_outcomes_used=False))
    base.immutable_json(PUBLIC/('prediction_replay.json' if replay else 'decision_runtime.json'),
        dict(groups=108, exact=replay, seconds=time.monotonic()-start, pid=os.getpid(),
             peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def evaluate(cfg, ident, data, jobs, oid, pid, fits, actions, *, replay=False, **_):
    base.inter.committed(PUBLIC/'decision_freeze.json')
    frozen = json.loads((PUBLIC/'decision_freeze.json').read_text()); assert frozen['identity'] == ident
    refs = {Path(r['path']).stem: r for r in frozen['groups']}
    rows, qualities, losses = [], [], []; start = time.monotonic()
    for c in base.floor_api.contexts({k: data[k] for k in parent.CAUSAL_KEYS}, jobs, oid):
        cv, cf, (floor, ff), (neural, nf) = base.floor_api.costs(c, data, np.arange(len(c['ids'])))
        for pair in range(6):
            name = c['name']+f'_pair{pair}'; doc = checked(refs[name]); fit = checked(doc['fit'])
            assert base.artifact(ROOT/doc['arrays']['path']) == doc['arrays']
            with np.load(ROOT/doc['arrays']['path'], allow_pickle=False) as z:
                a = {k: z[k].copy() for k in z.files}
            held = np.flatnonzero(np.isin(data['sites'][c['ids']], c['pairs'][pair][1]))
            np.testing.assert_array_equal(a['ids'], c['ids'][held])
            state = api.head.read_checkpoint(ROOT/fit['artifacts']['marginal']['path']); scale = state['norm']['cost_scale']
            for arm, ref in fit['artifacts'].items():
                s = api.head.read_checkpoint(ROOT/ref['path'])
                losses.append(dict(group=name, arm=arm, step=s['step'], first=s['trace'][0], last=s['trace'][-1]))
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
                truth = api.targets(cv[ix], floor[ix], neural[ix], c['job']['design']['easy_cut'], scale)
                for arm in api.ARMS:
                    q = api.quality(a[arm+'_scores'][at], truth, data['recordings'][c['ids'][ix]], data['frames'][c['ids'][ix]])
                    qualities.append(dict(**common, policy=arm, metric=q))
        beat('held_development_scored', group=c['name'])
    sites = sorted(set(data['sites']))

    def reduce(rr):
        return {p: {k: base.inter.paired_localities([r for r in rr if r['policy'] == p], sites, k,
            cfg['bootstrap_resamples'], cfg['bootstrap_seed']) for k in rr[0]['metric']}
            for p in sorted({r['policy'] for r in rr})}

    index = {(r['group'], r['site'], r['policy']): r['metric'] for r in rows}; contrasts = []
    for new, old in CONTRASTS:
        for r in (v for v in rows if v['policy'] == new):
            contrasts.append(dict(group=r['group'], site=r['site'], seed=r['seed'], policy=new+'_vs_'+old,
                                 metric=previous.contrast(r['metric'], index[(r['group'], r['site'], old)])))
    summary, paired = reduce(rows), reduce(contrasts)
    worst = {p: dict(risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and
        r['metric']['selected_positive_harm_ratio'] > .02 for r in rows if r['policy'] == p),
        undefined_selected_risk_views=sum(r['metric']['selected_positive_harm_ratio'] is None for r in rows if r['policy'] == p),
        abstaining_views=sum(r['metric']['intervention_rate'] == 0 for r in rows if r['policy'] == p),
        worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy'] == p and r['metric']['easy_gain_CV'] is not None)) for p in POLICIES}
    primary = paired['supervised_matched_vs_marginal_matched']
    selected = [r['metric'] for r in rows if r['policy'] == 'supervised_matched']
    positive = lambda k: primary[k]['ci95'] is not None and primary[k]['ci95'][0] > 0
    gates = dict(matched_count=all(r['metric']['intervention_difference_pp'] == 0 for r in contrasts
                                if '_matched_vs_' in r['policy']),
        matched_ADE_advantage=positive('ADE_gain_percent'),
        matched_all_reference_harm_reduction=positive('all_reference_harm_reduction_pp'),
        every_view_defined_risk_within_2percent=all(m['selected_positive_harm_ratio'] is not None and m['selected_positive_harm_ratio'] <= .02 for m in selected),
        every_view_easy_preserved=all(m['easy_gain_CV'] is not None and m['easy_gain_CV'] >= -2 for m in selected),
        no_zero_CV_harm=all(m['zero_CV_harmed'] == 0 for m in selected))
    gates['exploratory_screen_pass'] = all(gates.values())
    gates.update(independent_confirmation=False, formal_primary_replaced=False, calibration_certificate=False,
                 deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    old = json.loads((parent.PUBLIC/'summary.json').read_text())
    for p, key in [('floor', 'floor'), ('raw_joint', 'subset_aggregate_joint'), ('raw_independent', 'subset_aggregate_independent')]:
        for k, v in old['summary'][key].items():
            assert summary[p][k] == v, (p, k)
    out = dict(identity=ident, result_source='fresh_run_neural_training_actions_and_development_readout',
               forecasts_floor_utility_all_risk='cached_verified', independent_confirmation='not_run',
               summary=summary, paired=paired, quality=reduce(qualities), worst_views=worst, gates=gates,
               by_seed={str(s): reduce([r for r in rows if r['seed'] == s]) for s in (17, 29, 43)})
    base.immutable_json(PRIVATE/'details.json', dict(rows=rows, qualities=qualities, contrasts=contrasts, losses=losses))
    base.immutable_json(PUBLIC/'summary.json', out); base.immutable_json(PUBLIC/'gates.json', gates)
    base.immutable_json(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),
        dict(summary=base.artifact(PUBLIC/'summary.json'), details=base.artifact(PRIVATE/'details.json'),
             exact=replay, seconds=time.monotonic()-start, pid=os.getpid(),
             peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['register', 'audit', 'pilot', 'train', 'decide',
        'evaluate', 'replay_fit', 'replay_decide', 'replay_evaluate'])
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    if a.phase == 'register':
        _, ident = identity(); base.immutable_json(PUBLIC/'registration.json', ident); print(json.dumps(ident)); return
    base.torch.set_num_threads(4); base.torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); beat('started', phase=a.phase); v = load()
        reserve(v[0])
        if a.phase == 'audit':
            audit(*v)
        elif a.phase in ('pilot', 'train', 'replay_fit'):
            train(*v, resume=a.resume, pilot=a.phase == 'pilot', replay=a.phase == 'replay_fit')
        elif a.phase in ('decide', 'replay_decide'):
            decide(*v, resume=a.resume, replay=a.phase == 'replay_decide')
        else:
            evaluate(*v, replay=a.phase == 'replay_evaluate')
        beat('complete', phase=a.phase)


if __name__ == '__main__':
    main()
