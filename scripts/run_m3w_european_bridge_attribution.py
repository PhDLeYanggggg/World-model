"""Train motion-only bridges and freeze matched causal intervention controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_dual_event_bridge as bridge
from scripts import build_m3w_european_selection_data as adapter
from src.world_model.m3w_bridge_attribution import motion_pair, query_groups, decisions, independent_replay, POLICIES
from src.evaluation.m3w_native_metrics import native_errors

base, inc, parent = bridge.base, bridge.inc, bridge.parent
artifact, digest, array_hash = bridge.artifact, bridge.digest, bridge.array_hash
require_committed = bridge.require_committed
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_bridge_attribution_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_bridge_attribution_v1'
CONFIG = 'configs/m3w_european_bridge_attribution_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_bridge_attribution.py',
    'scripts/run_m3w_european_bridge_attribution.py', 'scripts/evaluate_m3w_european_bridge_attribution.py',
    'tests/test_m3w_bridge_attribution.py',
    'outputs/publication_readiness_2026_09/european_bridge_attribution_v1/registration.md']
TASKS = ('utility', 'all_risk')


def immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n'
    if path.exists():
        if path.read_text() != encoded: raise ValueError('Immutable receipt changed: '+str(path))
    else:
        temp = path.with_suffix(path.suffix+'.tmp'); temp.write_text(encoded); os.replace(temp, path)


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    assert cfg['policies'] == list(POLICIES) and cfg['seeds'] == [17, 29, 43]
    assert (cfg['groups'], cfg['new_neural_heads'], cfg['new_neural_updates'], cfg['new_ridge_fits']) == (18, 36, 72000, 36)
    assert cfg['risk_budget'] == .02 and cfg['ridge_alpha'] == .01
    assert not any(cfg[k] for k in ('calibration_access', 'confirmation_access', 'threshold_refit',
        'new_forecaster_training', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    bc, _, bank, pid, bid = bridge.registration()
    assert cfg['head_training'] == bc['head_training']
    assert digest(bridge.PUBLIC/'summary_metrics.json') == cfg['parent_summary_sha256']
    complete = json.loads((bridge.PUBLIC/'completion_checks.json').read_text())
    assert complete['all_passed'] and complete['identity'] == bid
    for ref in complete['evaluation_receipts']: assert artifact(ROOT/ref['path']) == ref
    value = dict(bindings={p: digest(ROOT/p) for p in FILES}, parent=bid,
        parent_summary=artifact(bridge.PUBLIC/'summary_metrics.json'),
        parent_completion=artifact(bridge.PUBLIC/'completion_checks.json'), rosters=bid['rosters'])
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, value)
    elif json.loads(path.read_text()) != value: raise ValueError('Registered implementation changed')
    if not create: require_committed(path)
    return cfg, bank, pid, value


def selection(data, bank, g):
    seed = int(g['group'].split('_seed')[1].split('_')[0])
    x, env, pair, old, meta = bridge.selection_pair(data, bank, g['producer'], seed, g['controller'])
    cv = base.baseline_numpy(data['history'], 1)-data['origin'][:, None]
    damp = base.baseline_numpy(data['history'], 3)-data['origin'][:, None]
    mx, me, mr, mp = motion_pair(data['geometry'], cv, damp, old['easy']['floor_bit'], old['all']['floor_bit'])
    return dict(full=(x, env, pair['easy'], pair['all']), motion_only=(mx, me, mr, mp)), meta


def fit_one(x, y, sites, cv, env, tx, tenv, directory, hid, cfg, seed, task, resume, pilot):
    if (directory/'complete.json').exists(): return base.checked(directory, hid)
    if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('Below 10 GiB; retain checkpoints')
    pr = base.ordinary.preprocess(x, y, cv, sites, '__excluded_outer__')
    kw = dict(seed=seed, settings=cfg['head_training'], identity=hid, directory=directory,
        resume=resume, stop_at=100 if pilot else None, heartbeat=lambda **v: beat(head=directory.name, **v))
    if task == 'utility':
        model, fit = base.geometric.fit(x, y, sites, env, pr, task='utility', **kw)
        predict = lambda xx, ee: base.geometric.predict(model, xx, ee, pr)
        kind = 'bounded_utility'
    else:
        fixed = inc.prior.fixed_rank_scale(y, sites, pr, seed=seed, batch_size=256, batches=40)
        model, fit = base.ranked.fit(x, y, sites, env, pr, rank_weight=1., epsilon=1e-6, fixed_denominator=fixed, **kw)
        predict = lambda xx, ee: base.hurdle.predict(model, xx, ee, pr)
        kind = 'ranked_risk'
    if pilot: return dict(identity=hid, fit=fit, checkpoint=artifact(directory/'checkpoint.pt'), kind=kind)
    scores = predict(tx, tenv)
    parent.atomic_npz(directory/'scores.npz', ids=np.arange(len(tx)), scores=scores)
    r = dict(identity=hid, fit=fit, kind=kind, replay_exact=True,
        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'), scores=artifact(directory/'scores.npz')))
    fresh, _ = inc.prior.previous.restored(r)
    count = min(len(tx), cfg['replay_rows'])
    np.testing.assert_array_equal(fresh(tx[:count], tenv[:count]), scores[:count])
    assert fit['unknown_rows_sampled'] == 0 and fit['complete']
    r['prefix_replay_rows'] = count
    immutable_json(directory/'complete.json', r)
    beat('head_complete', head=directory.name, seconds=fit['seconds'], parameters=fit['parameters'])
    return r


def train(cfg, bank, pid, identity, resume=False, pilot=False):
    cached = bridge.checked_training(identity['parent'])
    previous = {g['group']: g for g in cached['groups']}
    inc.ensure_frozen(); _, bcfg, ctx, bid, _, iid = inc.load()
    assert iid['rosters'] == identity['rosters']
    data = ctx[2]; held_data = adapter.load(parent, pid)
    assert not set(held_data['sites']) & set(sum(identity['rosters'], []))
    refs, ridge, rows = [], [], []
    for easy, all_event in bridge.source_pairs(bcfg, ctx, bid):
        fold, seed, held = easy['fold'], easy['seed'], easy['ids']
        for controller in range(3):
            if fold == controller: continue
            name = f'fold{fold}_seed{seed}_controller{controller}'; old = previous[name]
            take = np.isin(data['sites'][held], identity['rosters'][controller]); ids = held[take]
            assert array_hash(ids) == old['ids_sha256'] and len(ids) == old['training_rows']
            cvp = easy['cv'][take]
            damp = base.baseline_numpy(data['history'][ids], 3)-data['origin'][ids, None]
            x, env, r, p = motion_pair(data['geometry'][ids], cvp, damp, easy['bits'][take], all_event['bits'][take])
            tx, tenv, _, _ = selection(held_data, bank, old)[0]['motion_only']
            def error(pred):
                return native_errors(pred.astype(float)+data['origin'][ids, None], data['target_eval'][ids],
                    data['valid'][ids], np.ones(len(ids)))[0]
            cv = data['baseline_ade'][ids, 1]
            yy = bridge.targets(cv, error(r), error(p), old['easy_cut'])
            row = dict(old, input_sha256=array_hash(x), env_sha256=array_hash(env),
                endpoint_choice_sha256=array_hash(easy['bits'][take], all_event['bits'][take]),
                pair='motion_only', neural_candidate_in_input=False, neural_policy_bits_in_input=False)
            for task in TASKS:
                hid = dict(experiment=identity, group=name, task=task, seed=seed, training=row,
                    labels_sha256=array_hash(yy[task]), inference_sha256=array_hash(tx, tenv),
                    source_role='controller_B', readout_role='reused_opened_model_selection')
                directory = PRIVATE/'heads'/(name+'_'+task)
                beat('head_started', head=directory.name, rows=len(ids), pilot=pilot)
                receipt = fit_one(x, yy[task], data['sites'][ids], cv, env, tx, tenv,
                    directory, hid, cfg, seed, task, resume, pilot)
                if pilot:
                    immutable_json(PRIVATE/'pilot.json', receipt); beat('pilot_complete', steps=100); return
                refs.append(artifact(directory/'complete.json'))
                rd = PRIVATE/'ridge'/(name+'_'+task)
                inc.prior.fit_ridge(x, yy[task], cv, data['sites'][ids], tx, tenv, np.arange(len(tx)),
                    'utility' if task == 'utility' else 'risk', hid, rd)
                ridge.append(artifact(rd/'complete.json'))
                state = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
                before = torch.load(bridge.PRIVATE/'heads'/(name+'_'+task)/'checkpoint.pt', map_location='cpu', weights_only=False)
                np.testing.assert_array_equal(state['draws'], before['draws'])
                for k in ('weights', 'known', 'cost_scale'):
                    np.testing.assert_array_equal(state['preprocess'][k], before['preprocess'][k])
            rows.append(row); beat('group_complete', group=name)
    assert len(refs) == len(ridge) == 36 and len(rows) == 18
    immutable_json(PRIVATE/'training_complete.json', dict(identity=identity, heads=refs, ridge=ridge,
        groups=rows, updates=72000, all_passed=True, no_selection_labels_read=True,
        result_source='fresh_run', matched_parent_draws_support_cost_scale=True,
        cached_full_training=artifact(bridge.PRIVATE/'training_complete.json')))
    immutable_json(PUBLIC/'training_receipt.json', dict(manifest=artifact(PRIVATE/'training_complete.json'),
        new_heads=36, new_ridge=36, updates=72000, cached_full_heads=36, cached_full_ridge=36,
        result_source='fresh_run_motion_only_cached_verified_full', selection_labels_read=False,
        calibration_opened=False, confirmation_opened=False))


def checked_training(identity):
    bridge.checked_training(identity['parent'])
    r = json.loads((PRIVATE/'training_complete.json').read_text())
    assert r['identity'] == identity and r['all_passed'] and len(r['heads']) == len(r['ridge']) == 36
    for ref in r['heads']+r['ridge']:
        assert artifact(ROOT/ref['path']) == ref
        rr = json.loads((ROOT/ref['path']).read_text())
        for a in rr['artifacts'].values(): assert artifact(ROOT/a['path']) == a
    return r


def scores(pair, name):
    root = bridge.PRIVATE if pair == 'full' else PRIVATE
    out = []
    for family in ('heads', 'ridge'):
        for task in TASKS:
            with np.load(root/family/(name+'_'+task)/'scores.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], np.arange(len(z['scores'])))
                out.append(z['scores'].copy())
    return out


def decide(cfg, bank, pid, identity):
    trained = checked_training(identity); data = adapter.load(parent, pid); refs = []
    groups = query_groups(data['sites'], data['recordings'], data['frames']); ids = np.arange(len(data['sites']))
    moving = np.linalg.norm(data['history'][:, -1]-data['history'][:, -2], axis=1) > 0
    for g in trained['groups']:
        pairs, _ = selection(data, bank, g)
        for pair, (x, env, _, _) in pairs.items():
            name = g['group']; values = scores(pair, name)
            out, common, counts = decisions(*values, moving, env, groups, ids)
            other = independent_replay(*values, moving, env, groups, ids)
            for k in POLICIES: np.testing.assert_array_equal(out[k], other[k])
            if pair == 'full':
                with np.load(bridge.PRIVATE/'decisions'/(name+'.npz'), allow_pickle=False) as z:
                    np.testing.assert_array_equal(out['neural'], z['all_risk_only'])
            path = PRIVATE/'decisions'/(name+'_'+pair+'.npz')
            arrays = dict(out, common=common, counts=counts, envelope=env)
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    for k, v in arrays.items(): np.testing.assert_array_equal(z[k], v)
            else: parent.atomic_npz(path, **arrays)
            receipt = dict(identity=identity, group=g, pair=pair, artifact=artifact(path),
                input_sha256=array_hash(x), score_sha256=array_hash(*values), rows=len(x), queries=len(groups),
                zero_count_queries=int((counts == 0).sum()), matched_actions=int(counts.sum()),
                labels_read=False, scalar_decisions_exact=True)
            immutable_json(path.with_suffix('.json'), receipt); refs.append(artifact(path.with_suffix('.json')))
            beat('decisions_frozen', group=name, pair=pair)
    immutable_json(PRIVATE/'decisions_complete.json', dict(identity=identity, groups=refs,
        training=artifact(PRIVATE/'training_complete.json'), all_passed=True))
    immutable_json(PUBLIC/'decision_freeze.json', dict(identity=identity, groups=18, pairs=2, views=396,
        manifest=artifact(PRIVATE/'decisions_complete.json'), outcomes_read=False,
        role='reused_opened_model_selection', calibration_opened=False, confirmation_opened=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['register', 'pilot', 'train', 'decide', 'evaluate'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    for d in (PRIVATE, PUBLIC): d.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, bank, pid, identity = registration(create=args.phase == 'register')
        if args.phase in ('train', 'pilot'): train(cfg, bank, pid, identity, args.resume, args.phase == 'pilot')
        elif args.phase == 'decide': decide(cfg, bank, pid, identity)
        elif args.phase == 'evaluate':
            from scripts.evaluate_m3w_european_bridge_attribution import evaluate
            evaluate(sys.modules[__name__], cfg, bank, pid, identity)


if __name__ == '__main__': main()
