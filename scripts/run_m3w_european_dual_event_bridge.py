"""Fit aligned dual-event policy costs on producer-excluded source rows."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_selection_readout as parent
from scripts import build_m3w_european_selection_data as adapter
from src.world_model.m3w_dual_event_bridge import features, targets, choices, scalar_replay, POLICIES
from src.evaluation.m3w_native_metrics import native_errors

inc, base = parent.inc, parent.inc.base
artifact, digest, immutable_json, array_hash = inc.artifact, inc.digest, inc.immutable_json, inc.array_hash
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_dual_event_bridge_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_dual_event_bridge_v1'
CONFIG = 'configs/m3w_european_dual_event_bridge_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_dual_event_bridge.py',
    'scripts/run_m3w_european_dual_event_bridge.py', 'scripts/evaluate_m3w_european_dual_event_bridge.py',
    'tests/test_m3w_dual_event_bridge.py',
    'outputs/publication_readiness_2026_09/european_dual_event_bridge_v1/registration.md']
TASKS = ('utility', 'all_risk', 'easy_risk')


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def require_committed(path):
    rel = str(path.relative_to(ROOT))
    if subprocess.check_output(['git', 'show', 'HEAD:'+rel], cwd=ROOT) != path.read_bytes():
        raise ValueError('Must commit before execution: '+rel)
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', rel], cwd=ROOT).decode().strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'HEAD'], cwd=ROOT).decode().split()[0]
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=ROOT, check=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    assert (cfg['groups'], cfg['new_neural_heads'], cfg['neural_updates'], cfg['ridge_fits']) == (18, 54, 108000, 54)
    assert cfg['policies'] == list(POLICIES) and cfg['seeds'] == [17, 29, 43]
    assert (cfg['risk_budget_all'], cfg['risk_budget_easy'], cfg['ridge_alpha']) == (.02, .02, .01)
    assert not any(cfg[k] for k in ('new_forecaster_training', 'threshold_refit', 'calibration_access',
        'confirmation_access', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    assert digest(parent.PUBLIC/'summary_metrics.json') == cfg['parent_summary_sha256']
    pcfg, bank, pid = parent.registration()
    assert cfg['head_training'] == json.loads((ROOT/inc.CONFIG).read_text())['head_training']
    done = json.loads((parent.PUBLIC/'completion_checks.json').read_text())
    assert done['all_passed'] and done['identity'] == pid
    for ref in done['evaluation_receipts']:
        assert artifact(ROOT/ref['path']) == ref
    value = dict(bindings={p: digest(ROOT/p) for p in FILES}, parent=pid,
        parent_completion=artifact(parent.PUBLIC/'completion_checks.json'),
        parent_summary=artifact(parent.PUBLIC/'summary_metrics.json'), rosters=bank['producer_rosters'])
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, value)
    elif json.loads(path.read_text()) != value: raise ValueError('Registered implementation changed')
    if not create: require_committed(path)
    return cfg, pcfg, bank, pid, value


def source_pairs(bcfg, ctx, bid):
    data = ctx[2]; pending = None
    for name, fold, seed, event, design, a, d, x, env, bits, provenance in base.groups(bcfg, ctx, bid):
        held = design['held_ids']
        g = dict(original=name, x=x, env=env, roles=dict(readout=held))
        old = inc.prior.old_scores(g, data, bid)[1]
        policy = np.where(old[:, None, None], a['p'][held], d[held])
        item = dict(name=name, fold=fold, seed=seed, design=design, ids=held,
                    prediction=policy, bits=bits[held], old=old, cv=a['b'][held], lineage=a['lineage'])
        if event == 'all':
            pending = item
        else:
            assert pending is not None and (fold, seed) == (pending['fold'], pending['seed'])
            np.testing.assert_array_equal(held, pending['ids'])
            np.testing.assert_array_equal(item['cv'], pending['cv'])
            yield item, pending
            pending = None
    assert pending is None


def selection_pair(data, bank, fold, seed, controller):
    pair, meta, out = {}, {}, {}
    b = base.baseline_numpy(data['history'], 1)-data['origin'][:, None]
    d = base.baseline_numpy(data['history'], 3)-data['origin'][:, None]
    for event in ('easy', 'all'):
        name = f'fold{fold}_seed{seed}_{event}_controller{controller}'
        receipt = json.loads((parent.PRIVATE/'decisions'/(name+'.json')).read_text())
        assert receipt['group'] == bank['groups'][name]
        for ref in receipt['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
        with np.load(parent.PRIVATE/'decisions'/(name+'.npz'), allow_pickle=False) as z:
            out[event] = {k: z[k].copy() for k in ('old_stop', 'floor_bit', 'add_only')}
        p = np.load(ROOT/receipt['artifacts']['neural']['path'], allow_pickle=False, mmap_mode='r')
        floor = np.where(out[event]['floor_bit'][:, None, None], d, b)
        pair[event] = np.where(out[event]['old_stop'][:, None, None], p, floor)
        meta[event] = receipt['group']
    bits = np.column_stack([out[e][k] for k in ('old_stop', 'floor_bit') for e in ('easy', 'all')])
    x, env = features(data['geometry'], b, pair['easy'], pair['all'], bits)
    return x, env, pair, out, meta


def fit_one(x, y, sites, cv, env, tx, tenv, ids, directory, hid, cfg, seed, task, resume, pilot):
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
        model, fit = base.ranked.fit(x, y, sites, env, pr, rank_weight=1., epsilon=1e-6,
                                    fixed_denominator=fixed, **kw)
        predict = lambda xx, ee: base.hurdle.predict(model, xx, ee, pr)
        kind = 'ranked_risk'
    if pilot:
        return dict(identity=hid, fit=fit, checkpoint=artifact(directory/'checkpoint.pt'), kind=kind)
    scores = predict(tx, tenv)
    parent.atomic_npz(directory/'scores.npz', ids=ids, scores=scores)
    r = dict(identity=hid, fit=fit, kind=kind, replay_exact=True,
        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'), scores=artifact(directory/'scores.npz')))
    fresh, _ = inc.prior.previous.restored(r)
    n = min(len(tx), cfg['replay_rows'])
    np.testing.assert_array_equal(fresh(tx[:n], tenv[:n]), scores[:n])
    assert fit['unknown_rows_sampled'] == 0 and fit['complete']
    r['prefix_replay_rows'] = n
    immutable_json(directory/'complete.json', r)
    beat('head_complete', head=directory.name, seconds=fit['seconds'], parameters=fit['parameters'])
    return r


def train(cfg, bank, pid, identity, *, resume=False, pilot=False):
    inc.ensure_frozen()
    _, bcfg, ctx, bid, _, iid = inc.load()
    assert iid['rosters'] == identity['rosters']
    data = ctx[2]; selection = adapter.load(parent, pid)
    if set(selection['sites']) & set(sum(identity['rosters'], [])): raise ValueError('Selection/source overlap')
    refs, ridge, rows = [], [], []
    for easy, all_event in source_pairs(bcfg, ctx, bid):
        fold, seed = easy['fold'], easy['seed']; held = easy['ids']
        for controller in range(3):
            if fold == controller: continue
            name = f'fold{fold}_seed{seed}_controller{controller}'
            take = np.isin(data['sites'][held], identity['rosters'][controller]); ids = held[take]
            assert set(data['sites'][ids]) == set(identity['rosters'][controller])
            assert not set(data['sites'][ids]) & set(identity['rosters'][fold])
            r, p = easy['prediction'][take], all_event['prediction'][take]
            bits = np.column_stack([v[k][take] for k in ('old', 'bits') for v in (easy, all_event)])
            x, env = features(data['geometry'][ids], easy['cv'][take], r, p, bits)
            tx, tenv, _, _, meta = selection_pair(selection, bank, fold, seed, controller)
            def error(prediction):
                return native_errors(prediction.astype(float)+data['origin'][ids, None],
                    data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))[0]
            cv = data['baseline_ade'][ids, 1]
            yy = targets(cv, error(r), error(p), easy['design']['easy_cut'])
            row = dict(group=name, producer=fold, controller=controller,
                producer_roster=identity['rosters'][fold], controller_roster=identity['rosters'][controller],
                training_rows=len(ids), known_rows=int(np.isfinite(cv).sum()),
                ids_sha256=array_hash(ids), input_sha256=array_hash(x), env_sha256=array_hash(env),
                endpoint_choice_sha256=array_hash(bits), source_forecaster=easy['lineage']['final_producer'],
                easy_cut=easy['design']['easy_cut'], hard_cut=easy['design']['hard_cut'])
            for task in TASKS:
                hid = dict(experiment=identity, group=name, task=task, seed=seed, training=row,
                    labels_sha256=array_hash(yy[task]), inference_sha256=array_hash(tx, tenv),
                    source_role='controller_B', readout_role='reused_opened_model_selection')
                directory = PRIVATE/'heads'/(name+'_'+task)
                beat('head_started', head=directory.name, rows=len(ids), pilot=pilot)
                receipt = fit_one(x, yy[task], data['sites'][ids], cv, env, tx, tenv,
                    np.arange(len(tx)), directory, hid, cfg, seed, task, resume, pilot)
                if pilot:
                    immutable_json(PRIVATE/'pilot.json', receipt)
                    beat('pilot_complete', steps=100, seconds=receipt['fit']['seconds']); return
                refs.append(artifact(directory/'complete.json'))
                rd = PRIVATE/'ridge'/(name+'_'+task)
                inc.prior.fit_ridge(x, yy[task], cv, data['sites'][ids], tx, tenv,
                    np.arange(len(tx)), 'utility' if task == 'utility' else 'risk', hid, rd)
                ridge.append(artifact(rd/'complete.json'))
            states = [torch.load(PRIVATE/'heads'/(name+'_'+t)/'checkpoint.pt', map_location='cpu', weights_only=False) for t in TASKS]
            for s in states[1:]:
                np.testing.assert_array_equal(s['draws'], states[0]['draws'])
                for k in ('mean', 'std', 'weights', 'known'):
                    np.testing.assert_array_equal(s['preprocess'][k], states[0]['preprocess'][k])
            rows.append(row); beat('group_complete', group=name)
    assert len(refs) == len(ridge) == 54 and len(rows) == 18
    immutable_json(PRIVATE/'training_complete.json', dict(identity=identity, heads=refs, ridge=ridge,
        groups=rows, updates=108000, all_passed=True, no_selection_labels_read=True,
        result_source='fresh_run', matched_training_draws=True, matched_feature_statistics=True))
    immutable_json(PUBLIC/'training_receipt.json', dict(manifest=artifact(PRIVATE/'training_complete.json'),
        heads=54, ridge=54, updates=108000, result_source='fresh_run', source_roles=identity['rosters'],
        selection_labels_read=False, calibration_opened=False, confirmation_opened=False))


def checked_training(identity):
    r = json.loads((PRIVATE/'training_complete.json').read_text())
    assert r['identity'] == identity and r['all_passed'] and len(r['heads']) == len(r['ridge']) == 54
    for ref in r['heads']+r['ridge']:
        assert artifact(ROOT/ref['path']) == ref
        rr = json.loads((ROOT/ref['path']).read_text())
        for a in rr['artifacts'].values(): assert artifact(ROOT/a['path']) == a
    return r


def decide(cfg, bank, pid, identity):
    trained = checked_training(identity); data = adapter.load(parent, pid); refs = []
    moving = np.linalg.norm(data['history'][:, -1]-data['history'][:, -2], axis=1) > 0
    for g in trained['groups']:
        name = g['group']; x, env, _, _, _ = selection_pair(data, bank, g['producer'],
                                                         int(name.split('_seed')[1].split('_')[0]), g['controller'])
        values = {}
        for family in ('heads', 'ridge'):
            for task in TASKS:
                path = PRIVATE/family/(name+'_'+task)/'scores.npz'
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'], np.arange(len(x)))
                    values[family+'__'+task] = z['scores'].copy()
        out = dict(envelope=env, **values)
        for arm in POLICIES:
            family = 'ridge' if arm == 'ridge_dual' else 'heads'
            scores = [values[family+'__'+t] for t in TASKS]
            out[arm] = choices(*scores, moving, env, arm=arm)
            np.testing.assert_array_equal(out[arm], scalar_replay(*scores, moving, env, arm))
        path = PRIVATE/'decisions'/(name+'.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                for k, v in out.items(): np.testing.assert_array_equal(z[k], v)
        else: parent.atomic_npz(path, **out)
        receipt = dict(identity=identity, group=g, artifact=artifact(path), input_sha256=array_hash(x),
            rows=len(x), labels_read=False, scalar_decisions_exact=True)
        immutable_json(path.with_suffix('.json'), receipt); refs.append(artifact(path.with_suffix('.json')))
        beat('decisions_frozen', group=name)
    immutable_json(PRIVATE/'decisions_complete.json', dict(identity=identity, groups=refs,
        training=artifact(PRIVATE/'training_complete.json'), all_passed=True))
    immutable_json(PUBLIC/'decision_freeze.json', dict(identity=identity, groups=18, views=126,
        manifest=artifact(PRIVATE/'decisions_complete.json'), outcomes_read=False,
        selection_role='reused_opened_model_selection', calibration_opened=False, confirmation_opened=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['register', 'pilot', 'train', 'decide', 'evaluate'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    for d in (PRIVATE, PUBLIC): d.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, _, bank, pid, identity = registration(create=args.phase == 'register')
        if args.phase in ('train', 'pilot'):
            train(cfg, bank, pid, identity, resume=args.resume, pilot=args.phase == 'pilot')
        elif args.phase == 'decide': decide(cfg, bank, pid, identity)
        elif args.phase == 'evaluate':
            from scripts.evaluate_m3w_european_dual_event_bridge import evaluate
            evaluate(sys.modules[__name__], cfg, bank, pid, identity)


if __name__ == '__main__': main()
