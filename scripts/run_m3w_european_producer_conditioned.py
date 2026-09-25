"""Matched producer-aware neural controller refits; no new trajectory forecaster."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_floor_relative as base
from scripts import run_m3w_european_producer_transport as transport
from scripts import run_m3w_european_causal_abstention as stopped
from src.world_model.m3w_producer_conditioned import ARMS, augment, producer_roles, safe_choice, independent_choice
from src.world_model.m3w_floor_relative import matched_features, relative_targets, fixed_rank_scale
from src.world_model.m3w_european_source_intervention import causal_cost_features
from src.world_model.m3w_european_conditional_risk import pointwise_rule
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_producer_conditioned_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_producer_conditioned_v1'
CONFIG = 'configs/m3w_european_producer_conditioned_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_producer_conditioned.py', 'src/world_model/m3w_producer_conditioned.py',
    'tests/test_m3w_producer_conditioned.py', 'tests/test_m3w_producer_conditioned_protocol.py',
    'outputs/publication_readiness_2026_09/european_producer_conditioned_v1/registration.md')
digest, artifact, immutable_json, array_hash = base.digest, base.artifact, base.immutable_json, base.array_hash


def beat(state, **kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    base.cross.json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def validate(cfg):
    expected = dict(mode='fitting', arms=list(ARMS), seeds=[17, 29, 43], events=['all', 'easy'],
        reference='floor', risk_budget=.02, rank_weight=1., rank_epsilon=1e-6, fixed_scale_batches=40,
        groups=18, producer_branches=2, views=180, new_heads=108, updates=216000,
        bootstrap_resamples=3000, bootstrap_seed=39271, replay_rows_per_producer=4096)
    for k, v in expected.items():
        if cfg[k] != v: raise ValueError('Changed registered comparison: '+k)
    for k in ('new_forecaster_training', 'threshold_refit', 'calibration_refit', 'reserved_roles_opened',
              'deployment_changed', 'stage5c_executed', 'smc_enabled'):
        if cfg[k]: raise ValueError('Development-only controller refit')


def load():
    cfg = json.loads((ROOT/CONFIG).read_text()); validate(cfg)
    stopped.ensure_both_frozen()
    bcfg, ctx, _, bid = base.load('fitting')
    assert cfg['head_training'] == bcfg['head_training']
    previous = ROOT/'outputs/publication_readiness_2026_09/european_support_factorization_v1'
    assert digest(previous/'summary_metrics.json') == cfg['prior_summary_sha256']
    done = json.loads((previous/'completion_checks.json').read_text()); assert done['all_passed']
    for f, sha in done['artifact_hashes'].items():
        if digest(previous/f) != sha: raise ValueError('Preceding evidence changed')
    tid = json.loads((transport.PRIVATE/'identity.json').read_text()); transport.assert_identity(tid)
    pid = ctx[3]['parent_identity']['geometric_identity']['old_identity']['producer_identity']
    assert tid['previous_identity']['producer_identity'] == pid
    identity = dict(bindings={f: digest(ROOT/f) for f in FILES}, parent=bid,
        prior_completion=artifact(previous/'completion_checks.json'), transport_identity=artifact(transport.PRIVATE/'identity.json'),
        transport_bank=artifact(transport.PUBLIC/'bank_completion.json'),
        transport_replay=artifact(transport.PUBLIC/'bank_replay.json'))
    immutable_json(PRIVATE/'identity.json', identity)
    return cfg, bcfg, ctx, bid, tid, identity


def restored(r):
    for ref in r['artifacts'].values():
        if artifact(ROOT/ref['path']) != ref: raise ValueError('Checkpoint/score artifact changed')
    s = torch.load(ROOT/r['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False)
    assert s['identity'] == r['identity'] and s['step'] == 2000
    pr = s['preprocess']; width = s['settings']['width']; seed = s['seed']
    if r['kind'] == 'ordinary_utility':
        model = base.ordinary.build_head(width, pr, seed)
        predict = lambda xx, dd: base.ordinary.predict_neural(model, xx, np.zeros(len(xx), bool), pr)
    elif r['kind'] == 'bounded_utility':
        model = base.geometric.initialize_head(width, pr, seed, 'utility', s['mean_envelope'])
        predict = lambda xx, dd: base.geometric.predict(model, xx, dd, pr)
    else:
        model = base.hurdle.initialize(pr, s['prior'], width, seed)
        predict = lambda xx, dd: base.hurdle.predict(model, xx, dd, pr)
    model.load_state_dict(s['model'])
    return predict, s


def neural_bank(fold, seed, half, design, data, tid, cfg, replay=False):
    key, rec = transport.small_record(fold, half, seed, design, data, tid)
    ref = json.loads((transport.PRIVATE/'predictions'/(key+'.json')).read_text())
    assert ref['identity'] == tid and ref['producer_checkpoint'] == rec['checkpoint']
    ids = design['held_ids']; p = transport.previous.read_predictions(ref, ids)
    if replay:
        fcfg = json.loads((ROOT/transport.previous.bank.parent.CONFIG).read_text())
        model = transport.model_from(rec, fcfg); n = min(cfg['replay_rows_per_producer'], len(ids))
        fresh = transport.predict(model, data, ids[:n], 128)
        np.testing.assert_array_equal(fresh, p[:n])
    return p, dict(checkpoint=rec['checkpoint'], prediction=artifact(ROOT/ref['path']), training_sites=rec['identity']['training_sites'])


def make_banks():
    cfg, bcfg, ctx, bid, tid, identity = load(); data = ctx[2]; refs = []; replayed = set()
    for name, fold, seed, event, design, a, d4, _, env, bits4, provenance in base.groups(bcfg, ctx, bid):
        ids = design['held_ids']; b = a['b']; damping = transport.baseline_numpy(data['history'], 3)-data['origin'][:, None]
        x, _ = causal_cost_features(data['geometry'], b, damping)
        env = base.geometric.rollout_envelope(b, damping); same = np.all(b == damping, axis=(1, 2))
        moving = np.linalg.norm(np.diff(data['history'], axis=1), axis=2).sum(1) > 0
        for half in (0, 1):
            key = f'fold{fold}_half{half}_seed{seed}'; path = PRIVATE/'floor_banks'/(name+f'_half{half}.npz')
            nr, nref = neural_bank(fold, seed, half, design, data, tid, cfg, replay=key not in replayed); replayed.add(key)
            rows = {}; records = {}
            for task, directory in [('utility', base.PRIVATE/'inner_shared'/key),
                                    ('risk', base.PRIVATE/'fitting/inner_risk'/(key+'_'+event))]:
                r = json.loads((directory/'complete.json').read_text())
                assert r['identity']['experiment'] == (base.shared_identity(bid) if task == 'utility' else bid)
                assert set(r['identity']['lineage']['training_sites']) == set(nref['training_sites'])
                assert not set(nref['training_sites']) & set(data['sites'][ids])
                pred, state = restored(r)
                with np.load(ROOT/r['artifacts']['scores']['path'], allow_pickle=False) as z:
                    prefix = z['ids'][:cfg['replay_rows_per_producer']]
                    value = pred(x[prefix], env[prefix])
                    if task == 'utility': value[same[prefix]] = 0
                    np.testing.assert_array_equal(value, z['scores'][:len(prefix)])
                rows[task] = pred(x[ids], env[ids])
                if task == 'utility': rows[task][same[ids]] = 0
                records[task] = artifact(directory/'complete.json')
            switch = pointwise_rule(rows['utility'][:, 0]-rows['utility'][:, 1], rows['risk'], moving[ids],
                                    budget=cfg['risk_budget'], support_available=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    for k, v in dict(ids=ids, switch=switch, **rows).items(): np.testing.assert_array_equal(z[k], v)
            else: np.savez(path, ids=ids, switch=switch, **rows)
            receipt = dict(identity=identity, group=name, half=half, neural=nref, floor_producers=records,
                artifact=artifact(path), source_exclusion_pass=True, floor_checkpoint_prefix_replays=2)
            immutable_json(path.with_suffix('.json'), receipt); refs.append(artifact(path.with_suffix('.json')))
        beat('floor_banks_complete', group=name)
    assert len(refs) == 36 and len(replayed) == 18
    immutable_json(PRIVATE/'banks_complete.json', dict(identity=identity, groups=refs, neural_prefix_replays=18,
        floor_prefix_replays=72, new_outcome_readout=False, all_passed=True))


def checked_banks(identity):
    r = json.loads((PRIVATE/'banks_complete.json').read_text()); assert r['identity'] == identity and r['all_passed']
    assert len(r['groups']) == 36
    for ref in r['groups']:
        assert artifact(ROOT/ref['path']) == ref
        rec = json.loads((ROOT/ref['path']).read_text()); assert rec['identity'] == identity
        assert artifact(ROOT/rec['artifact']['path']) == rec['artifact']


def branch_inputs(name, half, design, data, a, identity):
    rec = json.loads((PRIVATE/'floor_banks'/(name+f'_half{half}.json')).read_text())
    assert rec['identity'] == identity and artifact(ROOT/rec['artifact']['path']) == rec['artifact']
    ids = design['held_ids']; n = transport.previous.read_predictions(rec['neural']['prediction'], ids)
    with np.load(ROOT/rec['artifact']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids); bits = z['switch'].copy()
    b = a['b'][ids]; damping = transport.baseline_numpy(data['history'][ids], 3)-data['origin'][ids, None]
    d = np.where(bits[:, None, None], damping, b)
    x, env = matched_features(data['geometry'][ids], b, d, n, bits)
    return x, env, n, d, bits


def tag_inputs(design, data, tid, fold):
    halves = tid['previous_identity']['producer_identity']['halves'][str(fold)]
    return producer_roles(data['sites'], design['train_ids'], design['held_ids'], halves)


def fit_one(x, y, sites, cv, env, target_x, target_env, keys, *, directory, hid, cfg, seed, task, resume, pilot):
    if (directory/'complete.json').exists(): return base.checked(directory, hid)
    if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('Below10GiB; preserve checkpoints')
    pr = base.ordinary.preprocess(x, y, cv, sites, '__excluded_outer__')
    kwargs = dict(seed=seed, settings=cfg['head_training'], identity=hid, directory=directory, resume=resume,
        stop_at=100 if pilot else None, heartbeat=lambda **kw: beat(head=directory.name, **kw))
    if task == 'utility':
        model, fit = base.geometric.fit(x, y, sites, env, pr, task='utility', **kwargs)
        predict = lambda xx, dd: base.geometric.predict(model, xx, dd, pr); kind = 'bounded_utility'
    else:
        fixed = fixed_rank_scale(y, sites, pr, seed=seed, batch_size=cfg['head_training']['batch_size'], batches=cfg['fixed_scale_batches'])
        model, fit = base.ranked.fit(x, y, sites, env, pr, rank_weight=cfg['rank_weight'], epsilon=cfg['rank_epsilon'], fixed_denominator=fixed, **kwargs)
        predict = lambda xx, dd: base.hurdle.predict(model, xx, dd, pr); kind = 'ranked_risk'
    if pilot: return dict(identity=hid, fit=fit, checkpoint=artifact(directory/'checkpoint.pt'))
    scores = predict(target_x, target_env)
    path = directory/'scores.npz'; temp = path.with_suffix('.tmp.npz')
    np.savez(temp, ids=keys, scores=scores); os.replace(temp, path)
    r = dict(identity=hid, fit=fit, kind=kind, replay_exact=True,
        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'), scores=artifact(path)))
    fresh, state = restored(r); block = len(keys)//2; checks = []
    for offset in (0, block):
        end = offset+min(cfg['replay_rows_per_producer'], block)
        np.testing.assert_array_equal(fresh(target_x[offset:end], target_env[offset:end]), scores[offset:end])
        checks.append(end-offset)
    assert fit['unknown_rows_sampled'] == 0
    r['prefix_replay_rows'] = checks; immutable_json(directory/'complete.json', r)
    beat('head_complete', head=directory.name, seconds=fit['seconds'], parameters=fit['parameters'])
    return r


def train(resume=False, pilot=False):
    cfg, bcfg, ctx, bid, tid, identity = load(); checked_banks(identity); data = ctx[2]; refs = []; pilots = []
    for name, fold, seed, event, design, a, d, x, env, bits, provenance in base.groups(bcfg, ctx, bid):
        train_ids, ids = design['train_ids'], design['held_ids']; tags = tag_inputs(design, data, tid, fold)
        branch = [branch_inputs(name, half, design, data, a, identity) for half in (0, 1)]
        target = np.concatenate([b[0] for b in branch]); target_env = np.concatenate([b[1] for b in branch])
        target_tags = np.repeat([0, 1], len(ids)); keys = np.concatenate([ids*2, ids*2+1]); row_ids = np.tile(ids, 2)
        da, _ = native_errors(d[train_ids].astype(float)+data['origin'][train_ids, None], data['target_eval'][train_ids], data['valid'][train_ids], np.ones(len(train_ids)))
        na, _ = native_errors(a['p'][train_ids].astype(float)+data['origin'][train_ids, None], data['target_eval'][train_ids], data['valid'][train_ids], np.ones(len(train_ids)))
        cv = data['baseline_ade'][train_ids, 1]
        targets = relative_targets(cv, da, na, reference='floor', event=event, easy_cut=design['easy_cut'])
        for arm in ARMS:
            xx = augment(x[train_ids], tags, train_ids, arm); tx = augment(target, target_tags, row_ids, arm)
            for task, y in zip(('utility', 'risk'), targets):
                directory = PRIVATE/'heads'/(name+'_'+arm+'_'+task)
                hid = dict(experiment=identity, group=name, seed=seed, arm=arm, task=task, event=event,
                    fitting_ids_sha256=array_hash(train_ids), feature_train_sha256=array_hash(xx),
                    label_sha256=array_hash(y), inference_sha256=array_hash(keys, tx), lineage=a['lineage'],
                    floor_producers=provenance, tags_sha256=array_hash(tags), env_sha256=array_hash(env[train_ids]))
                beat('head_started', head=directory.name, pilot=pilot, fitting_rows=len(xx))
                r = fit_one(xx, y, data['sites'][train_ids], cv, env[train_ids], tx, target_env, keys,
                    directory=directory, hid=hid, cfg=cfg, seed=seed, task=task, resume=resume, pilot=pilot)
                if pilot: pilots.append(r)
                else: refs.append(artifact(directory/'complete.json'))
            if pilot:
                immutable_json(PRIVATE/'pilot.json', dict(heads=pilots, new_training=True, resumes_inside_budget=True))
                return
        beat('training_group_complete', group=name)
    assert len(refs) == 108
    immutable_json(PRIVATE/'training_complete.json', dict(identity=identity, heads=refs, all_passed=True, updates=216000))


def checked_training(identity):
    r = json.loads((PRIVATE/'training_complete.json').read_text()); assert r['identity'] == identity and len(r['heads']) == 108
    for ref in r['heads']:
        assert artifact(ROOT/ref['path']) == ref
        rr = json.loads((ROOT/ref['path']).read_text()); base.checked((ROOT/ref['path']).parent, rr['identity'])
    return r


def saved_scores(name, arm, task, identity, ids):
    directory = PRIVATE/'heads'/(name+'_'+arm+'_'+task); r = json.loads((directory/'complete.json').read_text())
    assert r['identity']['experiment'] == identity
    keys = np.concatenate([ids*2, ids*2+1]); scores = base.load_scores(directory, r['identity'], keys)
    return scores, r


def decide():
    cfg, bcfg, ctx, bid, tid, identity = load(); checked_banks(identity); training = checked_training(identity); data = ctx[2]; refs = []
    for name, fold, seed, event, design, a, d4, x, env, bits4, provenance in base.groups(bcfg, ctx, bid):
        ids = design['held_ids']; branches = [branch_inputs(name, h, design, data, a, identity) for h in (0, 1)]
        target = np.concatenate([b[0] for b in branches]); tenv = np.concatenate([b[1] for b in branches]); tags = np.repeat([0, 1], len(ids))
        moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
        out = {}; scores = {}; states = {}
        for arm in ARMS:
            values = []
            for task in ('utility', 'risk'):
                v, r = saved_scores(name, arm, task, identity, ids); values.append(v)
                state = torch.load(ROOT/r['artifacts']['checkpoint']['path'], map_location='cpu', weights_only=False); states[(arm, task)] = state
            for h in (0, 1):
                sl = slice(h*len(ids), (h+1)*len(ids)); u, risk = [v[sl] for v in values]
                out[f'half{h}__{arm}'] = safe_choice(u, risk, moving); scores[f'half{h}__{arm}__utility'] = u; scores[f'half{h}__{arm}__risk'] = risk
        for task in ('utility', 'risk'):
            control = states[('global', task)]
            for arm in ('producer', 'placebo'):
                s = states[(arm, task)]
                np.testing.assert_array_equal(s['draws'], control['draws'])
                for field in ('known', 'weights', 'constant'): np.testing.assert_array_equal(s['preprocess'][field], control['preprocess'][field])
                for field in ('mean', 'std'): np.testing.assert_array_equal(s['preprocess'][field][:380], control['preprocess'][field][:380])
                assert s['preprocess']['cost_scale'] == control['preprocess']['cost_scale'] and s['step'] == 2000
                for field in ('label_sha256', 'env_sha256', 'fitting_ids_sha256'):
                    assert s['identity'][field] == control['identity'][field]
                assert s['settings'] == control['settings'] and s['seed'] == control['seed']
                assert sum(v.numel() for v in s['model'].values()) == sum(v.numel() for v in control['model'].values())
                if task == 'risk': assert s['fixed_denominator'] == control['fixed_denominator']
        for variant in ('wrong_tag', 'legacy'):
            vs = []
            for task in ('utility', 'risk'):
                if variant == 'wrong_tag':
                    _, r = saved_scores(name, 'producer', task, identity, ids)
                    tx = augment(target, 1-tags, np.tile(ids, 2), 'producer')
                else:
                    r = json.loads((base.PRIVATE/'fitting/heads'/(name+'_floor_'+task)/'complete.json').read_text())
                    assert r['identity']['experiment'] == bid; tx = target
                predict, _ = restored(r); vs.append(predict(tx, tenv))
            for h in (0, 1):
                sl = slice(h*len(ids), (h+1)*len(ids)); u, risk = [v[sl] for v in vs]
                out[f'half{h}__{variant}'] = safe_choice(u, risk, moving)
                scores[f'half{h}__{variant}__utility'] = u; scores[f'half{h}__{variant}__risk'] = risk
        path = PRIVATE/'decisions'/(name+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                for k, v in dict(ids=ids, **out, **scores).items(): np.testing.assert_array_equal(z[k], v)
        else: np.savez(path, ids=ids, **out, **scores)
        r = dict(identity=identity, group=name, artifact=artifact(path), choices=list(out), sampler_and_preprocess_match=True)
        immutable_json(path.with_suffix('.json'), r); refs.append(artifact(path.with_suffix('.json')))
        beat('decisions_frozen', group=name, policies=len(out))
    assert len(refs) == 18
    immutable_json(PRIVATE/'decisions_complete.json', dict(identity=identity, groups=refs, training=artifact(PRIVATE/'training_complete.json'),
        all_passed=True, new_readout=False))


def ensure_frozen():
    r = json.loads((PRIVATE/'decisions_complete.json').read_text()); assert r['all_passed'] and len(r['groups']) == 18
    for f, sha in r['identity']['bindings'].items():
        if digest(ROOT/f) != sha: raise ValueError('Frozen binding changed')
    checked_banks(r['identity']); checked_training(r['identity'])
    assert artifact(ROOT/r['training']['path']) == r['training']
    for ref in r['groups']:
        assert artifact(ROOT/ref['path']) == ref
        rec = json.loads((ROOT/ref['path']).read_text()); assert rec['identity'] == r['identity']
        assert artifact(ROOT/rec['artifact']['path']) == rec['artifact']


def evaluate(resume=False):
    ensure_frozen(); cfg, bcfg, ctx, bid, tid, identity = load(); data = ctx[2]; refs = []
    for name, fold, seed, event, design, a, d4, x, env, bits4, provenance in base.groups(bcfg, ctx, bid):
        path = PRIVATE/'evaluation'/(name+'.json')
        if path.exists():
            if not resume: raise ValueError('Existing readout requires resume')
            r = json.loads(path.read_text()); assert r['verified'] and r['identity'] == identity; refs.append(artifact(path)); continue
        ids = design['held_ids']; sites = data['sites'][ids]; roster = sorted(set(sites)); moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
        with np.load(PRIVATE/'decisions'/(name+'.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); arrays = {k: z[k].copy() for k in z.files if k != 'ids'}
        choices = [k for k in arrays if len(k.split('__')) == 2]
        for key in choices:
            np.testing.assert_array_equal(arrays[key], independent_choice(arrays[key+'__utility'], arrays[key+'__risk'], moving))
        checks = 0; coordinate_checks = 0
        def errors(p):
            nonlocal coordinate_checks
            absolute = p.astype(float)+data['origin'][ids, None]
            native = native_errors(absolute, data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            alternate = base.cross.independent.coordinate_errors(absolute, data['target_eval'][ids], data['valid'][ids])
            for q, r in zip(native, alternate): base.cross.independent.close(q, r); coordinate_checks += 1
            return native
        def metric(p, reference, mask):
            nonlocal checks
            r = paired_scene_metrics(p[mask], reference[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel', bootstrap_resamples=3000, seed=39271)
            base.cross.independent.check_metric(p, reference, sites, mask, r, cfg); checks += 1
            return r
        d4a, d4f = errors(d4[ids]); n4a, n4f = errors(a['p'][ids]); cv = data['baseline_ade'][ids, 1]
        with np.load(stopped.PRIVATE/'fitting/decisions'/(name+'.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); old_stop = z['floor_both__stop'].copy()
        old_error = np.where(old_stop, n4a, d4a)
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= design['easy_cut']), hard=cv >= design['hard_cut'], complete=data['valid'][ids].all(1))
        anchor = dict(ADE={s: metric(old_error, d4a, m) for s, m in masks.items()},
            FDE=metric(np.where(old_stop, n4f, d4f), d4f, masks['all']))
        old_result = json.loads((stopped.PRIVATE/'fitting/evaluation'/(name+'.json')).read_text())['views']['floor_both__stop']
        assert anchor['ADE'] == old_result['ADE_vs_floor'] and anchor['FDE'] == old_result['FDE_vs_floor']
        views, contrasts, raw = {}, {}, {}
        for half in (0, 1):
            xx, denv, nn, dd, dbits = branch_inputs(name, half, design, data, a, identity)
            na, nf = errors(nn); da, df = errors(dd); selected = {}; selected_fde = {}
            for arm in (*ARMS, 'wrong_tag', 'legacy'):
                key = f'half{half}__{arm}'; use = arrays[key]; err = np.where(use, na, da); fde = np.where(use, nf, df)
                selected[arm], selected_fde[arm] = err, fde
                uy, ry = relative_targets(cv, da, na, reference='floor', event=event, easy_cut=design['easy_cut'])
                known = np.isfinite(cv); up, rp = arrays[key+'__utility'], arrays[key+'__risk']; reliability = {}
                for site in roster:
                    reliability[site] = {}
                    for label, mask in [('population', known & (sites == site)), ('selected', known & (sites == site) & use)]:
                        denom = float(ry[mask, 0].sum()); predicted = float(rp[mask, 0].sum())
                        reliability[site][label] = dict(rows=int(mask.sum()),
                            utility_mae=np.abs(up[mask]-uy[mask]).mean(0).tolist() if mask.any() else None,
                            risk_mae=np.abs(rp[mask]-ry[mask]).mean(0).tolist() if mask.any() else None,
                            realized_harm_ratio=float(ry[mask, 1].sum()/denom) if denom > 0 else None,
                            predicted_harm_ratio=float(rp[mask, 1].sum()/predicted) if predicted > 0 else None)
                views[key] = dict(ADE_vs_floor4={s: metric(err, d4a, m) for s, m in masks.items()},
                    ADE_vs_floor2={s: metric(err, da, m) for s, m in masks.items()},
                    ADE_vs_old_stop4={s: metric(err, old_error, m) for s, m in masks.items()},
                    FDE_vs_floor4=metric(fde, d4f, masks['all']), easy_vs_CV=metric(err, cv, masks['easy']),
                    switch_rate=float(use.mean()), switched_rows=int(use.sum()), unknown_ADE_switches=int((use & ~known).sum()),
                    zero_CV=dict(rows=int((cv == 0).sum()), harmed_rows=int(((cv == 0) & (err > 0)).sum())), reliability=reliability)
            contrasts[f'half{half}'] = {arm: dict(ADE={s: metric(selected['producer'], selected[arm], m) for s, m in masks.items()},
                FDE=metric(selected_fde['producer'], selected_fde[arm], masks['all'])) for arm in ('global', 'placebo', 'wrong_tag', 'legacy')}
            raw[f'half{half}'] = dict(neural_vs_full=metric(na, n4a, masks['all']), floor_vs_full=metric(da, d4a, masks['all']))
        assert len(views) == 10 and checks == 189 and coordinate_checks == 12
        r = dict(identity=identity, group=name, views=views, contrasts=contrasts, raw=raw, anchor=anchor, verified=True,
            saved_decisions_verified=10, coordinate_arrays_verified=12, metric_reductions_verified=checks,
            old_anchor_metrics_exact=5,
            result_source='fresh_run_new_controller_readout_cached_verified_neural_forecasts')
        immutable_json(path, r); refs.append(artifact(path)); beat('group_evaluated', group=name, views=len(views), metrics=checks)
    assert len(refs) == 18
    immutable_json(PRIVATE/'evaluation_complete.json', dict(identity=identity, groups=refs, all_passed=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=('prepare', 'banks', 'pilot', 'train', 'decide', 'evaluate'))
    p.add_argument('--resume', action='store_true'); args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); beat('phase_started', phase=args.phase)
        if args.phase == 'prepare': load()
        elif args.phase == 'banks': make_banks()
        elif args.phase in ('pilot', 'train'): train(resume=args.resume, pilot=args.phase == 'pilot')
        elif args.phase == 'decide': decide()
        else: evaluate(args.resume)
        beat('phase_complete', phase=args.phase)


if __name__ == '__main__': main()
