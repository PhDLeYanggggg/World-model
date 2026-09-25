"""Matched CV/floor target learning using source-cross-fitted protected floors."""
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
from scripts import run_m3w_european_cross_moment as cross
from scripts.report_m3w_european_cross_moment import verify as verify_cross
from src.world_model import m3w_native_gain_harm as ordinary
from src.world_model import m3w_geometric_cost_head as geometric
from src.world_model import m3w_cross_moment_rank as ranked
from src.world_model import m3w_hurdle_risk as hurdle
from src.world_model.m3w_floor_relative import assert_producer_exclusion, matched_features, relative_targets, fixed_rank_scale
from src.world_model.m3w_european_source_forecast import baseline_numpy, fit_design
from src.world_model.m3w_european_source_intervention import causal_cost_features, paired_cost_labels
from src.world_model.m3w_european_conditional_risk import event_labels, pointwise_rule
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_floor_relative_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_floor_relative_v1'
CONFIG = 'configs/m3w_european_floor_relative_v1.json'
DIAG = ROOT/'outputs/publication_readiness_2026_09/european_floor_opportunity_v1'
FILES = (CONFIG, 'scripts/run_m3w_european_floor_relative.py', 'src/world_model/m3w_floor_relative.py',
         'tests/test_m3w_floor_relative.py', 'tests/test_m3w_floor_relative_protocol.py',
         'outputs/publication_readiness_2026_09/european_floor_relative_v1/registration.md')
digest, immutable_json, array_hash = cross.digest, cross.immutable_json, cross.array_hash


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def validate(cfg):
    for key, value in dict(modes=['batch', 'fitting'], seeds=[17, 29, 43], events=['all', 'easy'],
                           references=['cv', 'floor'], risk_budget=.02, total_new_heads=234,
                           total_updates=468000, bootstrap_resamples=3000, bootstrap_seed=39271).items():
        if cfg[key] != value:
            raise ValueError('Changed registered comparison: '+key)
    if cfg['variants'] != dict(cv_targets=['cv', 'cv'], floor_utility=['floor', 'cv'],
                              floor_risk=['cv', 'floor'], floor_both=['floor', 'floor']):
        raise ValueError('Every factorial target comparison required')
    if any(cfg[k] for k in ('new_forecaster_training', 'threshold_refit', 'calibration_refit',
                           'reserved_roles_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled')):
        raise ValueError('Source development only; no deployment')


def load(mode):
    cfg = json.loads((ROOT/CONFIG).read_text()); validate(cfg)
    cross.configure(mode); ctx = cross.load(); previous = verify_cross()
    if digest(DIAG/'summary_metrics.json') != cfg['prior_floor_summary_sha256']:
        raise ValueError('Prior diagnostic changed')
    if cfg['head_training'] != ctx[0]['head_training']:
        raise ValueError('Matched fixed training budget required')
    identity = dict(mode=mode, bindings={f: digest(ROOT/f) for f in FILES},
        cross_analysis_sha256=digest(cross.PUBLIC/'analysis.json'),
        cross_identity_sha256=digest(cross.PRIVATE/'identity.json'),
        cross_decisions_sha256=digest(cross.PRIVATE/'decisions_complete.json'),
        common_parent_sha256=digest(cross.parent.PRIVATE/'identity.json'))
    immutable_json(PRIVATE/mode/'identity.json', identity)
    return cfg, ctx, previous, identity


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def checked(directory, hid):
    r = json.loads((directory/'complete.json').read_text())
    if r['identity'] != hid or not r['fit']['complete'] or r['fit']['step'] != 2000 or not r['replay_exact']:
        raise ValueError('Incomplete or changed fitted head')
    for ref in r['artifacts'].values():
        if digest(ROOT/ref['path']) != ref['sha256']:
            raise ValueError('Changed fitted artifact')
    return r


def load_scores(directory, hid, ids):
    r = checked(directory, hid)
    with np.load(ROOT/r['artifacts']['scores']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(ids, z['ids'])
        return z['scores'].copy()


def fit_head(x, y, sites, cv, envelope, same, target_x, target_envelope, target_same, ids,
             *, directory, hid, cfg, seed, kind, mode, resume, pilot=False):
    if (directory/'complete.json').exists():
        return checked(directory, hid)
    if shutil.disk_usage(PRIVATE).free < 10*1024**3:
        raise OSError('Below10GiB: retain checkpoints')
    pr = ordinary.preprocess(x, y, cv, sites, '__excluded_outer__')
    kwargs = dict(seed=seed, settings=cfg['head_training'], identity=hid, directory=directory,
                  resume=resume, stop_at=100 if pilot else None,
                  heartbeat=lambda **kw: beat(head=directory.name, mode=mode, **kw))
    beat('head_started', head=directory.name, mode=mode, kind=kind, fitting_rows=len(x))
    if kind == 'ordinary_utility':
        model, fit = ordinary.fit_neural(x, y, sites, same, pr, arm='mse', **kwargs)
        predict = lambda model, xx, dd, ss: ordinary.predict_neural(model, xx, ss, pr)
    elif kind == 'bounded_utility':
        model, fit = geometric.fit(x, y, sites, envelope, pr, task='utility', **kwargs)
        predict = lambda model, xx, dd, ss: geometric.predict(model, xx, dd, pr)
    else:
        fixed = (fixed_rank_scale(y, sites, pr, seed=seed, batch_size=cfg['head_training']['batch_size'],
                    batches=cfg['fixed_scale_batches']) if mode == 'fitting' else None)
        model, fit = ranked.fit(x, y, sites, envelope, pr, rank_weight=cfg['rank_weight'],
            epsilon=cfg['rank_epsilon'], fixed_denominator=fixed, **kwargs)
        predict = lambda model, xx, dd, ss: hurdle.predict(model, xx, dd, pr)
    if pilot:
        immutable_json(PRIVATE/'pilot.json', dict(identity=hid, fit=fit, checkpoint=artifact(directory/'checkpoint.pt')))
        return None
    pred = predict(model, target_x, target_envelope, target_same)
    state = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
    if state['identity'] != hid or state['step'] != 2000:
        raise ValueError('Wrong checkpoint identity')
    if kind == 'ordinary_utility':
        replay = ordinary.build_head(cfg['head_training']['width'], pr, seed)
    elif kind == 'bounded_utility':
        replay = geometric.initialize_head(cfg['head_training']['width'], pr, seed, 'utility', state['mean_envelope'])
    else:
        replay = hurdle.initialize(pr, state['prior'], cfg['head_training']['width'], seed)
    replay.load_state_dict(state['model'])
    n = min(cfg['replay_rows'], len(ids))
    np.testing.assert_array_equal(predict(replay, target_x[:n], target_envelope[:n], target_same[:n]), pred[:n])
    path = directory/'scores.npz'; temp = path.with_suffix('.tmp.npz')
    np.savez(temp, ids=ids, scores=pred); os.replace(temp, path)
    r = dict(identity=hid, fit=fit, kind=kind, replay_exact=True, replay_rows=n,
             artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'), scores=artifact(path)))
    immutable_json(directory/'complete.json', r)
    beat('head_complete', mode=mode, head=directory.name, steps=fit['step'], seconds=fit['seconds'])
    return r


def shared_identity(identity):
    return dict(bindings=identity['bindings'], common_parent_sha256=identity['common_parent_sha256'])


def inner_jobs(cfg, ctx):
    pid = ctx[3]['parent_identity']['geometric_identity']['old_identity']['producer_identity']
    data = ctx[2]
    for fold in range(3):
        halves = pid['halves'][str(fold)]
        parent_sites = set(halves[0]+halves[1]); outer = set(data['sites'])-parent_sites
        for seed in cfg['seeds']:
            for side in range(2):
                fitting, target = set(halves[side]), set(halves[1-side])
                assert_producer_exclusion(fitting, target, outer, parent_sites)
                train_ids = np.flatnonzero(np.isin(data['sites'], sorted(fitting)))
                ids = np.flatnonzero(np.isin(data['sites'], sorted(target)))
                yield fold, seed, side, train_ids, ids, dict(training_sites=sorted(fitting),
                    target_sites=sorted(target), outer_sites=sorted(outer), fitting_parent=sorted(parent_sites),
                    train_ids_sha256=array_hash(train_ids), target_ids_sha256=array_hash(ids))


def train_inner(mode, *, resume=False, pilot=False):
    cfg, ctx, _, identity = load(mode); data = ctx[2]
    b = baseline_numpy(data['history'], 1)-data['origin'][:, None]
    d = baseline_numpy(data['history'], 3)-data['origin'][:, None]
    x, _ = causal_cost_features(data['geometry'], b, d)
    envelope = geometric.rollout_envelope(b, d); same = np.all(b == d, axis=(1, 2))
    moving = np.linalg.norm(np.diff(data['history'], axis=1), axis=2).sum(1) > 0
    archives = []
    for fold, seed, side, train, ids, lineage in inner_jobs(cfg, ctx):
        name = f'fold{fold}_half{side}_seed{seed}'
        cv = data['baseline_ade'][train, 1]
        da, _ = native_errors(d[train].astype(float)+data['origin'][train, None],
            data['target_eval'][train], data['valid'][train], np.ones(len(train)))
        utility = paired_cost_labels(cv, da)
        easy_cut = float(np.quantile(cv[np.isfinite(cv) & (cv > 0)], .25))
        directory = PRIVATE/'inner_shared'/name
        base = dict(lineage=lineage, feature_train_sha256=array_hash(x[train]),
                    target_ids_sha256=array_hash(ids), seed=seed)
        hid = dict(experiment=shared_identity(identity), **base, task='utility', target_sha256=array_hash(utility))
        r = fit_head(x[train], utility, data['sites'][train], cv, envelope[train], same[train],
            x[ids], envelope[ids], same[ids], ids, directory=directory, hid=hid, cfg=cfg, seed=seed,
            kind='ordinary_utility', mode='shared', resume=resume, pilot=pilot)
        if pilot:
            return
        us = load_scores(directory, hid, ids); u = us[:, 0]-us[:, 1]
        for event in cfg['events']:
            y = event_labels(cv, utility[:, 1], easy_cut=easy_cut, event=event)
            rd = PRIVATE/mode/'inner_risk'/(name+'_'+event)
            rhid = dict(experiment=identity, **base, event=event, target_sha256=array_hash(y), easy_cut=easy_cut)
            fit_head(x[train], y, data['sites'][train], cv, envelope[train], same[train],
                x[ids], envelope[ids], same[ids], ids, directory=rd, hid=rhid, cfg=cfg, seed=seed,
                kind='ranked_risk', mode=mode, resume=resume)
            risk = load_scores(rd, rhid, ids)
            bits = pointwise_rule(u, risk, moving[ids], budget=cfg['risk_budget'], support_available=True)
            path = PRIVATE/mode/'inner_decisions'/(name+'_'+event+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
            ref = dict(identity=identity, lineage=lineage, utility=artifact(directory/'complete.json'),
                       risk=artifact(rd/'complete.json'), decision_sha256=array_hash(ids, bits), event=event)
            if path.exists():
                old = json.loads(path.with_suffix('.json').read_text())
                if old['provenance'] != ref or old['artifact'] != artifact(path):
                    raise ValueError('Changed cross-fitted floor decision')
            else:
                np.savez(path, ids=ids, switch=bits)
                immutable_json(path.with_suffix('.json'), dict(provenance=ref, artifact=artifact(path)))
            archives.append(artifact(path.with_suffix('.json')))
        beat('inner_group_complete', mode=mode, group=name)
    assert len(archives) == 36
    immutable_json(PRIVATE/mode/'inner_complete.json', dict(identity=identity, archives=archives,
        source_exclusion_pass=True, utility_heads=18, risk_heads=36, new_training=True))


def floor_bits(cfg, ctx, identity, fold, seed, event, design):
    data = ctx[2]; mode = identity['mode']; n = len(data['sites'])
    complete = json.loads((PRIVATE/mode/'inner_complete.json').read_text())
    if complete['identity'] != identity or not complete['source_exclusion_pass']:
        raise ValueError('Source-excluded inner floor bank required')
    bits, seen = np.zeros(n, bool), np.zeros(n, int)
    provenance = []
    for f, s, side, train, ids, lineage in inner_jobs(cfg, ctx):
        if (f, s) != (fold, seed):
            continue
        path = PRIVATE/mode/'inner_decisions'/f'fold{fold}_half{side}_seed{seed}_{event}.json'
        if artifact(path) not in complete['archives']:
            raise ValueError('Inner decision receipt not bound by completed bank')
        r = json.loads(path.read_text()); p = r['provenance']
        if p['identity'] != identity or p['lineage'] != lineage or artifact(ROOT/r['artifact']['path']) != r['artifact']:
            raise ValueError('Cross-fitted floor lineage changed')
        for task in ('utility', 'risk'):
            if artifact(ROOT/p[task]['path']) != p[task]:
                raise ValueError('Floor producer changed')
            head = json.loads((ROOT/p[task]['path']).read_text())
            for a in head['artifacts'].values():
                if artifact(ROOT/a['path']) != a:
                    raise ValueError('Inner checkpoint or scores changed')
        with np.load(ROOT/r['artifact']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            if array_hash(ids, z['switch']) != p['decision_sha256']:
                raise ValueError('Cross-fitted decision mismatch')
            bits[ids] = z['switch']; seen[ids] += 1
        provenance.append(artifact(path))
    held = design['held_ids']; refs = cross.read_decisions(ctx[3])
    with np.load(ROOT/refs[f'damping097_fold{fold}_seed{seed}_{event}']['path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], held); bits[held] = z['hurdle_original']; seen[held] += 1
    if not (seen == 1).all():
        raise ValueError('Each row must have exactly one correctly excluded floor producer')
    return bits, provenance


def groups(cfg, ctx, identity):
    data = ctx[2]
    for _, candidate, fold, seed, design in cross.parent.jobs(ctx[1], data, ctx[3]['parent_identity']):
        if candidate != 'neural':
            continue
        a = cross.parent.assemble(candidate, fold, seed, design, data, ctx[3]['parent_identity'])
        damping = baseline_numpy(data['history'], 3)-data['origin'][:, None]
        for event in cfg['events']:
            bits, provenance = floor_bits(cfg, ctx, identity, fold, seed, event, design)
            d = np.where(bits[:, None, None], damping, a['b'])
            x, envelope = matched_features(data['geometry'], a['b'], d, a['p'], bits)
            yield f'fold{fold}_seed{seed}_{event}', fold, seed, event, design, a, d, x, envelope, bits, provenance


def train_main(mode, *, resume=False):
    cfg, ctx, _, identity = load(mode); data = ctx[2]; records = []
    for name, fold, seed, event, design, a, d, x, envelope, bits, provenance in groups(cfg, ctx, identity):
        train, ids = design['train_ids'], design['held_ids']
        cv = data['baseline_ade'][train, 1]
        da, _ = native_errors(d[train].astype(float)+data['origin'][train, None], data['target_eval'][train],
                               data['valid'][train], np.ones(len(train)))
        na, _ = native_errors(a['p'][train].astype(float)+data['origin'][train, None], data['target_eval'][train],
                               data['valid'][train], np.ones(len(train)))
        saved_states = []
        for reference in cfg['references']:
            targets = relative_targets(cv, da, na, reference=reference, event=event, easy_cut=design['easy_cut'])
            for task, y in zip(('utility', 'risk'), targets):
                hid = dict(experiment=identity, neural_lineage=a['lineage'], floor_producers=provenance,
                    group=name, reference=reference, task=task, feature_train_sha256=array_hash(x[train]),
                    envelope_train_sha256=array_hash(envelope[train]), target_sha256=array_hash(y),
                    floor_decision_sha256=array_hash(np.arange(len(bits)), bits))
                directory = PRIVATE/mode/'heads'/f'{name}_{reference}_{task}'
                r = fit_head(x[train], y, data['sites'][train], cv, envelope[train], np.zeros(len(train), bool),
                    x[ids], envelope[ids], np.zeros(len(ids), bool), ids, directory=directory, hid=hid,
                    cfg=cfg, seed=seed, kind='bounded_utility' if task == 'utility' else 'ranked_risk',
                    mode=mode, resume=resume)
                records.append(artifact(directory/'complete.json'))
                saved_states.append(torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False))
        for state in saved_states[1:]:
            np.testing.assert_array_equal(state['draws'], saved_states[0]['draws'])
            for field in ('mean', 'std', 'known', 'weights'):
                np.testing.assert_array_equal(state['preprocess'][field], saved_states[0]['preprocess'][field])
            assert state['preprocess']['cost_scale'] == saved_states[0]['preprocess']['cost_scale']
        beat('main_group_complete', mode=mode, group=name, matched_sampler_and_preprocess=True)
    assert len(records) == 72
    immutable_json(PRIVATE/mode/'training_complete.json', dict(identity=identity, heads=records,
        checkpoint_replays=72, matched_sampler_groups=18, new_heads=72, new_updates=144000))


def head_scores(mode, name, ids, identity):
    directory = PRIVATE/mode/'heads'/name
    manifest = json.loads((PRIVATE/mode/'training_complete.json').read_text())
    if manifest['identity'] != identity or artifact(directory/'complete.json') not in manifest['heads']:
        raise ValueError('Head not bound by completed training manifest')
    r = json.loads((directory/'complete.json').read_text())
    if r['identity']['experiment'] != identity:
        raise ValueError('Wrong main experiment')
    return load_scores(directory, r['identity'], ids)


def decide(mode):
    cfg, ctx, _, identity = load(mode); data = ctx[2]
    record = json.loads((PRIVATE/mode/'training_complete.json').read_text())
    assert record['identity'] == identity and len(record['heads']) == 72
    records = []
    for name, fold, seed, event, design, a, d, x, envelope, floor, provenance in groups(cfg, ctx, identity):
        ids = design['held_ids']; moving = np.linalg.norm(np.diff(data['history'][ids], axis=1), axis=2).sum(1) > 0
        scores = {ref: {task: head_scores(mode, name+'_'+ref+'_'+task, ids, identity)
                        for task in ('utility', 'risk')} for ref in cfg['references']}
        choices = {}
        for variant, (u, r) in cfg['variants'].items():
            utility = scores[u]['utility'][:, 0]-scores[u]['utility'][:, 1]
            choices[variant] = pointwise_rule(utility, scores[r]['risk'], moving, budget=cfg['risk_budget'], support_available=True)
        refs = cross.read_decisions(ctx[3])
        with np.load(ROOT/refs['neural_'+name]['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); choices['old_rebased'] = z['hurdle_original'].copy()
        path = PRIVATE/mode/'decisions'/(name+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids)
                for k, v in choices.items(): np.testing.assert_array_equal(z[k], v)
        else:
            np.savez(path, ids=ids, floor=floor[ids], **choices)
        records.append(dict(group=name, artifact=artifact(path),
            choices_sha256={k: array_hash(ids, v) for k, v in choices.items()}))
        beat('decisions_frozen', mode=mode, group=name)
    immutable_json(PRIVATE/mode/'decisions_complete.json', dict(identity=identity, groups=records,
        training_manifest=artifact(PRIVATE/mode/'training_complete.json'),
        inner_manifest=artifact(PRIVATE/mode/'inner_complete.json'),
        new_outcomes_read=False, all_passed=True))


def ensure_both_frozen():
    for mode in ('batch', 'fitting'):
        r = json.loads((PRIVATE/mode/'decisions_complete.json').read_text())
        if not r['all_passed'] or len(r['groups']) != 18:
            raise ValueError('Both complete decision banks required before evaluation')
        for path, sha in r['identity']['bindings'].items():
            if digest(ROOT/path) != sha: raise ValueError('Changed frozen binding')
        for field in ('training_manifest', 'inner_manifest'):
            if artifact(ROOT/r[field]['path']) != r[field]:
                raise ValueError('Training or inner manifest changed after decisions')
        for row in r['groups']:
            if artifact(ROOT/row['artifact']['path']) != row['artifact']:
                raise ValueError('Changed decision bank')


def evaluate(mode, *, resume=False):
    ensure_both_frozen()
    cfg, ctx, _, identity = load(mode); data = ctx[2]
    manifests = json.loads((PRIVATE/mode/'decisions_complete.json').read_text())
    archives = {r['group']: r for r in manifests['groups']}; results = []
    for name, fold, seed, event, design, a, d, x, envelope, floor, provenance in groups(cfg, ctx, identity):
        path = PRIVATE/mode/'evaluation'/(name+'.json')
        if path.exists():
            if not resume: raise ValueError('Existing evaluation requires --resume')
            r = json.loads(path.read_text()); assert r['identity'] == identity and r['verified']
            results.append(artifact(path)); continue
        ids = design['held_ids']; sites = data['sites'][ids]; roster = sorted(set(sites))
        with np.load(ROOT/archives[name]['artifact']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); np.testing.assert_array_equal(floor[ids], z['floor'])
            choices = {k: z[k].copy() for k in (*cfg['variants'], 'old_rebased')}
        scores = {ref: {task: head_scores(mode, name+'_'+ref+'_'+task, ids, identity)
                        for task in ('utility', 'risk')} for ref in cfg['references']}
        moving = np.linalg.norm(np.diff(data['history'][ids], axis=1), axis=2).sum(1) > 0
        for variant, (u, r) in cfg['variants'].items():
            utility = scores[u]['utility'][:, 0]-scores[u]['utility'][:, 1]
            m = scores[r]['risk']; limit = np.asarray(cfg['risk_budget'], dtype=m.dtype)*m[:, 0]
            separate = np.array([bool(moving[i] and utility[i] > 0 and m[i, 0] > 0 and m[i, 1] <= limit[i]) for i in range(len(ids))])
            np.testing.assert_array_equal(separate, choices[variant])
        # Only after the saved causal switches are verified, inspect held-out development labels.
        costs = []
        for pred in (d[ids].astype(float)+data['origin'][ids, None], a['p'][ids].astype(float)+data['origin'][ids, None]):
            error = native_errors(pred, data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            separate = cross.independent.coordinate_errors(pred, data['target_eval'][ids], data['valid'][ids])
            for v, w in zip(error, separate): cross.independent.close(v, w)
            costs.append(error)
        (da, df), (na, nf) = costs; cv = data['baseline_ade'][ids, 1]
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= design['easy_cut']),
                     hard=cv >= design['hard_cut'], complete=data['valid'][ids].all(1))
        def metric(e, r, mask):
            v = paired_scene_metrics(e[mask], r[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
                bootstrap_resamples=cfg['bootstrap_resamples'], seed=cfg['bootstrap_seed'])
            cross.independent.check_metric(e, r, sites, mask, v, cfg)
            return v
        selected = {k: np.where(v, na, da) for k, v in choices.items()}
        zero = np.isfinite(cv) & (cv == 0); views = {}
        for variant, use in choices.items():
            e = selected[variant]
            views[variant] = dict(ADE_vs_floor={s: metric(e, da, m) for s, m in masks.items()},
                ADE_vs_CV={s: metric(e, cv, masks[s]) for s in ('all', 'easy', 'hard')},
                ADE_vs_matched_cv_target={s: metric(e, selected['cv_targets'], m) for s, m in masks.items()},
                FDE_vs_floor=metric(np.where(use, nf, df), df, masks['all']),
                switch_rate=float(use.mean()), switched_rows=int(use.sum()),
                unknown_ADE_switches=int((use & ~np.isfinite(na)).sum()),
                zero_CV=dict(rows=int(zero.sum()), harmed_rows=int((e[zero] > 0).sum())))
        old_path = ROOT/'data/stage_cvpr2027_experiments/european_floor_opportunity_v1'/mode/'groups'/(name+'.json')
        old_receipt = json.loads((DIAG/'completion_checks.json').read_text())['group_receipts'][mode][name]
        if artifact(old_path) != old_receipt:
            raise ValueError('Preceding diagnostic group changed')
        previous = json.loads(old_path.read_text())
        for s in masks:
            assert views['old_rebased']['ADE_vs_floor'][s] == previous['metrics']['rebased_neural']['ADE_vs_floor'][s]
        assert views['old_rebased']['FDE_vs_floor'] == previous['metrics']['rebased_neural']['FDE_vs_floor']['all']
        r = dict(identity=identity, group=name, views=views, verified=True, matched_old_metrics=5,
            saved_decision_arrays_verified=4, independent_coordinate_arrays=4, independent_metric_reductions=60,
            result_source='fresh_run_training_and_evaluation', deployment_changed=False)
        immutable_json(path, r); results.append(artifact(path))
        beat('group_evaluated', mode=mode, group=name)
    assert len(results) == 18
    immutable_json(PRIVATE/mode/'evaluation_complete.json', dict(identity=identity, groups=results, all_passed=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('batch', 'fitting'), required=True)
    parser.add_argument('--phase', choices=('prepare', 'pilot', 'inner', 'train', 'decide', 'evaluate'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat('phase_started', phase=args.phase, mode=args.mode)
        if args.phase == 'prepare': load(args.mode)
        elif args.phase in ('inner', 'pilot'): train_inner(args.mode, resume=args.resume, pilot=args.phase == 'pilot')
        elif args.phase == 'train': train_main(args.mode, resume=args.resume)
        elif args.phase == 'decide': decide(args.mode)
        else: evaluate(args.mode, resume=args.resume)
        beat('phase_complete', phase=args.phase, mode=args.mode)


if __name__ == '__main__':
    main()
