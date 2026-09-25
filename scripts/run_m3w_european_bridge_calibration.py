"""Fit aligned source-C calibration with fixed A/B models, then freeze readout."""
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
    raise RuntimeError('Native arm64 required before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_bridge_attribution as attribution
from scripts import build_m3w_european_selection_data as adapter
from src.world_model.m3w_bridge_attribution import motion_pair
from src.evaluation.m3w_bridge_risk_calibration import fit, apply, evidence, RULES
from src.evaluation.m3w_native_metrics import native_errors

bridge, base, inc, parent = attribution.bridge, attribution.base, attribution.inc, attribution.parent
artifact, digest, array_hash = attribution.artifact, attribution.digest, attribution.array_hash
immutable_json, require_committed = attribution.immutable_json, attribution.require_committed
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_bridge_calibration_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_bridge_calibration_v1'
CONFIG = 'configs/m3w_european_bridge_calibration_v1.json'
FILES = [CONFIG, 'src/evaluation/m3w_bridge_risk_calibration.py',
    'scripts/run_m3w_european_bridge_calibration.py', 'scripts/evaluate_m3w_european_bridge_calibration.py',
    'tests/test_m3w_bridge_risk_calibration.py',
    'outputs/publication_readiness_2026_09/european_bridge_calibration_v1/registration.md']


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    assert cfg['rules'] == list(RULES) and cfg['seeds'] == [17, 29, 43]
    assert (cfg['groups'], cfg['calibration_maps'], cfg['decision_views']) == (18, 72, 288)
    assert cfg['risk_budget'] == .02 and cfg['easy_net_limit_percent'] == 2.
    assert not any(cfg[k] for k in ('new_neural_training', 'reserved_calibration_access',
        'confirmation_access', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    _, bank, pid, aid = attribution.registration()
    assert digest(attribution.PUBLIC/'summary_metrics.json') == cfg['parent_summary_sha256']
    verify = json.loads((attribution.PUBLIC/'verification.json').read_text())
    assert verify['all_passed']
    for p, h in verify['artifacts'].items(): assert digest(attribution.PUBLIC/p) == h
    for p, h in verify['source_bindings'].items(): assert digest(ROOT/p) == h
    value = dict(bindings={p: digest(ROOT/p) for p in FILES}, parent=aid, rosters=aid['rosters'],
        parent_verification=artifact(attribution.PUBLIC/'verification.json'),
        parent_summary=artifact(attribution.PUBLIC/'summary_metrics.json'))
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, value)
    elif json.loads(path.read_text()) != value: raise ValueError('Registered implementation changed')
    if not create: require_committed(path)
    return cfg, bank, pid, value


def restored(pair, group, family, task):
    root = bridge.PRIVATE if pair == 'full' else attribution.PRIVATE
    directory = root/('heads' if family == 'neural' else 'ridge')/(group+'_'+task)
    receipt = json.loads((directory/'complete.json').read_text())
    for ref in receipt['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
    if family == 'neural':
        pred, state = inc.prior.previous.restored(receipt)
    else:
        state = torch.load(directory/'ridge.pt', map_location='cpu', weights_only=False)
        def pred(x, env):
            return inc.prior.bounded_ridge_scores(base.ordinary.predict_ridge(state['head'], x,
                np.zeros(len(x), bool), state['preprocess']), env, 'utility' if task == 'utility' else 'risk')
    return pred, state, artifact(directory/'complete.json')


def save_arrays(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
            for k, value in arrays.items(): np.testing.assert_array_equal(z[k], value)
    else: parent.atomic_npz(path, **arrays)


def calibrate(cfg, bank, pid, identity):
    trained = attribution.checked_training(identity['parent'])
    groups = {g['group']: g for g in trained['groups']}
    inc.ensure_frozen(); _, bcfg, ctx, bid, _, iid = inc.load(); data = ctx[2]
    assert iid['rosters'] == identity['rosters']
    selection_data = adapter.load(parent, pid); refs = []; source_refs = []
    assert not set(selection_data['sites']) & set(sum(identity['rosters'], []))
    for easy, all_event in bridge.source_pairs(bcfg, ctx, bid):
        a, seed, held = easy['fold'], easy['seed'], easy['ids']
        for b in range(3):
            if a == b: continue
            if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('Disk reserve reached; retain receipts')
            c = next(k for k in range(3) if k not in (a, b)); name = f'fold{a}_seed{seed}_controller{b}'; g = groups[name]
            take = np.isin(data['sites'][held], identity['rosters'][c]); ids = held[take]
            sites = data['sites'][ids]
            assert set(sites) == set(identity['rosters'][c])
            assert not set(sites) & (set(g['producer_roster']) | set(g['controller_roster']))
            cvp = easy['cv'][take]; damp = base.baseline_numpy(data['history'][ids], 3)-data['origin'][ids, None]
            r, p = easy['prediction'][take], all_event['prediction'][take]
            bits = np.column_stack([v[k][take] for k in ('old', 'bits') for v in (easy, all_event)])
            x, env = bridge.features(data['geometry'][ids], cvp, r, p, bits)
            pairs = dict(full=(x, env, r, p), motion_only=motion_pair(data['geometry'][ids], cvp, damp,
                easy['bits'][take], all_event['bits'][take]))
            tpairs, _ = attribution.selection(selection_data, bank, g)
            moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
            for pair, (x, env, r, p) in pairs.items():
                beat('source_C_scoring', group=name, pair=pair, rows=len(ids), C=c)
                tx, tenv, _, _ = tpairs[pair]; values = {}; models = []
                for family in cfg['families']:
                    for task in attribution.TASKS:
                        pred, state, ref = restored(pair, name, family, task)
                        assert state['preprocess']['training_sites'] == sorted(g['controller_roster'])
                        score = pred(x, env); n = min(len(tx), cfg['replay_rows'])
                        old = attribution.scores(pair, name)[(0 if family == 'neural' else 2)+(task == 'all_risk')]
                        np.testing.assert_array_equal(pred(tx[:n], tenv[:n]), old[:n])
                        values[family+'__'+task] = score; models.append(ref)
                def error(prediction):
                    return native_errors(prediction.astype(float)+data['origin'][ids, None], data['target_eval'][ids],
                        data['valid'][ids], np.ones(len(ids)))[0]
                labels = dict(cv=data['baseline_ade'][ids, 1], reference=error(r), candidate=error(p))
                directory = PRIVATE/'source'/(name+'_'+pair)
                save_arrays(directory/'scores.npz', dict(ids=ids, sites=sites, moving=moving, envelope=env, **values))
                save_arrays(directory/'labels.npz', labels)
                source = dict(identity=identity, group=g, pair=pair, calibration_fold=c,
                    calibration_roster=identity['rosters'][c], input_sha256=array_hash(x), ids_sha256=array_hash(ids),
                    models=models, rows=len(ids), inference_used_future=False,
                    arrays=dict(scores=artifact(directory/'scores.npz'), labels=artifact(directory/'labels.npz')))
                immutable_json(directory/'receipt.json', source); source_refs.append(artifact(directory/'receipt.json'))
                for family in cfg['families']:
                    u, m = [values[family+'__'+task] for task in attribution.TASKS]
                    fitted = fit(u, m, moving, env, **labels, sites=sites, easy_cut=g['easy_cut'], grid=cfg['grid'])
                    loo = {}
                    for site in sorted(set(sites)):
                        keep = sites != site; test = ~keep
                        inner = fit(u[keep], m[keep], moving[keep], env[keep], **{k:v[keep] for k,v in labels.items()},
                            sites=sites[keep], easy_cut=g['easy_cut'], grid=cfg['grid'])
                        rule = inner['rules']['selected_risk_grid']; use = apply(u[test], m[test], moving[test], env[test], rule)
                        loo[site] = dict(rule=rule, held_out=evidence(use, **{k:v[test] for k,v in labels.items()},
                            sites=sites[test], easy_cut=g['easy_cut']), diagnostic_not_used_for_final_rule=True)
                    value = dict(identity=identity, group=g, pair=pair, family=family, source=artifact(directory/'receipt.json'),
                        fitted=fitted, leave_one_source_C=loo, selection_labels_read=False,
                        independently_calibrated=False, result_source='fresh_run_source_C_calibration')
                    path = PRIVATE/'maps'/(name+'_'+pair+'_'+family+'.json'); immutable_json(path, value)
                    refs.append(artifact(path)); beat('map_fitted', group=name, pair=pair, family=family)
    assert len(refs) == 72 and len(source_refs) == 36
    immutable_json(PRIVATE/'calibration_complete.json', dict(identity=identity, maps=refs, source=source_refs,
        groups=list(groups.values()), all_passed=True, selection_labels_read=False, calibration_role='opened_source_C'))
    immutable_json(PUBLIC/'calibration_receipt.json', dict(manifest=artifact(PRIVATE/'calibration_complete.json'),
        maps=72, new_neural_training=False, cached_predictors=True, result_source='fresh_run_source_C_calibration',
        reserved_calibration_opened=False, confirmation_opened=False))


def checked_calibration(identity):
    done = json.loads((PRIVATE/'calibration_complete.json').read_text())
    assert done['identity'] == identity and done['all_passed'] and len(done['maps']) == 72
    for ref in done['maps']+done['source']:
        assert artifact(ROOT/ref['path']) == ref
    for ref in done['source']:
        r = json.loads((ROOT/ref['path']).read_text())
        for a in r['arrays'].values(): assert artifact(ROOT/a['path']) == a
    return done


def decide(cfg, bank, pid, identity):
    done = checked_calibration(identity); data = adapter.load(parent, pid); refs = []
    moving = np.linalg.norm(data['history'][:, -1]-data['history'][:, -2], axis=1) > 0
    for g in done['groups']:
        pairs, _ = attribution.selection(data, bank, g)
        for pair, (x, env, _, _) in pairs.items():
            values = attribution.scores(pair, g['group'])
            for family in cfg['families']:
                name = g['group']+'_'+pair+'_'+family
                mapping = PRIVATE/'maps'/(name+'.json'); fitted = json.loads(mapping.read_text())['fitted']
                u, m = values[:2] if family == 'neural' else values[2:]
                out = {k: apply(u, m, moving, env, rule) for k, rule in fitted['rules'].items()}
                with np.load(attribution.PRIVATE/'decisions'/(g['group']+'_'+pair+'.npz'), allow_pickle=False) as z:
                    np.testing.assert_array_equal(out['none'], z[family])
                path = PRIVATE/'decisions'/(name+'.npz'); save_arrays(path, out)
                receipt = dict(identity=identity, group=g, pair=pair, family=family, artifact=artifact(path),
                    map=artifact(mapping), input_sha256=array_hash(x), score_sha256=array_hash(u, m),
                    rows=len(x), labels_read=False, raw_parent_policy_exact=True)
                immutable_json(path.with_suffix('.json'), receipt); refs.append(artifact(path.with_suffix('.json')))
        beat('decisions_frozen', group=g['group'])
    immutable_json(PRIVATE/'decisions_complete.json', dict(identity=identity, decisions=refs,
        calibration=artifact(PRIVATE/'calibration_complete.json'), all_passed=True))
    immutable_json(PUBLIC/'decision_freeze.json', dict(identity=identity, views=288, maps=72,
        manifest=artifact(PRIVATE/'decisions_complete.json'), outcomes_read=False,
        role='reused_opened_model_selection', reserved_calibration_opened=False, confirmation_opened=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['register', 'calibrate', 'decide', 'evaluate']); args = p.parse_args()
    for d in (PRIVATE, PUBLIC): d.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, bank, pid, identity = registration(create=args.phase == 'register')
        if args.phase == 'calibrate': calibrate(cfg, bank, pid, identity)
        elif args.phase == 'decide': decide(cfg, bank, pid, identity)
        elif args.phase == 'evaluate':
            from scripts.evaluate_m3w_european_bridge_calibration import evaluate
            evaluate(sys.modules[__name__], cfg, bank, pid, identity)


if __name__ == '__main__': main()
