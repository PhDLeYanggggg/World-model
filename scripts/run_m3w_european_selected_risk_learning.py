"""Source-only matched event-risk learning with frozen query decision budgets."""
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
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_bridge_calibration as previous
from src.world_model import m3w_selected_risk_learning as method
from src.world_model.m3w_bridge_attribution import query_groups
from src.world_model.m3w_dual_event_bridge import choices
from src.evaluation.m3w_native_metrics import native_errors
base = previous.base
artifact, digest, immutable_json, array_hash = previous.artifact, previous.digest, previous.immutable_json, previous.array_hash
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selected_risk_learning_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_selected_risk_learning_v1'
CONFIG = 'configs/m3w_european_selected_risk_learning_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_selected_risk_learning.py',
    'scripts/run_m3w_european_selected_risk_learning.py', 'scripts/evaluate_m3w_european_selected_risk_learning.py',
    'tests/test_m3w_selected_risk_learning.py',
    'outputs/publication_readiness_2026_09/european_selected_risk_learning_v1/registration.md']


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    assert cfg['seeds'] == [17, 29, 43] and cfg['policies'] == list(method.POLICIES)
    assert cfg['arms'] == list(method.ARMS) and (cfg['new_heads'], cfg['views'], cfg['updates']) == (72, 432, 144000)
    assert cfg['risk_budget'] == .02 and cfg['easy_limit_percent'] == 2.
    assert not any(cfg[k] for k in ('selection_access', 'reserved_calibration_access', 'confirmation_access',
        'new_forecaster_training', 'threshold_refit', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    _, _, _, pid = previous.registration()
    assert digest(previous.PUBLIC/'summary_metrics.json') == cfg['parent_summary_sha256']
    v = json.loads((previous.PUBLIC/'verification.json').read_text()); assert v['all_passed']
    for p, h in v['artifacts'].items(): assert digest(previous.PUBLIC/p) == h
    for p, h in v['source_bindings'].items(): assert digest(ROOT/p) == h
    identity = dict(bindings={f:digest(ROOT/f) for f in FILES}, parent=pid,
                    verification=artifact(previous.PUBLIC/'verification.json'), rosters=pid['rosters'])
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, identity)
    else:
        assert json.loads(path.read_text()) == identity
        previous.require_committed(path)
    return cfg, identity


def contexts(identity):
    groups = {g['group']:g for g in previous.checked_calibration(identity['parent'])['groups']}
    previous.inc.ensure_frozen(); _, cfg, ctx, bid, _, _ = previous.inc.load(); data = ctx[2]
    for easy, all_event in previous.bridge.source_pairs(cfg, ctx, bid):
        a, seed, held = easy['fold'], easy['seed'], easy['ids']
        for b in range(3):
            if a == b: continue
            c = next(i for i in range(3) if i not in (a, b)); name = f'fold{a}_seed{seed}_controller{b}'
            g = groups[name]; pairs = {}
            for role, fold in (('B', b), ('C', c)):
                take = np.isin(data['sites'][held], identity['rosters'][fold]); ids = held[take]
                cv = easy['cv'][take]; damp = base.baseline_numpy(data['history'][ids], 3)-data['origin'][ids, None]
                r, p = easy['prediction'][take], all_event['prediction'][take]
                bits = np.column_stack([v[k][take] for k in ('old', 'bits') for v in (easy, all_event)])
                x, env = previous.bridge.features(data['geometry'][ids], cv, r, p, bits)
                pairs[role] = dict(full=(x, env, r, p), motion_only=previous.motion_pair(
                    data['geometry'][ids], cv, damp, easy['bits'][take], all_event['bits'][take]), ids=ids)
            assert not set(data['sites'][pairs['C']['ids']]) & (set(g['producer_roster']) | set(g['controller_roster']))
            yield g, data, pairs


def restore(directory):
    state = torch.load(directory/'checkpoint.pt', map_location='cpu', weights_only=False)
    model = method.EventMomentHead(len(state['preprocess']['mean']), state['settings']['width'])
    model.load_state_dict(state['model']); model.eval()
    return model, state


def pair_inputs(g, data, pairs, pair):
    bi, ci = pairs['B']['ids'], pairs['C']['ids']
    bx, be, br, bp = pairs['B'][pair]; cx, ce, _, _ = pairs['C'][pair]
    source = previous.PRIVATE/'source'/(g['group']+'_'+pair)
    prior = json.loads((source/'receipt.json').read_text())
    assert prior['input_sha256'] == array_hash(cx) and prior['ids_sha256'] == array_hash(ci)
    with np.load(source/'scores.npz', allow_pickle=False) as z: old = {k:z[k].copy() for k in z.files}
    fun, state, ref = previous.restored(pair, g['group'], 'neural', 'utility')
    bu = fun(bx, be); np.testing.assert_array_equal(fun(cx[:4096], ce[:4096]), old['neural__utility'][:4096])
    rf, rs, rr = previous.restored(pair, g['group'], 'neural', 'all_risk')
    bm = rf(bx, be)
    moving = np.linalg.norm(data['history'][bi, -1]-data['history'][bi, -2], axis=1) > 0
    eligible = moving & (be > 0) & (bu[:, 0] > bu[:, 1])
    raw = choices(bu, bm, bm, moving, be, arm='all_risk_only')
    masks = np.column_stack((np.ones(len(bi), bool), eligible, raw))
    # Only B labels are exposed to training. C outcomes remain in the parent label bank.
    def error(pred):
        return native_errors(pred.astype(float)+data['origin'][bi, None], data['target_eval'][bi],
                             data['valid'][bi], np.ones(len(bi)))[0]
    cv = data['baseline_ade'][bi, 1]; by = method.event_targets(cv, error(br), error(bp), g['easy_cut'])
    pr = base.ordinary.preprocess(bx, by[:, :2], cv, data['sites'][bi], '__C_excluded__')
    for k in ('mean', 'std', 'known', 'weights'): np.testing.assert_array_equal(pr[k], rs['preprocess'][k])
    assert pr['cost_scale'] == rs['preprocess']['cost_scale']
    z = base.ordinary.standardized(bx, pr); score = np.abs(z).max(1)
    edges = np.quantile(score[pr['known']], [.9, .99])
    cscore = np.abs(base.ordinary.standardized(cx, pr)).max(1)
    bins = np.searchsorted(edges, cscore, side='left').astype(np.int8)
    meta = dict(group=g, pair=pair, B_rows=len(bi), C_rows=len(ci), B_ids_sha256=array_hash(bi),
        C_ids_sha256=array_hash(ci), B_x_sha256=array_hash(bx), C_x_sha256=array_hash(cx),
        B_targets_sha256=array_hash(by), B_masks_sha256=array_hash(masks), parent_source=artifact(source/'receipt.json'),
        utility_model=ref, risk_model=rr, training_sites=sorted(set(data['sites'][bi])),
        readout_sites=sorted(set(data['sites'][ci])), support_edges=edges.tolist(), C_labels_used_for_fit=False)
    return bx, be, by, masks, pr, cx, ce, old, bins, cscore, meta


def decision_bank(old, new, env, groups, ids):
    move = old['moving']; u = old['neural__utility']; out = dict(reference=np.zeros(len(ids), bool))
    for family in ('neural', 'ridge'):
        m = old[family+'__all_risk']; uu = old[family+'__utility']
        out['raw_'+family] = choices(uu, m, m, move, env, arm='all_risk_only')
    for arm, m in new.items():
        for mode in ('all', 'dual', 'scene', 'joint'):
            out[arm+'_'+mode] = method.decisions(u, m, move, env, groups, ids, mode)
            np.testing.assert_array_equal(out[arm+'_'+mode], method.scalar_decisions(u, m, move, env, groups, ids, mode))
    out['selected_hash_matched'] = method.matched_hash(u, move, env, groups, ids, out['selected_joint'])
    assert set(out) == set(method.POLICIES)
    return out


def train(cfg, identity, resume=False, pilot=False):
    heads, decisions, inputs = [], [], []
    for g, data, pairs in contexts(identity):
        bi, ci = pairs['B']['ids'], pairs['C']['ids']; name = g['group']; seed = int(name.split('_seed')[1].split('_')[0])
        queries = query_groups(data['sites'][ci], data['recordings'][ci], data['frames'][ci])
        for pair in cfg['pairs']:
            if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('10 GiB disk reserve')
            beat('source_pair', group=name, pair=pair)
            bx, be, by, masks, pr, cx, ce, old, bins, support, meta = pair_inputs(g, data, pairs, pair)
            input_path = PRIVATE/'inputs'/(name+'_'+pair+'.json'); immutable_json(input_path, meta)
            inputs.append(artifact(input_path)); new = {}
            for arm in cfg['arms']:
                directory = PRIVATE/'heads'/(name+'_'+pair+'_'+arm)
                hid = dict(experiment=identity, inputs=artifact(input_path), arm=arm, seed=seed)
                complete = directory/'complete.json'
                if complete.exists():
                    r = json.loads(complete.read_text()); assert r['identity'] == hid
                    for ref in r['artifacts'].values(): assert artifact(ROOT/ref['path']) == ref
                    with np.load(directory/'scores.npz', allow_pickle=False) as z: new[arm] = z['scores'].copy()
                else:
                    model, fitted = method.fit(bx, by, data['sites'][bi], be, masks, pr, arm=arm, seed=seed,
                        settings=cfg['head_training'], identity=hid, directory=directory, resume=resume,
                        stop_at=100 if pilot else None,
                        heartbeat=lambda **kw: beat(group=name, pair=pair, arm=arm, **kw))
                    if pilot:
                        immutable_json(PRIVATE/'pilot.json', dict(fit=fitted, checkpoint=artifact(directory/'checkpoint.pt'),
                            projected_total_training_seconds=fitted['seconds']/100*cfg['updates'], real_updates=100))
                        return
                    assert fitted['complete'] and fitted['unknown_rows_sampled'] == 0
                    score = method.predict(model, cx, ce, pr)
                    previous.parent.atomic_npz(directory/'scores.npz', ids=ci, scores=score)
                    restored, state = restore(directory)
                    np.testing.assert_array_equal(method.predict(restored, cx[:4096], ce[:4096], state['preprocess']), score[:4096])
                    r = dict(identity=hid, fit=fitted, result_source='fresh_run',
                        artifacts=dict(checkpoint=artifact(directory/'checkpoint.pt'), scores=artifact(directory/'scores.npz')))
                    immutable_json(complete, r); new[arm] = score
                heads.append(artifact(complete))
            a, b = [torch.load(PRIVATE/'heads'/(name+'_'+pair+'_'+arm)/'checkpoint.pt', map_location='cpu', weights_only=False)
                    for arm in cfg['arms']]
            np.testing.assert_array_equal(a['draws'], b['draws']); np.testing.assert_array_equal(a['loss_scales'], b['loss_scales'])
            out = decision_bank(old, new, ce, queries, ci)
            path = PRIVATE/'decisions'/(name+'_'+pair+'.npz')
            previous.save_arrays(path, dict(out, ids=ci, support_bins=bins, support_score=support))
            receipt = dict(identity=identity, group=g, pair=pair, inputs=artifact(input_path),
                           decisions=artifact(path), heads=heads[-2:], C_outcomes_used_for_fit=False)
            immutable_json(path.with_suffix('.json'), receipt); decisions.append(artifact(path.with_suffix('.json')))
            beat('pair_frozen', group=name, pair=pair)
    assert len(heads) == 72 and len(decisions) == 36
    immutable_json(PRIVATE/'training_complete.json', dict(identity=identity, heads=heads, decisions=decisions,
        inputs=inputs, all_passed=True, new_updates=144000))
    immutable_json(PUBLIC/'decision_freeze.json', dict(identity=identity, manifest=artifact(PRIVATE/'training_complete.json'),
        heads=72, views=432, new_updates=144000, C_readout_role=cfg['readout_role'], C_outcomes_used_for_fit=False,
        six_selection_localities_opened=False, reserved_calibration_opened=False, confirmation_opened=False))


def checked_training(identity):
    done = json.loads((PRIVATE/'training_complete.json').read_text()); assert done['identity'] == identity and done['all_passed']
    for ref in done['heads']+done['decisions']+done['inputs']: assert artifact(ROOT/ref['path']) == ref
    for ref in done['heads']:
        r = json.loads((ROOT/ref['path']).read_text())
        for a in r['artifacts'].values(): assert artifact(ROOT/a['path']) == a
    for ref in done['decisions']:
        r = json.loads((ROOT/ref['path']).read_text()); assert artifact(ROOT/r['decisions']['path']) == r['decisions']
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['register', 'pilot', 'train', 'evaluate'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    for path in (PUBLIC, PRIVATE): path.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, identity = registration(create=args.phase == 'register')
        if args.phase in ('pilot', 'train'): train(cfg, identity, args.resume, args.phase == 'pilot')
        elif args.phase == 'evaluate':
            from scripts.evaluate_m3w_european_selected_risk_learning import evaluate
            evaluate(sys.modules[__name__], cfg, identity)


if __name__ == '__main__': main()
