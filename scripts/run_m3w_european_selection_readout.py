"""Verify frozen source policies before admitting model-selection-only localities."""
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
    raise RuntimeError('Native arm64 Python required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_european_incumbent_relative as inc
from src.world_model.m3w_frozen_selection import INPUTS, POLICIES, motion_floor, infer
from src.world_model.m3w_fixed_producer_roles import forecaster_baseline, bounded_ridge_scores
from src.world_model.m3w_european_source_forecast import SourceForecaster
from src.world_model.m3w_native_forecast import predict

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selection_readout_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_selection_readout_v1'
digest, artifact, immutable_json = inc.digest, inc.artifact, inc.immutable_json
CONFIG = 'configs/m3w_european_selection_readout_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_frozen_selection.py',
    'scripts/run_m3w_european_selection_readout.py', 'scripts/build_m3w_european_selection_data.py',
    'scripts/evaluate_m3w_european_selection_readout.py', 'tests/test_m3w_frozen_selection.py',
    'outputs/publication_readiness_2026_09/european_selection_readout_v1/registration.md',
    'src/data_unification/m3w_european_squares_source.py', 'src/world_model/m3w_european_source_forecast.py',
    'src/world_model/m3w_native_forecast.py', 'src/world_model/m3w_floor_relative.py',
    'src/world_model/m3w_incumbent_relative.py', 'src/evaluation/m3w_native_metrics.py']


def beat(state, **kw):
    r = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    inc.base.cross.json_write(PRIVATE/'heartbeat.json', r)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
    print(json.dumps(r), flush=True)


def restore(ref, kind=None):
    if artifact(ROOT/ref['path']) != ref:
        raise ValueError('Changed frozen head receipt')
    r = json.loads((ROOT/ref['path']).read_text())
    if kind == 'ridge':
        for a in r['artifacts'].values():
            if artifact(ROOT/a['path']) != a: raise ValueError('Changed ridge artifact')
        state = torch.load(ROOT/r['artifacts']['model']['path'], map_location='cpu', weights_only=False)
        if state['identity'] != r['identity']: raise ValueError('Changed ridge identity')
        task = r['identity']['task']
        return lambda x, env: bounded_ridge_scores(inc.base.ordinary.predict_ridge(
            state['head'], x, np.zeros(len(x), bool), state['preprocess']), env, task)
    if kind is not None: r = dict(r, kind=kind)
    return inc.prior.previous.restored(r)[0]


def head_refs(g):
    out = {}
    for task in ('utility', 'risk'):
        out['old_stop_'+task] = artifact(inc.base.PRIVATE/'fitting/heads'/(g['original']+'_floor_'+task)/'complete.json')
        out['previous_matched_'+task] = artifact(inc.prior.PRIVATE/'heads'/(g['name']+'_producer_matched_'+task)/'complete.json')
        for arm in ('floor_reference', 'incumbent_reference'):
            out[arm+'_'+task] = artifact(inc.PRIVATE/'heads'/(g['name']+'_'+arm+'_'+task)/'complete.json')
        out['ridge_incumbent_'+task] = artifact(inc.PRIVATE/'ridge'/(g['name']+'_'+task)/'complete.json')
    return out


def restored_heads(refs):
    return {(arm, task): restore(refs[arm+'_'+task], 'ridge' if arm == 'ridge_incumbent' else None)
            for arm in ('old_stop', 'previous_matched', 'floor_reference', 'incumbent_reference', 'ridge_incumbent')
            for task in ('utility', 'risk')}


def source_preflight():
    inc.ensure_frozen()
    cfg, bcfg, ctx, bid, pid, iid = inc.load()
    data = ctx[2]
    fcfg = json.loads((ROOT/'configs/m3w_european_source_forecast_v1.json').read_text())
    bank, results = {}, []
    for g in inc.groups(bcfg, ctx, bid, pid):
        beat('source_replay', group=g['name'])
        ids = g['roles']['readout'][:4096]
        inputs = {k: data[k][ids] for k in INPUTS}
        fr = g['a']['lineage']['final_producer']['checkpoint']
        if artifact(ROOT/fr['path']) != fr: raise ValueError('Changed source forecaster')
        s = torch.load(ROOT/fr['path'], map_location='cpu', weights_only=False)
        index = forecaster_baseline(s, iid['rosters'][g['fold']], g['fold'], g['seed'])
        model = SourceForecaster(fcfg['architecture'], index); model.load_state_dict(s['model'])
        neural = predict(model, inputs, np.arange(len(ids)), 128)
        np.testing.assert_array_equal(neural, g['a']['p'][ids])
        key = f"damping097_fold{g['fold']}_seed{g['seed']}"
        uref = artifact(ROOT/ctx[4][key+'_utility']['artifacts']['checkpoint']['path'].replace('checkpoint.pt', 'complete.json'))
        rref = artifact(inc.base.cross.PRIVATE_ROOT/'fitting/heads'/(key+'_'+g['event'])/'complete.json')
        floor = motion_floor(inputs, restore(uref, 'ordinary_utility'), restore(rref, 'ranked_risk'))
        np.testing.assert_array_equal(floor[0], g['a']['b'][ids])
        np.testing.assert_array_equal(floor[1], g['d'][ids])
        np.testing.assert_array_equal(floor[2], g['bits'][ids])
        refs = head_refs(g)
        out, x, env = infer(inputs, neural, floor, restored_heads(refs))
        np.testing.assert_array_equal(x, g['x'][ids]); np.testing.assert_array_equal(env, g['env'][ids])
        with np.load(inc.PRIVATE/'decisions'/(g['name']+'.npz'), allow_pickle=False) as z:
            where = np.searchsorted(z['ids'], ids)
            np.testing.assert_array_equal(z['ids'][where], ids)
            for k, value in out.items():
                if k.startswith('previous_matched__'): continue
                np.testing.assert_array_equal(value, z[k][where])
        bank[g['name']] = dict(fold=g['fold'], controller=g['controller'], seed=g['seed'], event=g['event'],
            forecaster=fr, baseline_index=index, floor_utility=uref, floor_risk=rref, heads=refs,
            fit_sources=sorted(iid['rosters'][g['fold']]+iid['rosters'][g['controller']]),
            easy_cut=g['design']['easy_cut'], hard_cut=g['design']['hard_cut'])
        results.append(dict(group=g['name'], rows=len(ids), predictions_exact=True,
            floor_exact=True, features_exact=True, policies_exact=len(POLICIES)))
    immutable_json(PUBLIC/'frozen_bank.json', dict(architecture=fcfg['architecture'], groups=bank,
        parent_summary=artifact(inc.PUBLIC/'summary_metrics.json'), producer_rosters=iid['rosters']))
    immutable_json(PUBLIC/'source_replay.json', dict(result_source='fresh_run', source_only=True,
        model_selection_opened=False, calibration_opened=False, confirmation_opened=False,
        groups=results, all_passed=True, input_api=list(INPUTS), policies=list(POLICIES)))
    beat('source_replay_complete', groups=len(results))


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    bank = json.loads((PUBLIC/'frozen_bank.json').read_text())
    replay = json.loads((PUBLIC/'source_replay.json').read_text())
    if not replay['all_passed'] or len(replay['groups']) != 36: raise ValueError('Source replay missing')
    assert cfg['policies'] == list(POLICIES) and len(bank['groups']) == cfg['groups'] == 36
    assert not any(cfg[k] for k in ('new_fitting', 'threshold_refit', 'calibration_access',
        'confirmation_access', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    value = dict(bindings={p: digest(ROOT/p) for p in FILES}, bank=artifact(PUBLIC/'frozen_bank.json'),
        source_replay=artifact(PUBLIC/'source_replay.json'), producer_rosters=bank['producer_rosters'],
        roles=artifact(ROOT/'outputs/publication_readiness_2026_09/european_squares_roles_v1/roles.json'))
    path = PUBLIC/'registration_lock.json'
    if create: immutable_json(path, value)
    elif json.loads(path.read_text()) != value: raise ValueError('Registered implementation or bank changed')
    if not create:
        # Opening a new role requires the registration to exist in a pushed commit.
        for p in (*FILES, str(path.relative_to(ROOT)), str((PUBLIC/'frozen_bank.json').relative_to(ROOT)),
                  str((PUBLIC/'source_replay.json').relative_to(ROOT))):
            if subprocess.check_output(['git', 'show', 'HEAD:'+p], cwd=ROOT) != (ROOT/p).read_bytes():
                raise ValueError('Uncommitted registration '+p)
        commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(path.relative_to(ROOT))], cwd=ROOT).decode().strip()
        remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'HEAD'], cwd=ROOT).decode().split()[0]
        subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=ROOT, check=True)
    return cfg, bank, value


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz'); np.savez(tmp, **arrays); os.replace(tmp, path)


def decide(cfg, bank, identity):
    from scripts import build_m3w_european_selection_data as adapter
    from scripts.build_m3w_european_squares_source import store_array
    inc.ensure_frozen()
    data = adapter.load(sys.modules[__name__], identity)
    inputs = {k: data[k] for k in INPUTS}; n = len(data['sites']); refs = []
    if sorted(set(data['sites'])) != cfg['localities']: raise ValueError('Missing selection locality')
    for name, group in bank['groups'].items():
        path = PRIVATE/'decisions'/(name+'.npz'); receipt_path = path.with_suffix('.json')
        if receipt_path.exists():
            r = json.loads(receipt_path.read_text())
            if r['identity'] != identity or r['group'] != group: raise ValueError('Decision identity changed')
            for a in r['artifacts'].values():
                if artifact(ROOT/a['path']) != a: raise ValueError('Cached inference changed')
            refs.append(artifact(receipt_path)); beat('decisions_cached_verified', group=name); continue
        if shutil.disk_usage(PRIVATE).free < 10*1024**3: raise OSError('Below10GiB; retain predictions')
        if set(group['fit_sources']) & set(cfg['localities']): raise ValueError('Source overlap')
        beat('infer_group', group=name, rows=n)
        pred_path = PRIVATE/'forecasts'/f"fold{group['fold']}_seed{group['seed']}.npy"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        if pred_path.with_suffix('.json').exists():
            p = json.loads(pred_path.with_suffix('.json').read_text())
            if p['identity'] != identity or p['checkpoint'] != group['forecaster'] or artifact(pred_path) != p['artifact']:
                raise ValueError('Prediction lineage changed')
        else:
            fr = group['forecaster']
            if artifact(ROOT/fr['path']) != fr: raise ValueError('Changed forecaster')
            s = torch.load(ROOT/fr['path'], map_location='cpu', weights_only=False)
            index = forecaster_baseline(s, bank['producer_rosters'][group['fold']], group['fold'], group['seed'])
            if index != group['baseline_index']: raise ValueError('Frozen baseline differs')
            model = SourceForecaster(bank['architecture'], index); model.load_state_dict(s['model'])
            neural = predict(model, inputs, np.arange(n), 128); store_array(pred_path, neural)
            immutable_json(pred_path.with_suffix('.json'), dict(identity=identity, checkpoint=fr, artifact=artifact(pred_path)))
        neural = np.load(pred_path, allow_pickle=False, mmap_mode='r')
        floor = motion_floor(inputs, restore(group['floor_utility'], 'ordinary_utility'), restore(group['floor_risk'], 'ranked_risk'))
        out, _, _ = infer(inputs, neural, floor, restored_heads(group['heads']))
        atomic_npz(path, floor_bit=floor[2], floor_utility=floor[3], floor_risk=floor[4], **out)
        r = dict(identity=identity, group=group, rows=n, result_source='fresh_run',
            labels_read=False, artifacts=dict(decisions=artifact(path), neural=artifact(pred_path)))
        immutable_json(receipt_path, r); refs.append(artifact(receipt_path)); beat('decisions_frozen', group=name)
    r = dict(identity=identity, groups=refs, all_passed=True, new_training=False,
        new_outcomes_evaluated=False, rows=n, policies=list(POLICIES), result_source='fresh_run',
        calibration_opened=False, confirmation_opened=False)
    immutable_json(PRIVATE/'decisions_complete.json', r)
    immutable_json(PUBLIC/'decision_freeze.json', dict(manifest=artifact(PRIVATE/'decisions_complete.json'),
        groups=len(refs), rows=n, policies=len(POLICIES), identity=identity, result_source='fresh_run',
        new_outcomes_evaluated=False, calibration_opened=False, confirmation_opened=False))
    beat('inference_complete', groups=len(refs), rows=n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['preflight', 'register', 'build', 'decide', 'evaluate'], required=True)
    args = parser.parse_args()
    for d in (PRIVATE, PUBLIC): d.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'preflight': source_preflight()
        elif args.phase == 'register': registration(create=True)
        else:
            cfg, bank, identity = registration()
            if args.phase == 'build':
                from scripts.build_m3w_european_selection_data import build
                build(sys.modules[__name__], cfg, identity)
            elif args.phase == 'decide': decide(cfg, bank, identity)
            else:
                from scripts.evaluate_m3w_european_selection_readout import evaluate
                evaluate(sys.modules[__name__], cfg, bank, identity)


if __name__ == '__main__': main()
