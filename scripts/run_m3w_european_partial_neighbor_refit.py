"""Matched source-only neural context refit, with frozen predictions before scoring."""
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

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '4')
import numpy as np
import torch
from scripts import run_m3w_european_source_forecast as parent
from src.world_model.m3w_european_source_forecast import SourceForecaster, fit_design, pack_scene
from src.world_model.m3w_partial_context import PartialContextSourceForecaster
from src.world_model.m3w_observation_quality import masked_neighbors, repair_geometry
from src.world_model.m3w_native_forecast import fit_trial, predict
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation import m3w_partial_neighbor_refit as metrics

BASE = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = BASE/'european_partial_neighbor_refit_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_partial_neighbor_refit_v1'
CONFIG = 'configs/m3w_european_partial_neighbor_refit_v1.json'
FILES = [CONFIG, 'scripts/run_m3w_european_partial_neighbor_refit.py',
    'src/world_model/m3w_partial_context.py', 'src/world_model/m3w_observation_quality.py',
    'src/evaluation/m3w_partial_neighbor_refit.py', 'tests/test_m3w_partial_neighbor_refit.py',
    'tests/test_m3w_partial_context.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]
digest, immutable_json, array_hash = parent.digest, parent.immutable_json, parent.array_hash


def artifact(path): return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def beat(state, **kw):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **kw)
    parent.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def committed(path):
    rel = str(path.relative_to(ROOT))
    old = subprocess.check_output(['git', 'show', 'HEAD:'+rel], cwd=ROOT)
    assert old == path.read_bytes(), 'Committed freeze required'


def load(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    seal_path = BASE/'european_observation_quality_v1/verification.json'
    assert digest(seal_path) == cfg['observation_seal_sha256']
    seal = json.loads(seal_path.read_text())
    for p,h in seal['source_bindings'].items(): assert digest(ROOT/p) == h
    for n,h in seal['artifacts'].items(): assert digest(seal_path.parent/n) == h
    reg, manifest, receipts, old_identity = parent.load_source()
    data = parent.prepare(manifest, receipts, old_identity)
    jobs = []
    for fold in cfg['producer_folds']:
        sites = sorted(s for s,v in old_identity['folds'].items() if v == fold)
        design = fit_design(data, sites, data['baseline_ade'])
        for seed in cfg['seeds']:
            key = f'single{fold}_seed{seed}'
            ti = dict(identity=old_identity, kind='single', fold=fold, seed=seed, fit_sites=sites,
                train_ids_sha256=array_hash(design['train_ids']), held_ids_sha256=array_hash(design['held_ids']),
                factor_sha256=array_hash(design['factors']), normalizers=design['normalizers'],
                baseline_index=design['baseline_index'], selection_relative_scores=design['selection_relative_scores'],
                easy_cut=design['easy_cut'], hard_cut=design['hard_cut'])
            old = parent.complete(key, ti, reg)
            pr = json.loads((parent.PRIVATE/'predictions'/(key+'.json')).read_text())
            assert pr['identity'] == ti and digest(ROOT/pr['path']) == pr['sha256']
            jobs.append(dict(key=key, design=design, old_identity=ti, old=old,
                old_prediction=dict(path=pr['path'], sha256=pr['sha256'])))
    assert len(jobs) == 9 and cfg['steps'] == reg['training']['steps'] == 4000
    assert cfg['seeds'] == reg['seeds'] == [17, 29, 43]
    identity = dict(observation_seal=artifact(seal_path), parent_identity=old_identity,
        bindings={p:digest(ROOT/p) for p in FILES},
        controls=[dict(key=j['key'], checkpoint=dict(path=j['old']['checkpoint'], sha256=j['old']['checkpoint_sha256']),
            prediction=j['old_prediction']) for j in jobs])
    dest = PUBLIC/'registration.json'
    if create: immutable_json(dest, identity)
    else:
        assert json.loads(dest.read_text()) == identity; committed(dest)
    return cfg, reg, data, receipts, jobs, identity


def prepared(data, receipts, identity, build=False):
    dest = PRIVATE/'partial_geometry.npy'; seal = PRIVATE/'geometry.json'
    if not seal.exists():
        if not build: raise ValueError('Prepare input-only geometry before training')
        size = data['geometry'].nbytes
        if shutil.disk_usage(PRIVATE).free < 10*2**30+size: raise OSError('Preserve10GiB disk')
        tmp = dest.with_suffix('.tmp.npy')
        out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=data['geometry'].shape)
        cursor = 0; changed = 0
        for ri, (home, receipt) in enumerate(receipts):
            inp = {k.removeprefix('input_'):np.load(home/v['name'], mmap_mode='r', allow_pickle=False)
                   for k,v in receipt['arrays'].items() if k.startswith('input_')}
            for start,end in zip(inp['query_offsets'][:-1], inp['query_offsets'][1:]):
                scene = {k:v[start:end] for k,v in inp.items() if k != 'query_offsets'}
                g, ids = pack_scene(scene); span = slice(cursor,cursor+len(g))
                np.testing.assert_array_equal(g, data['geometry'][span])
                np.testing.assert_array_equal(data['recordings'][span], np.full(len(g),ri))
                np.testing.assert_array_equal(data['agents'][span], scene['agent_id'][ids])
                neighbor = masked_neighbors(scene); np.testing.assert_array_equal(ids, neighbor['target'])
                new = repair_geometry(g, neighbor); changed += int(np.any(g!=new, axis=1).sum())
                out[span] = new; cursor += len(g)
            beat('geometry_record', record=ri+1, rows=cursor)
        assert cursor == len(data['sites']) == 318969 and changed == 282529
        out.flush(); del out; os.replace(tmp, dest)
        immutable_json(seal, dict(registration=artifact(PUBLIC/'registration.json'), geometry=artifact(dest),
            rows=cursor, changed_rows=changed, labels_used=False, independent_roles_read=False))
    doc = json.loads(seal.read_text())
    assert doc['registration'] == artifact(PUBLIC/'registration.json') and doc['geometry'] == artifact(dest)
    result = dict(data); result['geometry'] = np.load(dest, mmap_mode='r', allow_pickle=False)
    return result


def make_model(job, reg, arm):
    torch.manual_seed(job['old_identity']['seed'])
    if arm == 'legacy_control': return SourceForecaster(reg['architecture'], job['design']['baseline_index'])
    return PartialContextSourceForecaster(job['design']['baseline_index'], width=64, heads=4, layers=2)


def state_matches(new, old, *, parameters):
    for k in ('draws', 'train_ids', 'factors', 'sampler_rng'):
        np.testing.assert_array_equal(new[k], old[k])
    assert new['step'] == old['step'] == 4000
    if parameters:
        for k,v in old['model'].items(): torch.testing.assert_close(v, new['model'][k], rtol=0, atol=0)


def train(cfg, reg, data, partial, jobs, identity, phase, resume):
    legacy = phase in ('control', 'pilot'); selected = jobs[:1] if legacy else jobs
    if not legacy:
        assert json.loads((PUBLIC/'legacy_control.json').read_text())['parameters_exact']
    for j in selected:
        arm = 'legacy_control' if legacy else 'partial'
        home = PRIVATE/arm/j['key']; complete = home/'complete.json'
        binding = dict(registration=artifact(PUBLIC/'registration.json'), arm=arm,
            parent_trial=j['old_identity'], geometry=artifact(PRIVATE/'geometry.json') if not legacy else None)
        if complete.exists():
            r = json.loads(complete.read_text()); assert r['identity'] == binding
            assert artifact(ROOT/r['checkpoint']['path']) == r['checkpoint']
            beat('cached_verified_endpoint', arm=arm, trial=j['key']); continue
        if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; resume checkpoint')
        model = make_model(j, reg, arm)
        if not legacy:
            ref = make_model(j, reg, 'legacy_control')
            for k,v in ref.state_dict().items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
            del ref
        fit = fit_trial(model, data if legacy else partial, j['design'], seed=j['old_identity']['seed'],
            settings=reg['training'], identity=binding, directory=home, resume=resume,
            stop_at=cfg['pilot_updates'] if phase=='pilot' else None,
            heartbeat=lambda **kw:beat(arm=arm, trial=j['key'], **kw))
        checkpoint = home/'checkpoint.pt'
        if phase == 'pilot':
            immutable_json(PRIVATE/'pilot.json', dict(fit=fit, checkpoint=artifact(checkpoint), resumes_same_budget=True))
            return
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        old = torch.load(ROOT/j['old']['checkpoint'], map_location='cpu', weights_only=False)
        state_matches(state, old, parameters=legacy)
        assert fit['held_rows_sampled'] == 0 and fit['complete']
        immutable_json(complete, dict(identity=binding, checkpoint=artifact(checkpoint), fit=fit,
            draw_counts_match=True, held_rows_sampled=0, result_source='fresh_native_torch_training'))
        if legacy:
            immutable_json(PUBLIC/'legacy_control.json', dict(parameters_exact=True, sampling_exact=True,
                steps=4000, complete=artifact(complete), old_checkpoint=dict(path=j['old']['checkpoint'], sha256=j['old']['checkpoint_sha256'])))
        beat('trained_endpoint', trial=j['key'], arm=arm, seconds=fit['seconds'])


def frozen_training(jobs):
    refs = []
    for j in jobs:
        home = PRIVATE/'partial'/j['key']; r = json.loads((home/'complete.json').read_text())
        assert r['fit']['step'] == 4000 and r['held_rows_sampled'] == 0
        assert r['identity']['parent_trial'] == j['old_identity']
        assert r['identity']['registration'] == artifact(PUBLIC/'registration.json')
        assert r['identity']['geometry'] == artifact(PRIVATE/'geometry.json')
        assert r['identity']['arm'] == 'partial'
        assert artifact(ROOT/r['checkpoint']['path']) == r['checkpoint']
        refs.append(artifact(home/'complete.json'))
    return refs


def produce(reg, data, partial, jobs, verify=False):
    refs = frozen_training(jobs); predictions = []
    for j in jobs:
        ids = j['design']['held_ids']; home = PRIVATE/'partial'/j['key']
        cp = torch.load(home/'checkpoint.pt', map_location='cpu', weights_only=False)
        model = make_model(j, reg, 'partial'); model.load_state_dict(cp['model'])
        p = predict(model, partial, ids, 128)
        old = make_model(j, reg, 'legacy_control')
        old.load_state_dict(torch.load(ROOT/j['old']['checkpoint'], map_location='cpu', weights_only=False)['model'])
        with np.load(ROOT/j['old_prediction']['path'], allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached['ids'], ids)
            np.testing.assert_array_equal(predict(old, data, ids, 128), cached['prediction'])
        dest = home/'prediction.npz'
        if verify or dest.exists():
            with np.load(dest, allow_pickle=False) as cached:
                np.testing.assert_array_equal(cached['ids'], ids)
                np.testing.assert_array_equal(cached['prediction'], p)
        else:
            tmp = dest.with_suffix('.tmp')
            with tmp.open('wb') as f: np.savez(f, ids=ids, prediction=p)
            os.replace(tmp, dest)
        predictions.append(dict(key=j['key'], partial=artifact(dest), legacy=j['old_prediction'],
            rows=len(ids), legacy_fresh_inference_exact=True))
        beat('prediction_replayed' if verify else 'prediction_frozen', trial=j['key'], rows=len(ids))
    doc = dict(registration=artifact(PUBLIC/'registration.json'), training=refs, predictions=predictions,
        comparison_targets_used=False, new_models=9, new_model_updates=36000, legacy_replay_updates=4000,
        independent_roles_read=False)
    immutable_json(PUBLIC/'prediction_freeze.json', doc)
    if verify: immutable_json(PUBLIC/'prediction_replay.json', dict(all_exact=True, model_pairs=9,
        prediction_freeze_sha256=digest(PUBLIC/'prediction_freeze.json')))


def evaluate(cfg, data, jobs, verify=False):
    path = PUBLIC/'prediction_freeze.json'; committed(path)
    frozen = json.loads(path.read_text()); rows = []; proxies = []
    for j,ref in zip(jobs,frozen['predictions']):
        assert ref['key'] == j['key']
        for arm in ('partial','legacy'): assert artifact(ROOT/ref[arm]['path']) == ref[arm]
        with np.load(ROOT/ref['partial']['path'], allow_pickle=False) as z: ids, p = z['ids'].copy(), z['prediction'].copy()
        with np.load(ROOT/ref['legacy']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); old = z['prediction'].copy()
        y, valid = data['target_eval'][ids], data['valid'][ids]
        origin = data['origin'][ids, None]
        ae, fe = native_errors(p.astype(float)+origin, y, valid, np.ones(len(ids)))
        oa, of = native_errors(old.astype(float)+origin, y, valid, np.ones(len(ids)))
        cv = data['baseline_ade'][ids, 1]
        subset = metrics.masks(cv, j['design']['easy_cut'], j['design']['hard_cut'])
        b = j['design']['baseline_index']; sites = data['sites'][ids]
        assert not set(sites) & set(j['old_identity']['fit_sites'])
        for site in sorted(set(sites)):
            take = sites == site
            for endpoint,a,o in [('ADE',ae,oa),('FDE',fe,of)]:
                r, endpoint_cv = metrics.endpoint_references(data, ids, b, endpoint)
                for name,m in subset.items():
                    result = metrics.slice_metrics(a[take],o[take],r[take],endpoint_cv[take],m[take])
                    rows.append(dict(trial=j['key'], fold=j['old_identity']['fold'], seed=j['old_identity']['seed'],
                        site=str(site), subset=name, endpoint=endpoint, metric=result))
            def smooth(pred): return float(np.linalg.norm(np.diff(pred[take], n=2, axis=1),axis=2).mean())
            proxies.append(dict(trial=j['key'],site=str(site),rows=int(take.sum()),
                recordings=int(len(np.unique(data['recordings'][ids][take]))),
                finite_output=bool(np.isfinite(p[take]).all()),
                raw_step_acceleration_new=smooth(p),raw_step_acceleration_legacy=smooth(old)))
        beat('source_scored', trial=j['key'])
    roster = sorted(set(data['sites']))
    assert len(rows) == 9*8*2*4 and len(roster) == 12
    summaries = {}
    for endpoint in ('ADE','FDE'):
        for subset in ('all','positive_easy','hard','zero_CV'):
            rr = [r for r in rows if r['endpoint']==endpoint and r['subset']==subset]
            summaries[endpoint+'_'+subset] = {key:metrics.paired_localities(rr,roster,key,cfg['bootstrap_draws'],cfg['bootstrap_seed'])
                for key in ('gain_vs_legacy_percent','gain_vs_reference_percent','gain_vs_CV_percent','absolute_harm_vs_legacy')}
    per_seed = {str(seed): metrics.paired_localities([r for r in rows if r['endpoint']=='ADE' and r['subset']=='all' and r['seed']==seed],
        roster,'gain_vs_legacy_percent',cfg['bootstrap_draws'],cfg['bootstrap_seed']) for seed in cfg['seeds']}
    doc = dict(result_source='fresh_source_development_readout_cached_verified_legacy_controls', rows=rows,
        summaries=summaries, per_seed=per_seed, motion_proxies=proxies, expected_localities=roster,
        independent_confirmation=False, new_risk_head_training=False, deployment_changed=False)
    immutable_json(PUBLIC/'evaluation.json',doc)
    if verify: immutable_json(PUBLIC/'evaluation_replay.json',dict(all_exact=True,rows=len(rows), evaluation_sha256=digest(PUBLIC/'evaluation.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['register','prepare','pilot','control','train','predict','evaluate','replay','verify_eval'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True); PUBLIC.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        beat('started',phase=args.phase,threads=4,workers=0,architecture=platform.machine())
        cfg, reg, data, receipts, jobs, identity = load(args.phase=='register')
        if args.phase in ('prepare','train','predict','replay'):
            partial = prepared(data,receipts,identity,build=args.phase=='prepare')
        else: partial = None
        if args.phase in ('pilot','control','train'): train(cfg,reg,data,partial,jobs,identity,args.phase,args.resume)
        if args.phase in ('predict','replay'): produce(reg,data,partial,jobs,args.phase=='replay')
        if args.phase in ('evaluate','verify_eval'): evaluate(cfg,data,jobs,args.phase=='verify_eval')
        beat('complete',phase=args.phase)


if __name__ == '__main__': main()
