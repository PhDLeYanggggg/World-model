"""Fixed neural assets, fresh incremental joint controls on opened source queries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_incumbent_relative as parent
from src.world_model.m3w_european_source_intervention import query_subset
from src.world_model.m3w_incremental_joint import problem, controls
from src.world_model.m3w_incumbent_relative import replay
import numpy as np
import torch

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_incremental_joint_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_incremental_joint_v1'
CONFIG = 'configs/m3w_european_incremental_joint_v1.json'
FILES = (CONFIG, 'scripts/run_m3w_european_incremental_joint.py',
    'src/world_model/m3w_incremental_joint.py', 'tests/test_m3w_incremental_joint.py',
    'src/world_model/m3w_native_joint_controls.py', 'src/world_model/m3w_interaction_controls.py',
    'src/world_model/m3w_joint_intervention.py',
    'outputs/publication_readiness_2026_09/european_incremental_joint_v1/registration.md')
artifact, digest, immutable_json, array_hash = parent.artifact, parent.digest, parent.immutable_json, parent.array_hash


def beat(state, **values):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **values)
    parent.base.cross.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def load():
    cfg = json.loads((ROOT/CONFIG).read_text())
    assert cfg['parent_summary_sha256'] == digest(parent.PUBLIC/'summary_metrics.json')
    checked = json.loads((parent.PUBLIC/'completion_checks.json').read_text())
    assert checked['all_passed']
    for f, sha in checked['artifact_hashes'].items(): assert digest(parent.PUBLIC/f) == sha
    for f, sha in checked['source_bindings'].items(): assert digest(ROOT/f) == sha
    parent.ensure_frozen()
    pcfg, bcfg, ctx, bid, pid, iid = parent.load(); data = ctx[2]
    assert cfg['groups'] == 36 and cfg['seeds'] == [17, 29, 43] and cfg['fraction'] == .5
    assert not any(cfg[k] for k in ('new_training', 'threshold_selection', 'reserved_roles_opened',
        'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    mask = query_subset(data['sites'], data['recordings'], data['frames'],
                        cfg['joint_queries_per_locality'], cfg['query_salt'])
    identity = dict(bindings={f: digest(ROOT/f) for f in FILES}, parent=iid,
        parent_completion=artifact(parent.PUBLIC/'completion_checks.json'), query_hash=array_hash(mask),
        query_rows=int(mask.sum()), rosters=iid['rosters'])
    immutable_json(PRIVATE/'identity.json', identity)
    return cfg, bcfg, ctx, bid, pid, identity, mask


def check(receipt, identity):
    r = json.loads(receipt.read_text()); assert r['identity'] == identity and r['verified']
    for a in r['artifacts'].values(): assert artifact(ROOT/a['path']) == a
    return r


def direct_proxy(p, selected, floor, neural, old, widths):
    incumbent = np.where(old[:, None, None], neural, floor)
    chosen = np.where((old | selected)[:, None, None], neural, floor)
    if not len(p.edges): return 0.
    i, j = p.edges.T; threshold = .5*float(np.median(widths))
    def score(x):
        d = np.sqrt(((x[i]-x[j])**2).sum(-1))
        return (np.maximum(1-d/threshold, 0)**2).mean(1)
    return float(np.maximum(score(chosen)-score(incumbent), 0).mean())


def decide(resume=False):
    cfg, bcfg, ctx, bid, pid, identity, qmask = load(); data = ctx[2]; refs = []
    for g in parent.groups(bcfg, ctx, bid, pid):
        receipt = PRIVATE/'decisions'/(g['name']+'.json')
        if receipt.exists():
            if not resume: raise ValueError('Existing decision requires --resume')
            check(receipt, identity); refs.append(artifact(receipt)); continue
        start = time.monotonic(); held = g['roles']['readout']; keep = qmask[held]; ids = held[keep]
        with np.load(parent.PRIVATE/'decisions'/(g['name']+'.npz'), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], held)
            old, full = z['old_stop'][keep], z['add_only'][keep]
            utility, risk = z['incumbent_reference__utility'][keep], z['incumbent_reference__risk'][keep]
        moving = np.linalg.norm(data['history'][ids, -1]-data['history'][ids, -2], axis=1) > 0
        np.testing.assert_array_equal(full, replay(utility, risk, moving, old, arm='incumbent_reference', direction='add'))
        assert not np.any(old & ~full)
        state = torch.load(parent.PRIVATE/'heads'/(g['name']+'_incumbent_reference_utility')/'checkpoint.pt',
                           map_location='cpu', weights_only=False)
        scale = float(state['preprocess']['cost_scale']); del state
        bits = {p: old.copy() for p in cfg['policies']}
        matched, active = np.zeros(len(ids), bool), np.zeros(len(ids), bool)
        keys = np.column_stack((data['recordings'][ids], data['frames'][ids]))
        unique, inverse = np.unique(keys, axis=0, return_inverse=True); summaries = []
        for qi, key in enumerate(unique):
            loc = np.flatnonzero(inverse == qi); ix = ids[loc]
            floor = g['d'][ix].astype(float)+data['origin'][ix, None]
            neural = g['a']['p'][ix].astype(float)+data['origin'][ix, None]
            p = problem(floor=floor, neural=neural, old=old[loc], pool=full[loc] & ~old[loc],
                current=data['origin'][ix], widths=data['width'][ix], utility=utility[loc], risk=risk[loc],
                cost_scale=scale, pair_weight=cfg['pair_weight'], radius_widths=cfg['edge_radius_bbox_widths'],
                threshold_widths=cfg['proximity_threshold_bbox_widths'])
            choices, r = controls(p, ix, seconds=cfg['solver_seconds'], salt=cfg['hash_salt'])
            for name, v in choices.items():
                bits[name][loc] = old[loc] | v
                direct = direct_proxy(p, v, floor, neural, old[loc], data['width'][ix])
                np.testing.assert_allclose(direct, r['arms'][name]['mean_pair_proxy'], rtol=1e-8, atol=1e-10)
            matched[loc] = r['matched']; active[loc] = r['nonadditive_supported_edges_at_count'] > 0
            summaries.append(dict(recording=int(key[0]), frame=int(key[1]), site=str(data['sites'][ix[0]]), **r))
            if qi % 96 == 0: beat('joint_inference', group=g['name'], query=qi, queries=len(unique))
        np.testing.assert_array_equal(bits['full_add'], full)
        for v in bits.values(): assert not np.any(old & ~v)
        directory = receipt.parent; directory.mkdir(parents=True, exist_ok=True)
        path = receipt.with_suffix('.npz'); temp = path.with_suffix('.tmp.npz')
        np.savez(temp, ids=ids, matched=matched, active=active, **bits); os.replace(temp, path)
        query_path = directory/(g['name']+'_queries.json'); immutable_json(query_path, summaries)
        r = dict(identity=identity, verified=True, group=g['name'], rows=len(ids), queries=len(unique),
            producer=g['fold'], controller=g['controller'], readout=g['roles']['readout_fold'],
            seed=g['seed'], event=g['event'], cost_scale=scale, seconds=time.monotonic()-start,
            artifacts=dict(choices=artifact(path), queries=artifact(query_path)),
            result_source='fresh_run_controls_cached_verified_forecasters_and_heads',
            future_labels_used_for_decisions=False, original_actions_preserved=True)
        immutable_json(receipt, r); refs.append(artifact(receipt)); beat('group_decisions_frozen', group=g['name'], seconds=r['seconds'])
    assert len(refs) == 36
    immutable_json(PRIVATE/'decisions_complete.json', dict(identity=identity, groups=refs, all_passed=True))


def ensure_frozen():
    r = json.loads((PRIVATE/'decisions_complete.json').read_text()); assert r['all_passed'] and len(r['groups']) == 36
    for f, sha in r['identity']['bindings'].items(): assert digest(ROOT/f) == sha
    parent.ensure_frozen()
    for ref in r['groups']:
        assert artifact(ROOT/ref['path']) == ref; check(ROOT/ref['path'], r['identity'])
    return r


def evaluate(resume=False):
    frozen = ensure_frozen(); cfg, bcfg, ctx, bid, pid, identity, qmask = load()
    assert frozen['identity'] == identity
    data = ctx[2]; refs = []
    for g in parent.groups(bcfg, ctx, bid, pid):
        path = PRIVATE/'evaluation'/(g['name']+'.json')
        if path.exists():
            if not resume: raise ValueError('Existing readout requires --resume')
            r = json.loads(path.read_text()); assert r['identity'] == identity and r['verified']
            refs.append(artifact(path)); continue
        with np.load(PRIVATE/'decisions'/(g['name']+'.npz'), allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        ids = a.pop('ids'); np.testing.assert_array_equal(ids, g['roles']['readout'][qmask[g['roles']['readout']]])
        sites = data['sites'][ids]; roster = identity['rosters'][g['roles']['readout_fold']]
        checks = 0
        def errors(p):
            absolute = p.astype(float)+data['origin'][ids, None]
            aa = parent.prior.native_errors(absolute, data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))
            bb = parent.base.cross.independent.coordinate_errors(absolute, data['target_eval'][ids], data['valid'][ids])
            for x, y in zip(aa, bb): parent.base.cross.independent.close(x, y)
            return aa
        def metric(p, ref, mask):
            nonlocal checks
            r = parent.prior.paired_scene_metrics(p[mask], ref[mask], sites[mask], expected_scenes=roster,
                dataset='EuropeanSquares_released_detector_tracks', coordinate_unit='image_pixel',
                bootstrap_resamples=3000, seed=39271)
            parent.base.cross.independent.check_metric(p, ref, sites, mask, r, cfg); checks += 1
            return r
        da, df = errors(g['d'][ids]); na, nf = errors(g['a']['p'][ids]); cv = data['baseline_ade'][ids, 1]
        masks = dict(all=np.ones(len(ids), bool), easy=(cv > 0) & (cv <= g['design']['easy_cut']),
            hard=cv >= g['design']['hard_cut'], complete=data['valid'][ids].all(1),
            matched=a['matched'], nonadditive=a['matched'] & a['active'])
        err = {p: np.where(a[p], na, da) for p in cfg['policies']}
        fde = {p: np.where(a[p], nf, df) for p in cfg['policies']}
        views = {}
        for p in cfg['policies']:
            views[p] = dict(ADE_vs_incumbent={s: metric(err[p], err['old_stop'], m) for s, m in masks.items()},
                ADE_vs_full_add={s: metric(err[p], err['full_add'], m) for s, m in masks.items()},
                FDE_vs_incumbent=metric(fde[p], fde['old_stop'], masks['all']),
                easy_vs_CV=metric(err[p], cv, masks['easy']),
                switch_rate=float(a[p].mean()), added_rows=int((a[p] & ~a['old_stop']).sum()),
                unknown_ADE_switches=int((a[p] & ~np.isfinite(da)).sum()),
                zero_CV=dict(rows=int((cv == 0).sum()), harmed_rows=int(((cv == 0) & (err[p] > 0)).sum())),
                realized_positive_harm_by_source={})
            for site in roster:
                m = (sites == site) & np.isfinite(da); den = err['old_stop'][m].sum()
                views[p]['realized_positive_harm_by_source'][site] = dict(rows=int(m.sum()),
                    positive_harm=float(np.maximum(err[p][m]-err['old_stop'][m], 0).sum()),
                    benefit=float(np.maximum(err['old_stop'][m]-err[p][m], 0).sum()),
                    reference_sum=float(den), ratio=float(np.maximum(err[p][m]-err['old_stop'][m], 0).sum()/den) if den > 0 else None)
        contrasts = {p: {s: metric(err['half_joint'], err[p], m) for s, m in masks.items()}
            for p in ('half_independent', 'half_hash', 'half_unary')}
        r = dict(identity=identity, verified=True, group=g['name'], views=views, contrasts=contrasts,
            rows=len(ids), known_ADE=int(np.isfinite(da).sum()), complete_rows=int(data['valid'][ids].all(1).sum()),
            coordinate_arrays_verified=4, metric_reductions_verified=checks,
            producer=g['fold'], controller=g['controller'], readout=g['roles']['readout_fold'],
            seed=g['seed'], event=g['event'], result_source='fresh_run_frozen_control_readout')
        immutable_json(path, r); refs.append(artifact(path)); beat('group_evaluated', group=g['name'], metrics=checks)
    assert len(refs) == 36
    immutable_json(PRIVATE/'evaluation_complete.json', dict(identity=identity, groups=refs, all_passed=True))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--phase', choices=('decide', 'evaluate'), required=True)
    p.add_argument('--resume', action='store_true'); args = p.parse_args(); PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); beat('phase_started', phase=args.phase)
        (decide if args.phase == 'decide' else evaluate)(args.resume)
        beat('phase_complete', phase=args.phase)


if __name__ == '__main__': main()
