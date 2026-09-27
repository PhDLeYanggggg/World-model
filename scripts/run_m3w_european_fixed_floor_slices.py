"""Fit-defined causal slices and evaluation-only label strata of frozen heads."""
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
from scripts import run_m3w_european_fixed_floor_excess as parent
from src.world_model import m3w_fixed_floor_slices as api
import numpy as np

PUBLIC = parent.PUBLIC.parent/'european_fixed_floor_slices_v1'
PRIVATE = parent.PRIVATE.parent/'european_fixed_floor_slices_v1'
CONFIG = 'configs/m3w_european_fixed_floor_slices_v1.json'
FILES = [CONFIG, 'scripts/run_m3w_european_fixed_floor_slices.py',
    'src/world_model/m3w_fixed_floor_slices.py', 'tests/test_m3w_fixed_floor_slices.py',
    'tests/test_m3w_fixed_floor_slices_protocol.py',
    str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def beat(state, **kw):
    row = dict(state=state, pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    parent.inter.json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def load(register=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    sealpath = parent.PUBLIC/'verification.json'
    assert parent.digest(sealpath) == cfg['parent_seal_sha256']
    seal = json.loads(sealpath.read_text())
    for p, h in seal['source_bindings'].items(): assert parent.digest(ROOT/p) == h, p
    for p, h in seal['artifacts'].items(): assert parent.digest(parent.PUBLIC/p) == h, p
    pc, data, jobs, oid, pid, pbound, bound = parent.load()
    assert cfg['groups'] == 108 and cfg['causal_quantiles'] == [.25, .75]
    assert not any(cfg[k] for k in ('new_training', 'threshold_search', 'independent_roles_read',
        'deployment_changed', 'stage5c_executed', 'smc_enabled'))
    identity = dict(parent_seal=parent.artifact(sealpath),
        bindings={p: parent.digest(ROOT/p) for p in FILES}, source_rows=len(data['sites']),
        sites=sorted(set(data['sites'])), registered_before_slice_readout=True)
    if register: parent.immutable_json(PUBLIC/'registration.json', identity)
    else:
        assert json.loads((PUBLIC/'registration.json').read_text()) == identity
        parent.inter.committed(PUBLIC/'registration.json')
    return cfg, data, jobs, oid, pid, pbound, bound, identity


def sliced_masks(axes, edges, floor, ref_edges, valid, scale):
    masks = [('all', 'all', np.ones(len(floor), bool))]
    for axis, values in axes.items():
        bins = api.assign_bins(values, edges[axis])
        masks.extend((axis, label, bins == i) for i, label in enumerate(('low', 'middle', 'high')))
    inside = axes['feature_radius_over_limit'] <= 1
    masks += [('existing_support', 'inside', inside), ('existing_support', 'outside', ~inside)]
    known = np.isfinite(floor)
    rb = np.full(len(floor), -1)
    rb[known] = api.assign_bins(floor[known]/scale, ref_edges)
    masks.extend(('reference_error_eval_only', label, rb == i)
        for i, label in ((-1, 'unknown'), (0, 'low'), (1, 'middle'), (2, 'high')))
    masks += [('label_completeness_eval_only', 'unknown', ~known),
        ('label_completeness_eval_only', 'partial', known & ~valid.all(1)),
        ('label_completeness_eval_only', 'complete', known & valid.all(1))]
    assert np.array_equal(known, valid.any(1))
    for axis in sorted(set(a for a, _, _ in masks)):
        np.testing.assert_array_equal(sum(m.astype(int) for a, _, m in masks if a == axis), np.ones(len(floor)))
    return masks


def run(cfg, data, jobs, oid, pid, pbound, bound, identity, *, resume=False, replay=False, limit=None):
    records = []; refs = []; fit_metadata = []
    for c in parent.floor_api.contexts(data, jobs, oid):
        cv, _, (floor, _), (neural, _) = parent.floor_api.costs(c, data, np.arange(len(c['ids'])))
        sites = data['sites'][c['ids']]
        for pair in range(6):
            name, fit, held, y, pr, old, ms, mse, ident = parent.control(c, data, pair, pid, pbound, bound)
            destination = PRIVATE/'groups'/(name+'.json')
            if destination.exists() and resume and not replay:
                doc = json.loads(destination.read_text()); assert doc['identity'] == identity
                assert doc['parent_head'] == parent.artifact(parent.PRIVATE/'heads'/name/'complete.json')
            else:
                if destination.exists() and not replay: raise ValueError('Use --resume to preserve completed groups')
                if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Keep10GiB disk reserve')
                complete = parent.done(parent.PRIVATE/'heads'/name/'complete.json', ident)
                state = parent.torch.load(parent.PRIVATE/'heads'/name/'checkpoint.pt', map_location='cpu', weights_only=False)
                ridge = parent.torch.load(parent.floor_api.PRIVATE/'fits'/name/'ridge.pt', map_location='cpu', weights_only=False)['model']
                axes = api.causal_axes(data['geometry'][c['ids']], c['floor'], c['prediction'], c['x'], pr, ridge['support_limit'])
                edges = {k: api.fit_edges(v[fit], api.source_weights(sites[fit])).tolist() for k, v in axes.items()}
                ref_edges = api.fit_edges(y[pr['known'], 0]/pr['cost_scale'], pr['weights'][pr['known']]).tolist()
                score_fit = {}; score_held = {'mse': mse}
                for arm, checkpoint in [('mse', ms), ('excess', state)]:
                    model = parent.api.initialize(pr, checkpoint['settings']['width'], checkpoint['seed'], checkpoint['mean_envelope'])
                    model.load_state_dict(checkpoint['model'])
                    score_fit[arm] = parent.api.predict(model, c['x'][fit], c['env'][fit], pr)
                    prediction = parent.api.predict(model, c['x'][held], c['env'][held], pr)
                    if arm == 'mse': np.testing.assert_array_equal(prediction, mse)
                    else:
                        with np.load(parent.PRIVATE/'heads'/name/'scores.npz', allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['ids'], c['ids'][held])
                            np.testing.assert_array_equal(prediction, z['scores'])
                    score_held[arm] = prediction
                new_records = []
                for role, pos, scores in [('fit', fit, score_fit), ('held', held, score_held)]:
                    original = parent.floor_api.score(ridge, c, pos) if role == 'fit' else old
                    eligible = c['moving'][pos] & original['support'] & (original['scores'][:, 5] > original['scores'][:, 6])
                    np.testing.assert_array_equal(original['support'], axes['feature_radius_over_limit'][pos] <= 1)
                    masks = sliced_masks({k: v[pos] for k, v in axes.items()}, edges,
                        floor[pos], ref_edges, data['valid'][c['ids'][pos]], pr['cost_scale'])
                    for arm, score in scores.items():
                        take = eligible & (score[:, 1] <= .02*score[:, 0]) & (score[:, 3] <= .02*score[:, 2])
                        if role == 'held':
                            with np.load(parent.PRIVATE/'decisions'/(name+'.npz'), allow_pickle=False) as z:
                                np.testing.assert_array_equal(z['ids'], c['ids'][pos])
                                np.testing.assert_array_equal(z[arm], take)
                                np.testing.assert_array_equal(z['eligible'], eligible)
                        q = parent.api.signed(score.astype(float))[:, 0]/pr['cost_scale']
                        for site in sorted(set(sites[pos])):
                            local = sites[pos] == site
                            for axis, label, mask in masks:
                                sums = api.evaluate_slice(local & mask, floor[pos], neural[pos], take, eligible, q, pr['cost_scale'])
                                new_records.append(dict(site=str(site), role=role, policy=arm,
                                    axis=axis, bin=label, group=name, sums=sums))
                metadata = dict(group=name, fit_sites=sorted(set(sites[fit])), held_sites=sorted(set(sites[held])),
                    edges=edges, reference_edges_eval_only=ref_edges, support_limit=float(ridge['support_limit']),
                    cost_scale=float(pr['cost_scale']), fit_ids_hash=parent.inter.array_hash(c['ids'][fit]),
                    fit_axes_hash={k: parent.inter.array_hash(v[fit]) for k, v in axes.items()},
                    feature_mask_does_not_read_future=True)
                doc = dict(identity=identity, parent_head=parent.artifact(parent.PRIVATE/'heads'/name/'complete.json'),
                    metadata=metadata, rows=new_records, head_training_reused=complete['fit']['complete'])
                parent.immutable_json(destination, doc)
            refs.append(parent.artifact(destination)); records.extend(doc['rows']); fit_metadata.append(doc['metadata'])
            beat('group_replayed' if replay else 'group_complete', group=name, complete=len(refs))
            if limit and len(refs) >= limit: return
    assert len(refs) == cfg['groups']
    summary = api.reduce_slices(records, identity['sites'], cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    parent.immutable_json(PUBLIC/'summary.json', dict(identity=identity,
        result_source='fresh_frozen_model_diagnosis_cached_verified_models', groups=len(refs),
        records=len(records), summary=summary, new_training=False, independent_roles_read=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    parent.immutable_json(PUBLIC/'completion.json', dict(identity=identity, groups=refs,
        summary=parent.artifact(PUBLIC/'summary.json'), held_predictions_replayed=216,
        frozen_decisions_replayed=216, strata_partition_checks=108*2*12))
    if replay: parent.immutable_json(PUBLIC/'replay.json', dict(groups=108, exact=True, new_training=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['register', 'run', 'replay'], required=True)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0: raise ValueError('Positive diagnostic group limit required')
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    parent.torch.set_num_threads(4); parent.torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat('started', phase=args.phase)
        values = load(args.phase == 'register')
        if args.phase != 'register': run(*values, resume=args.resume, replay=args.phase == 'replay', limit=args.limit)
        beat('complete', phase=args.phase)


if __name__ == '__main__': main()
