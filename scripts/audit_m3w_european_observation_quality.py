"""Input-only source audit; future arrays and reserved recordings stay closed."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
import numpy as np
import torch
from scripts.fetch_m3w_european_squares import digest
from scripts.run_m3w_native_forecast import immutable_json, array_hash, json_write
from src.data_unification.m3w_european_squares_source import lookup
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv
from src.evaluation.m3w_european_squares_roles import require_source_training
from src.world_model.m3w_european_source_forecast import pack_scene
from src.world_model import m3w_observation_quality as method

BASE = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = BASE/'european_observation_quality_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_observation_quality_v1'
PACKED = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed'
CONFIG = 'configs/m3w_european_observation_quality_v1.json'
FILES = [CONFIG, 'src/world_model/m3w_observation_quality.py',
         'scripts/audit_m3w_european_observation_quality.py',
         'tests/test_m3w_observation_quality.py', str(PUBLIC.relative_to(ROOT)/'protocol.md')]


def artifact(p):
    return dict(path=str(p.relative_to(ROOT)), sha256=digest(p))


def beat(state, **values):
    row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), state=state, **values)
    json_write(PRIVATE/'heartbeat.json', row)
    with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)


def registration(create=False):
    cfg = json.loads((ROOT/CONFIG).read_text())
    seal_path = BASE/'european_cap_attribution_v1/verification.json'
    assert digest(seal_path) == cfg['parent_verification_sha256']
    seal = json.loads(seal_path.read_text())
    for path, sha in seal['source_bindings'].items(): assert digest(ROOT/path) == sha
    for name, sha in seal['artifacts'].items(): assert digest(seal_path.parent/name) == sha
    manifest_path = BASE/'european_squares_source_v1/analysis.json'
    assert digest(manifest_path) == cfg['source_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    roles_path = BASE/'european_squares_roles_v1/roles.json'
    assert digest(roles_path) == manifest['identity']['roles_sha256']
    roles = json.loads(roles_path.read_text())
    for ref in roles['dependency_bindings'].values(): assert digest(ROOT/ref['path']) == ref['sha256']
    assert manifest['totals']['target_rows'] == 318969 and len(manifest['record_receipts']) == 163
    packed = json.loads((PACKED/'receipt.json').read_text())
    assert packed['source_only'] and packed['rows'] == 318969
    assert packed['identity']['source_manifest_sha256'] == cfg['source_manifest_sha256']
    assert packed['records'] == [r['source_member'] for r in manifest['record_receipts']]
    identity = dict(source=artifact(manifest_path), roles=artifact(roles_path),
        parent_seal=artifact(seal_path), packed=artifact(PACKED/'receipt.json'),
        bindings={p:digest(ROOT/p) for p in FILES})
    dest = PUBLIC/'registration.json'
    if create: immutable_json(dest, identity)
    else:
        assert json.loads(dest.read_text()) == identity
        import subprocess
        subprocess.run(['git', 'ls-files', '--error-unmatch', str(dest.relative_to(ROOT))],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return manifest, roles, packed, identity


def audit_record(index, ref, roles, packed):
    before = time.monotonic()
    access = require_source_training(roles, ref['source_member'])
    home = ROOT/ref['directory']
    assert digest(home/'receipt.json') == ref['receipt_sha256']
    receipt = json.loads((home/'receipt.json').read_text())
    assert receipt['rows_sha256'] == access['rows_sha256'] and receipt['role'] == 'source_training'
    inp = {}
    for key, spec in receipt['arrays'].items():
        if key.startswith('input_'):
            path = home/spec['name']; assert digest(path) == spec['sha256']
            inp[key.removeprefix('input_')] = np.load(path, mmap_mode='r', allow_pickle=False)
    ids = np.flatnonzero(packed['recordings'] == index)
    targets = np.flatnonzero(inp['target_eligible'])
    np.testing.assert_array_equal(packed['frames'][ids], inp['query_frame'][targets])
    np.testing.assert_array_equal(packed['agents'][ids], inp['agent_id'][targets])
    np.testing.assert_array_equal(packed['history'][ids], inp['history_xy'][targets])
    diag = method.history_diagnostics(inp['history_xy'][targets], inp['history_boxes'][targets])
    changed = []; partial = []; old_count = []; new_count = []; missing = []
    cursor = 0
    for start, end in zip(inp['query_offsets'][:-1], inp['query_offsets'][1:]):
        scene = {k:v[start:end] for k,v in inp.items() if k != 'query_offsets'}
        g, target = pack_scene(scene)
        np.testing.assert_array_equal(g, packed['geometry'][ids[cursor:cursor+len(g)]])
        neighbors = method.masked_neighbors(scene)
        np.testing.assert_array_equal(target, neighbors['target'])
        new = method.repair_geometry(g, neighbors)
        changed.extend(np.any(new != g, axis=1))
        partial.extend(neighbors['partial_neighbors'])
        old_count.extend(neighbors['legacy_neighbor_count'])
        new_count.extend(neighbors['valid'][:, :, -1].sum(1))
        missing.extend((~neighbors['valid']).sum((1, 2)))
        cursor += len(g)
    assert cursor == len(ids)
    diag.update(neighbor_geometry_changed=np.asarray(changed), partial_nearest_neighbors=np.asarray(partial),
        has_partial_nearest_neighbor=np.asarray(partial)>0, legacy_neighbor_count=np.asarray(old_count),
        repaired_neighbor_count=np.asarray(new_count), repaired_missing_slots=np.asarray(missing))
    return inp, targets, ids, diag, dict(source=ref, rows=len(ids),
        packed_ids_sha256=array_hash(ids), cached_input_hashes={k:v['sha256'] for k,v in receipt['arrays'].items() if k.startswith('input_')},
        input_seconds=time.monotonic()-before)


def raw_check(z, ref, inp, targets, diag):
    with z.open(ref['source_member']) as f: rows, _ = read_raw_csv(f)
    receipt = json.loads((ROOT/ref['directory']/'receipt.json').read_text())
    assert hashlib.sha256(rows.tobytes()).hexdigest() == receipt['rows_sha256']
    result = np.zeros((len(targets), 3))
    agents, start, size = np.unique(rows['agent'], return_index=True, return_counts=True)
    target_agents = inp['agent_id'][targets]
    raw_pairs = 0
    for agent, first, count in zip(agents, start, size):
        positions = np.flatnonzero(target_agents == agent)
        if not len(positions): continue
        track = rows[first:first+count]; queries = inp['query_frame'][targets[positions]]
        grid = queries[:, None]-np.arange(7, -1, -1)*12
        boxes, valid = lookup(track, grid.ravel())
        assert valid.all()
        np.testing.assert_array_equal(boxes.reshape(-1, 8, 4), inp['history_boxes'][targets[positions]])
        result[positions] = method.prefix_continuity(track, queries)
        raw_pairs += len(positions)*8
    assert raw_pairs == len(targets)*8
    diag.update(raw_prefix_frame_presence=result[:, 0], raw_prefix_max_gap=result[:, 1],
                raw_prefix_mean_detector_confidence=result[:, 2])
    return dict(raw_rows=int(len(rows)), raw_history_boxes_matched=raw_pairs,
                raw_rows_sha256=receipt['rows_sha256'])


def run(phase, manifest, roles, packed_ref, identity):
    packed = {}
    for key in ('geometry', 'history', 'recordings', 'frames', 'agents', 'sites'):
        path = PACKED/(key+'.npy')
        assert digest(path) == packed_ref['arrays'][key]
        packed[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    src = json.loads((BASE/'european_squares_intake_v1/trajectory_manifest.json').read_text())['private_file']
    archive = ROOT/src['path']
    beat('checking_raw_archive', bytes=src['bytes'])
    assert digest(archive) == src['sha256']
    aggregate = {}; receipts = []; start = time.monotonic()
    with zipfile.ZipFile(archive) as z:
        for i, ref in enumerate(manifest['record_receipts']):
            if shutil.disk_usage(PRIVATE).free < 10*2**30: raise OSError('Preserve10GiB; resume per-record receipts')
            out = PRIVATE/'records'/f'{i:03d}'; out.mkdir(parents=True, exist_ok=True)
            complete = out/'complete.json'; cache = out/'diagnostics.npz'
            if complete.exists() and phase != 'verify':
                receipt = json.loads(complete.read_text())
                assert receipt['registration'] == artifact(PUBLIC/'registration.json')
                assert receipt['source'] == ref and receipt['diagnostics'] == artifact(cache)
                state = 'cached_verified'
            else:
                before = time.monotonic(); beat('record_started', index=i+1, total=163)
                inp, target, ids, diag, receipt = audit_record(i, ref, roles, packed)
                receipt.pop('input_seconds')
                receipt.update(raw_check(z, ref, inp, target, diag))
                receipt['registration'] = artifact(PUBLIC/'registration.json')
                receipt['summary'] = {k:method.summarize(v) for k,v in diag.items()}
                if phase == 'verify':
                    with np.load(cache, allow_pickle=False) as saved:
                        assert set(saved.files) == set(diag)
                        for k in diag: np.testing.assert_array_equal(saved[k], diag[k])
                else:
                    temp = cache.with_suffix('.tmp')
                    with temp.open('wb') as f: np.savez_compressed(f, **diag)
                    temp.replace(cache)
                receipt['diagnostics'] = artifact(cache)
                immutable_json(complete, receipt)
                state = 'raw_replay_exact' if phase == 'verify' else 'fresh_run'
                beat(state, index=i+1, seconds=time.monotonic()-before, rows=len(ids))
            receipts.append(artifact(complete))
            with np.load(cache, allow_pickle=False) as saved:
                site = aggregate.setdefault(ref['locality_group'], {})
                for k in saved.files: site.setdefault(k, []).append(saved[k].copy())
            beat('record_complete', index=i+1, source=state)
            if phase == 'pilot':
                immutable_json(PRIVATE/'pilot.json', dict(records=1, seconds_excluding_preflight=time.monotonic()-start,
                    free_GiB=shutil.disk_usage(PRIVATE).free/2**30, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                return
    per_site = {s:{k:method.summarize(np.concatenate(v)) for k,v in stats.items()} for s,stats in aggregate.items()}
    pooled = {k:method.summarize(np.concatenate([np.concatenate(v[k]) for v in aggregate.values()])) for k in next(iter(aggregate.values()))}
    doc = dict(result_source='fresh_run_raw_source_input_audit', cached_inputs='cached_verified',
        source_rows=318969, recordings=163, localities=12, per_site=per_site, pooled=pooled,
        receipts=receipts, registration=artifact(PUBLIC/'registration.json'),
        raw_archive_sha256=src['sha256'], full_legacy_geometry_exact=True,
        raw_history_boxes_matched=318969*8, future_labels_read=False, independent_roles_read=False,
        new_training=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    immutable_json(PUBLIC/'audit.json', doc)
    if phase == 'verify':
        immutable_json(PUBLIC/'raw_replay.json', dict(exact=True, recordings=163, rows=318969,
            audit_sha256=digest(PUBLIC/'audit.json'), all_diagnostic_arrays_exact=True))
    beat('complete', phase=phase, seconds_excluding_preflight=time.monotonic()-start,
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['register', 'pilot', 'run', 'verify'], required=True)
    args = p.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True); PUBLIC.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        beat('started', phase=args.phase, architecture=platform.machine(), workers=0, threads=4)
        manifest, roles, packed, identity = registration(args.phase == 'register')
        if args.phase != 'register': run(args.phase, manifest, roles, packed, identity)
        if args.phase == 'register': beat('complete', phase=args.phase)


if __name__ == '__main__': main()
