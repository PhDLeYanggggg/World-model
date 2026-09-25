"""Role-restricted, label-separated conversion for a frozen selection readout."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from scripts.build_m3w_european_squares_source import store_array
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv
from src.data_unification.m3w_european_squares_source import choose_query_frames, build_recording, INPUT_FIELDS
from src.world_model.m3w_european_source_forecast import pack_scene
from src.world_model.m3w_frozen_selection import require_selection


def verify_arrays(directory, receipt, digest):
    for value in receipt['arrays'].values():
        if digest(directory/value['name']) != value['sha256']:
            raise ValueError('Cached array changed')


def build(run, cfg, identity):
    root, private = run.ROOT, run.PRIVATE
    role_path = root/'outputs/publication_readiness_2026_09/european_squares_roles_v1/roles.json'
    roles = json.loads(role_path.read_text())
    for ref in roles['dependency_bindings'].values():
        if run.artifact(root/ref['path']) != dict(path=ref['path'], sha256=ref['sha256']):
            raise ValueError('Role dependency changed')
    manifest = json.loads((root/'outputs/publication_readiness_2026_09/european_squares_intake_v1/trajectory_manifest.json').read_text())
    source = manifest['private_file']; archive = root/source['path']
    run.beat('verify_archive', bytes=source['bytes'])
    if archive.stat().st_size != source['bytes'] or run.digest(archive) != source['sha256']:
        raise ValueError('Raw archive changed')
    rows = [r for r in roles['recordings'] if r['role'] == cfg['data_role']]
    if len(rows) != cfg['recordings'] or sorted({r['locality_group'] for r in rows}) != cfg['localities']:
        raise ValueError('Role cohort differs')
    if set(cfg['localities']) & set(sum(identity['producer_rosters'], [])):
        raise ValueError('Selection localities overlap source training')
    receipts = []
    with zipfile.ZipFile(archive) as z:
        for i, row in enumerate(rows):
            require_selection(roles, row['source_member'], purpose='frozen_model_selection_readout')
            directory = private/'records'/hashlib.sha256(row['source_member'].encode()).hexdigest()
            directory.mkdir(parents=True, exist_ok=True); path = directory/'receipt.json'
            if path.exists():
                r = json.loads(path.read_text())
                if r['identity'] != identity or r['rows_sha256'] != row['rows_sha256']:
                    raise ValueError('Recording identity changed')
                verify_arrays(directory, r, run.digest)
                receipts.append((directory, r)); run.beat('record_cached_verified', index=i+1); continue
            run.beat('build_record', index=i+1, total=len(rows))
            with z.open(row['source_member']) as f: raw, _ = read_raw_csv(f)
            if hashlib.sha256(raw.tobytes()).hexdigest() != row['rows_sha256']:
                raise ValueError('Raw recording row identity changed')
            queries, support = choose_query_frames(raw, maximum=cfg['query_cap_per_recording'])
            if support['all_past_eligible_targets'] != row['past_eligible_k8_stride12']:
                raise ValueError('Past-only support mismatch')
            inputs, labels = build_recording(raw, queries)
            checks = 0
            for qi in sorted({0, len(queries)//2, len(queries)-1}) if len(queries) else []:
                q = queries[qi]; prefix, _ = build_recording(raw[raw['frame'] <= q], [q])
                a, b = inputs['query_offsets'][qi:qi+2]
                for key in INPUT_FIELDS:
                    ref = np.array([0, b-a]) if key == 'query_offsets' else inputs[key][a:b]
                    np.testing.assert_array_equal(prefix[key], ref)
                checks += 1
            del raw
            needed = sum(v.nbytes for v in (*inputs.values(), *labels.values()))
            if shutil.disk_usage(private).free < needed+10*1024**3:
                raise OSError('Below10GiB reserve; retain completed records')
            arrays = {}
            for kind, values in [('input', inputs), ('label', labels)]:
                for key, value in values.items():
                    p = directory/(kind+'_'+key+'.npy'); store_array(p, value)
                    arrays[kind+'_'+key] = dict(name=p.name, sha256=run.digest(p), shape=list(value.shape), bytes=p.stat().st_size)
            nfuture = labels['future_valid'].sum(1)[inputs['target_eligible']]
            r = dict(identity=identity, source_member=row['source_member'], rows_sha256=row['rows_sha256'],
                locality_group=row['locality_group'], role=cfg['data_role'], arrays=arrays, support=support,
                queries=len(queries), visible_rows=len(inputs['agent_id']), targets=len(nfuture),
                complete=int((nfuture == 12).sum()), partial=int(((nfuture > 0)&(nfuture < 12)).sum()),
                unknown=int((nfuture == 0).sum()), future_truncation_checks=checks)
            run.immutable_json(path, r); receipts.append((directory, r))
            run.beat('record_complete', index=i+1, targets=len(nfuture))
    packed = pack(run, receipts, identity)
    public = dict(result_source='fresh_run', records=len(rows), localities=cfg['localities'],
        totals={k: sum(r[k] for _, r in receipts) for k in ('queries', 'visible_rows', 'targets', 'complete', 'partial', 'unknown', 'future_truncation_checks')},
        packed=run.artifact(private/'packed/receipt.json'), input_fields=packed['input_fields'],
        role=cfg['data_role'], future_targets_in_input=False, future_eligibility_filter=False,
        risk_calibration_opened=False, confirmation_opened=False, no_new_training=True,
        raw_archive_sha256=source['sha256'], registration=identity)
    run.immutable_json(run.PUBLIC/'data_audit.json', public)
    run.beat('data_complete', **public['totals'])


def pack(run, receipts, identity):
    directory = run.PRIVATE/'packed'; directory.mkdir(parents=True, exist_ok=True)
    path = directory/'receipt.json'
    if path.exists():
        r = json.loads(path.read_text())
        if r['identity'] != identity: raise ValueError('Packed identity changed')
        verify_arrays(directory, r, run.digest); return r
    n = sum(r['targets'] for _, r in receipts)
    specs = dict(geometry=('float32', (n, 476)), history=('float64', (n, 8, 2)), origin=('float64', (n, 2)),
        sites=('<U20', (n,)), recordings=('int32', (n,)), frames=('int64', (n,)), agents=('int64', (n,)),
        width=('float64', (n,)), target_eval=('float64', (n, 12, 2)), valid=('bool', (n, 12)))
    if shutil.disk_usage(directory).free < 10*1024**3+sum(np.dtype(dt).itemsize*np.prod(sh) for dt, sh in specs.values()):
        raise OSError('Packed data would cross10GiB floor')
    out = {k: np.lib.format.open_memmap(directory/(k+'.npy'), mode='w+', dtype=dt, shape=sh) for k, (dt, sh) in specs.items()}
    cursor = 0
    for ri, (source, r) in enumerate(receipts):
        arrays = {k: np.load(source/v['name'], allow_pickle=False, mmap_mode='r') for k, v in r['arrays'].items()}
        inputs = {k.removeprefix('input_'): v for k, v in arrays.items() if k.startswith('input_')}
        for a, b in zip(inputs['query_offsets'][:-1], inputs['query_offsets'][1:]):
            scene = {k: v[a:b] for k, v in inputs.items() if k != 'query_offsets'}
            g, local = pack_scene(scene); ids = a+local; dest = slice(cursor, cursor+len(g))
            h = inputs['history_xy'][ids]
            for k, v in dict(geometry=g, history=h, origin=h[:, -1], sites=r['locality_group'], recordings=ri,
                    frames=inputs['query_frame'][ids], agents=inputs['agent_id'][ids],
                    width=inputs['history_boxes'][ids, -1, 2]-inputs['history_boxes'][ids, -1, 0],
                    target_eval=arrays['label_future_xy'][ids], valid=arrays['label_future_valid'][ids]).items(): out[k][dest] = v
            cursor += len(g)
        run.beat('packed_record', index=ri+1, rows=cursor)
    if cursor != n: raise ValueError('Incomplete target population')
    for v in out.values(): v.flush()
    r = dict(identity=identity, rows=n, input_fields=[k for k in out if k not in ('target_eval', 'valid')],
        label_fields=['target_eval', 'valid'], arrays={k: dict(name=k+'.npy', sha256=run.digest(directory/(k+'.npy')),
        bytes=(directory/(k+'.npy')).stat().st_size) for k in out}, records=[run.artifact(d/'receipt.json') for d, _ in receipts])
    run.immutable_json(path, r); return r


def load(run, identity, *, labels=False):
    directory = run.PRIVATE/'packed'; r = json.loads((directory/'receipt.json').read_text())
    if r['identity'] != identity: raise ValueError('Packed identity changed')
    fields = r['label_fields'] if labels else r['input_fields']
    out = {}
    for k in fields:
        ref = r['arrays'][k]
        if run.digest(directory/ref['name']) != ref['sha256']: raise ValueError('Packed array hash mismatch')
        out[k] = np.load(directory/ref['name'], mmap_mode='r', allow_pickle=False)
    return out
