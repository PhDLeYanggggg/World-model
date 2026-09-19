"""Read admitted auxiliary arrays only; audit broader motion support and costs."""
import argparse
import json
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_population import (
    baseline_errors, summarize, complement_selected_baseline, scale_floor_mask,
)

CONFIG = 'configs/m3w_source_population_v1.json'
CODE = ('src/evaluation/m3w_source_population.py', 'scripts/audit_m3w_source_population.py',
        'tests/test_m3w_source_population.py',
        'outputs/publication_readiness_2026_09/source_population_decision.md')


def build():
    r = json.loads((ROOT/CONFIG).read_text())
    evidence = {p:file_digest(ROOT/p) for p in (CONFIG, *CODE)}
    for key in ('input_manifest', 'source_registration'):
        path = r[key]
        evidence[path] = file_digest(ROOT/path)
        if evidence[path] != r[key+'_sha256']:
            raise ValueError('Changed bound source: '+path)
    manifest = json.loads((ROOT/r['input_manifest']).read_text())
    if manifest['identity']['registration_sha256'] != r['source_registration_sha256']:
        raise ValueError('Changed source registration identity')
    allowed = set(r['allowed_sites'])
    if allowed & set(r['closed_sites']):
        raise ValueError('Closed site in audit')
    chunks, readouts = [], []
    root = (ROOT/r['input_manifest']).parent
    for rec in manifest['records']:
        recording = rec['recording']
        site = recording.split('/')[0]
        if site not in allowed:
            continue
        if rec['original_split'] != 'train' or rec['data_role'] != 'supervised_auxiliary_training':
            raise ValueError('Unapproved record role')
        arrays = {}
        for name in ('geometry', 'baseline', 'target', 'valid', 'scale', 'query_keys'):
            path = root/recording/(name+'.npy')
            digest = file_digest(path)
            if digest != rec['arrays'][name+'.npy']:
                raise ValueError('Changed source array: '+str(path))
            evidence[str(path.relative_to(ROOT))] = digest
            arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        g, y, valid, scale, keys = (arrays[k] for k in ('geometry', 'target', 'valid', 'scale', 'query_keys'))
        n = rec['rows']
        if len(g) != n or scale.shape != (n,) or keys.shape != (n, 2):
            raise ValueError('Row alignment failed')
        if not np.isfinite(scale).all() or np.any(scale <= 0) or not np.equal(keys, keys.astype(np.int64)).all():
            raise ValueError('Invalid past scale or identity')
        if len(np.unique(keys, axis=0)) != n:
            raise ValueError('Duplicate query identity')
        np.testing.assert_array_equal(g[:, 332:356].reshape(n, 12, 2), arrays['baseline'])
        ade, fde, event = baseline_errors(g, y, valid)
        complete = int(valid.all(1).sum())
        partial = int((valid.any(1) & ~valid.all(1)).sum())
        absent = int((~valid.any(1)).sum())
        if (complete, partial, absent) != (rec['complete_labels'], rec['partial_labels'], rec['absent_labels']):
            raise ValueError('Changed label support')
        static = np.all(g[:, :16] == 0, axis=1)
        chunk = dict(sites=np.repeat(site, n), recordings=np.repeat(recording, n),
            tracks=np.array([f'{recording}:{int(k)}' for k in keys[:, 1]]), frames=keys[:, 0].astype(np.int64),
            scale=np.array(scale), scale_floor=scale_floor_mask(scale),
            static_history=static, valid_count=valid.sum(1), event=event, ade=ade, fde=fde)
        chunks.append(chunk)
        readouts.append(dict(recording=recording, rows=n, complete=complete, partial=partial, absent=absent,
                             static_history=int(static.sum()), scale_floor=int(chunk['scale_floor'].sum())))
        print(json.dumps(readouts[-1]), flush=True)
    if len(chunks) != r['expected_recordings']:
        raise ValueError('Wrong admitted recording population')
    a = {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    if set(a['sites']) != allowed:
        raise ValueError('Wrong admitted site population')
    summary = summarize(a)
    summary['by_site'] = {s:summarize({k:v[a['sites'] == s] for k,v in a.items()}) for s in sorted(allowed)}
    summary['complement_selected'] = {}
    for name, mask in [('supported_masked', a['valid_count'] > 0), ('complete', a['valid_count'] == 12),
                       ('complete_moving', (a['valid_count'] == 12) & ~a['static_history'])]:
        summary['complement_selected'][name] = complement_selected_baseline(a['ade'][mask], a['sites'][mask])
    summary.update(result_source='fresh_run_aggregate_from_cached_verified_auxiliary_arrays',
        readouts=readouts, evidence_hashes=evidence, verified_arrays=len(readouts)*6,
        index_membership='all_past_eligible_rows_in_four_previously_explored_source_sites',
        event_label_source='cached_float32_supervision_complete_futures_only',
        original_val_test_rows=0, main_outer_external_rows=0, bookstore_rows=0,
        new_training=False, primary_metric_changed=False, independent_confirmation=False,
        coordinates='SDD_annotation_pixel_raw_frame_only', stage5c_executed=False, smc_enabled=False)
    return r, a, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    r, a, result = build()
    report = ROOT/r['reports']/'audit.json'
    archive = ROOT/r['private']/'cost_rows.npz'
    if args.verify:
        old = json.loads(report.read_text())
        if file_digest(archive) != old['row_archive']['sha256']:
            raise ValueError('Changed result archive')
        with np.load(archive, allow_pickle=False) as z:
            if set(z.files) != set(a):
                raise ValueError('Changed row schema')
            for k, v in a.items():
                np.testing.assert_array_equal(v, z[k])
        result['row_archive'] = old['row_archive']
        if result != old:
            raise ValueError('Replay report mismatch')
        print(json.dumps(dict(exact_replay=True, rows=len(a['scale']), verified_arrays=result['verified_arrays'])), flush=True)
        return
    if report.exists() or archive.exists():
        raise FileExistsError('Use --verify for completed runs; never overwrite evidence')
    archive.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **a)
    result['row_archive'] = dict(path=str(archive.relative_to(ROOT)), sha256=file_digest(archive))
    report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('rows','recordings','complete','partial','absent','static_history','scale_floor_rows')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
