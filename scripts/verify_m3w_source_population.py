"""Independent cost reduction and raw-row replay of the source-population audit."""
import json
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts.audit_m3w_sdd_state_support import load_source
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_forecast import geometry_features
from src.world_model.m3w_sdd_auxiliary import load_registration, source_entries
from src.world_model.m3w_sdd_step_adapter import SDDStepAdapter


def main():
    cfg = json.loads((ROOT/'configs/m3w_source_population_v1.json').read_text())
    path = ROOT/cfg['reports']/'audit.json'
    report = json.loads(path.read_text())
    for name, digest in report['evidence_hashes'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Changed audit dependency '+name)
    archive = ROOT/report['row_archive']['path']
    if file_digest(archive) != report['row_archive']['sha256']:
        raise ValueError('Changed private cost rows')
    with np.load(archive, allow_pickle=False) as z:
        a = {k:z[k] for k in z.files}
    source_reg = load_registration(ROOT, ROOT/cfg['source_registration'])
    manifest = json.loads((ROOT/cfg['input_manifest']).read_text())
    records = {r['recording']:r for r in manifest['records']}
    entries = source_entries(ROOT, source_reg)
    folder = (ROOT/cfg['input_manifest']).parent
    checks, inputs, poison, costs, evidence = [], 0, 0, 0, {}
    for entry in entries:
        key = entry['annotation_key']
        if key.split('/')[0] not in cfg['allowed_sites']:
            continue
        ids = np.flatnonzero(a['recordings'] == key)
        n = len(ids)
        if n != records[key]['rows']:
            raise ValueError('Recording alignment failed')
        load = lambda name:np.load(folder/key/(name+'.npy'), mmap_mode='r', allow_pickle=False)
        g, y, mask = load('geometry'), load('target'), load('valid')
        # Independent scalar-axis/time reduction, not baseline_errors().
        count = mask.sum(1)
        for k in range(7):
            x = g[:, 308+24*k:308+24*(k+1)].reshape(n, 12, 2).astype(float)
            d = np.sqrt((x[:, :, 0]-y[:, :, 0].astype(float))**2+
                        (x[:, :, 1]-y[:, :, 1].astype(float))**2)
            mean = np.divide((d*mask).sum(1), count, out=np.full(n, np.nan), where=count > 0)
            np.testing.assert_allclose(mean, a['ade'][ids, k], rtol=1e-13, atol=1e-12, equal_nan=True)
            np.testing.assert_allclose(np.where(mask[:, -1], d[:, -1], np.nan), a['fde'][ids, k],
                                       rtol=1e-13, atol=1e-12, equal_nan=True)
        costs += n*7
        raw_path = ROOT/entry['annotations_path']
        digest = file_digest(raw_path)
        if digest != entry['annotations_sha256'] or digest != records[key]['annotation_sha256']:
            raise ValueError('Changed raw annotation source')
        evidence[str(raw_path.relative_to(ROOT))] = digest
        raw, labels = load_source(raw_path)
        adapter = SDDStepAdapter(raw, labels, key, 12)
        if len(adapter) != n:
            raise ValueError('Raw past-index membership mismatch')
        current = adapter.points[adapter.index['current_row']]
        np.testing.assert_array_equal(current[:, :2].astype(np.int64), load('query_keys'))
        chosen = [0, n//2, n-1, int(np.nanargmax(a['ade'][ids, 1]))]
        for e in range(5):
            members = np.flatnonzero(a['event'][ids] == e)
            if len(members):
                chosen.append(int(members[0]))
        for q in np.unique(chosen):
            inp = adapter.get_inputs(q)
            np.testing.assert_array_equal(geometry_features(inp), g[q])
            target = adapter.get_labels(q)
            np.testing.assert_array_equal(target['future_xy_normalized'], y[q])
            np.testing.assert_array_equal(target['future_label_mask'], mask[q])
            inputs += 1
        # One query per recording tests that future arrays cannot alter inputs.
        q = int(chosen[len(chosen)//2])
        before = geometry_features(adapter.get_inputs(q))
        frame = adapter.identity(q)['frame_id']
        adapter.points[adapter.points[:, 0] > frame, 2:] = np.nan
        adapter.source[adapter.source[:, 5] > frame, 1:5] = np.nan
        np.testing.assert_array_equal(before, geometry_features(adapter.get_inputs(q)))
        poison += 1
        checks.append(dict(recording=key, index_rows=n, raw_queries=len(np.unique(chosen))))
        print(json.dumps(checks[-1]), flush=True)
    complete = a['valid_count'] == 12
    weights = np.zeros(len(a['scale']))
    for s in sorted(set(a['sites'])):
        m = complete & (a['sites'] == s)
        weights[m] = 1/(4*m.sum())
    cv = np.nan_to_num(a['ade'][:, 1], nan=0.)
    normalized = 100*np.dot(weights, cv*a['scale_floor'])/np.dot(weights, cv)
    native = 100*np.dot(weights, cv*a['scale_floor']*a['scale'])/np.dot(weights, cv*a['scale'])
    c = report['cohorts']['complete']
    np.testing.assert_allclose(normalized, c['scale_floor_cv_error_share']['percent_of_total'], rtol=1e-12)
    np.testing.assert_allclose(native, c['scale_floor_native_cv_error_share']['percent_of_total'], rtol=1e-12)
    out = dict(result_source='fresh_run_independent_reduction_and_raw_replay',
        audit_sha256=file_digest(path), row_archive_sha256=file_digest(archive),
        dependency_hashes_checked=len(report['evidence_hashes']), arrays_checked=report['verified_arrays'],
        raw_source_hashes=evidence, index_rows_rebuilt=sum(x['index_rows'] for x in checks),
        baseline_row_costs_recomputed=costs, raw_geometry_label_queries=inputs,
        future_array_poison_queries=poison, normalized_error_share_percent=float(normalized),
        native_error_share_percent=float(native), checks=checks,
        main_outer_bookstore_rows=0, primary_metric_changed=False, new_training=False,
        verifier_sha256=file_digest(Path(__file__)))
    output = ROOT/cfg['reports']/'verification.json'
    if output.exists():
        if json.loads(output.read_text()) != out:
            raise ValueError('Changed completed verification')
    else:
        output.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('checks','raw_source_hashes')}, indent=2))


if __name__ == '__main__':
    main()
