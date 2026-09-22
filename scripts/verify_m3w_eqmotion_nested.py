"""Separate lineage, sampler, native-cost arithmetic and cache alignment audit."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_m3w_eqmotion_nested import load
from scripts.run_m3w_native_forecast import assert_current, immutable_json, array_hash
from scripts.verify_m3w_native_joint_controls import distances
from src.evaluation.m3w_experiment_contract import file_digest


def audit_lineage(producer, outer, row_site, seed, sites):
    forbidden = {outer, row_site}
    if (not forbidden <= set(sites) or producer['seed'] != seed
            or producer['family'] != 'eqmotion_fixed_head'
            or set(producer['excluded_sites']) != forbidden
            or set(producer['training_sites']) != set(sites)-forbidden
            or set(producer['preprocessing_fit_sites']) != set(sites)-forbidden
            or producer['parents'] or producer['initialization'] != 'random_seed'
            or producer['checkpoint_selection_sites'] or producer['calibration_sites']
            or set(producer['research_design_exposed_sites']) != set(sites)):
        raise ValueError('Invalid independent producer lineage check')


def main():
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, cfg, data, _, identity, references, outer = load()
    root, public = ROOT/reg['output'], ROOT/reg['reports']
    report = json.loads((public/'analysis.json').read_text())
    replay = json.loads((public/'verification_with_replay.json').read_text())
    manifest = json.loads((root/'cost_views.json').read_text())
    assert report['identity'] == identity == manifest['identity']
    assert replay['all_checks_passed'] and not replay['full_checkpoint_replay']
    assert replay['analysis_sha256'] == file_digest(public/'analysis.json')
    assert report['views_manifest_sha256'] == file_digest(root/'cost_views.json')
    assert len(report['training']) == 18 and len(manifest['views']) == 12
    n = len(data['sites'])
    b = data['geometry'][:, 332:356].reshape(n, 12, 2)
    cv, cf = distances(b, data['target'], data['valid'], data['scale'])
    checked_rows, caches, unknown = 0, {}, 0
    for r in report['training']:
        key, pair = r['trial'], r['excluded_sites']
        assert file_digest(ROOT/r['checkpoint']) == r['checkpoint_sha256']
        cp = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=False)
        ref = torch.load(ROOT/references[key]['checkpoint'], map_location='cpu', weights_only=False)
        ids = np.flatnonzero(np.isin(data['sites'], pair))
        train = np.flatnonzero(~np.isin(data['sites'], pair))
        assert len(pair) == len(set(pair)) == 2 and cp['identity']['identity'] == identity
        assert cp['step'] == ref['step'] == 4000 and cp['settings'] == cfg['training']
        assert cp['seed'] in reg['seeds']
        for a, c in ((pair[0], pair[1]), (pair[1], pair[0])):
            audit_lineage(cp['identity']['producer'], a, c, cp['seed'], reg['sites'])
        np.testing.assert_array_equal(cp['train_ids'], train)
        for field in ('train_ids', 'factors', 'draws'):
            np.testing.assert_array_equal(cp[field], ref[field])
        assert cp['draws'][ids].sum() == 0 and cp['draws'].sum() == 4000*64
        for site in set(data['sites'][train]):
            use = data['sites'] == site; supported = use & np.isfinite(cv)
            expected = data['scale'][use]*use.sum()/supported.sum()/cv[supported].mean()
            np.testing.assert_allclose(cp['factors'][use], expected, rtol=1e-12, atol=1e-10)
        meta = json.loads((root/'cache_receipts'/f'{key}.json').read_text())
        assert meta['identity'] == cp['identity']
        assert meta['checkpoint_sha256'] == r['checkpoint_sha256']
        assert meta['row_ids_sha256'] == array_hash(ids)
        for field in ('prediction', 'supervision'):
            assert file_digest(ROOT/meta[field+'_path']) == meta[field+'_sha256']
        with np.load(ROOT/meta['prediction_path'], allow_pickle=False) as z:
            assert set(z.files) == {'ids', 'prediction'}
            np.testing.assert_array_equal(z['ids'], ids)
            prediction = z['prediction'].copy()
        ade, fde = distances(prediction, data['target'][ids], data['valid'][ids], data['scale'][ids])
        gain = cv[ids]-ade
        expected = dict(neural_ade=ade, neural_fde=fde, baseline_ade=cv[ids], baseline_fde=cf[ids],
            gain=gain, benefit=np.clip(gain, 0, None), harm=np.clip(-gain, 0, None),
            supported_steps=data['valid'][ids].sum(1), complete_future=data['valid'][ids].all(1))
        with np.load(ROOT/meta['supervision_path'], allow_pickle=False) as z:
            assert set(z.files) == {'ids', *expected}
            np.testing.assert_array_equal(z['ids'], ids)
            for field, values in expected.items():
                np.testing.assert_allclose(z[field], values, rtol=1e-12, atol=1e-9, equal_nan=True)
        checked_rows += len(ids); unknown += int(np.isnan(gain).sum())
        assert meta['rows'] == len(ids) and meta['unknown_ADE_rows'] == int(np.isnan(gain).sum())
        caches[key] = meta
    exclusions = 0
    for view in manifest['views']:
        site, seed = view['outer_site'], view['seed']; coverage = np.zeros(n, int)
        assert view['outer_producer'] == outer[f'{site}_seed{seed}']
        audit_lineage(view['outer_producer']['producer'], site, site, seed, reg['sites'])
        for group in view['groups']:
            inner = group['inner_site']; ids = np.flatnonzero(data['sites'] == inner)
            assert inner != site
            audit_lineage(group['producer'], site, inner, seed, reg['sites'])
            key = '__'.join(sorted([site, inner]))+f'_seed{seed}'
            assert group['cache'] == caches[key] and group['rows'] == len(ids)
            assert group['query_ids_sha256'] == array_hash(ids)
            coverage[ids] += 1; exclusions += 1
        np.testing.assert_array_equal(coverage, (data['sites'] != site).astype(int))
        assert view['training_query_ids_sha256'] == array_hash(np.flatnonzero(coverage))
        assert not view['risk_head_fitted'] and not view['risk_calibrated']
    assert checked_rows == report['producer_cache_rows'] == 1581804
    assert unknown == report['unknown_cost_rows'] and exclusions == 36
    assert report['training_draws'] == 4608000 and report['optimizer_updates'] == 72000
    result = dict(analysis_sha256=file_digest(public/'analysis.json'),
        verifier_sha256=file_digest(Path(__file__)),
        arithmetic_helper_sha256=file_digest(ROOT/'scripts/verify_m3w_native_joint_controls.py'),
        all_checks_passed=True, matched_fits=18, cost_rows_checked=checked_rows,
        unknown_cost_rows_preserved=unknown, ordered_exclusions=exclusions, views=12,
        independent_implementation_same_agent=True, independent_research_confirmation=False,
        new_training=False, selection_changes=False, risk_head_fitted=False, deployment=False)
    assert_current(identity); immutable_json(public/'independent_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
