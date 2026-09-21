"""Verify and physically partition nested OOF predictions and cost labels."""
import argparse
import json
from pathlib import Path

from run_m3w_native_nested import ROOT, load, specification, write_arrays
from run_m3w_native_forecast import array_hash, assert_current, immutable_json
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_native_cost_views import assemble_training_view, LABEL_KEYS
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    reg, previous, data, rows, identity, outer = load()
    root, public = ROOT/reg['output'], ROOT/reg['reports']
    manifest = json.loads((root/'cost_views.json').read_text())
    analysis = json.loads((public/'analysis.json').read_text())
    verification = json.loads((public/'verification_with_replay.json').read_text())
    if (manifest['identity'] != identity or analysis['identity'] != identity
            or not verification['all_checks_passed']
            or verification['analysis_sha256'] != file_digest(public/'analysis.json')
            or analysis['views_manifest_sha256'] != file_digest(root/'cost_views.json')
            or len(manifest['views']) != 12):
        raise ValueError('Verified completed nested cache required')
    summary, view_keys = [], set()
    for view in manifest['views']:
        site, seed = view['outer_site'], view['seed']
        if site not in reg['sites'] or seed not in reg['seeds'] or (site, seed) in view_keys:
            raise ValueError('Wrong or duplicate head view')
        view_keys.add((site, seed))
        key = f'{site}_seed{seed}'
        groups = []
        for group in view['groups']:
            inner = group['inner_site']
            _, ti, producer = specification(reg, data, identity, [site, inner], seed)
            meta = json.loads((root/'cache_receipts'/(producer+'.json')).read_text())
            if meta != group['cache'] or meta['identity'] != ti or ti['producer'] != group['producer']:
                raise ValueError('Head view disagrees with registered producer')
            for file_key, sha_key in [('prediction_path', 'prediction_sha256'), ('supervision_path', 'supervision_sha256')]:
                if file_digest(ROOT/meta[file_key]) != meta[sha_key]:
                    raise ValueError('Changed prediction or target archive')
            with np.load(ROOT/meta['prediction_path'], allow_pickle=False) as z:
                if set(z.files) != {'ids', 'prediction'}:
                    raise ValueError('Unexpected inference members')
                ids, prediction = z['ids'].copy(), z['prediction'].copy()
            with np.load(ROOT/meta['supervision_path'], allow_pickle=False) as z:
                if set(z.files) != {'ids', *LABEL_KEYS}:
                    raise ValueError('Unexpected label members')
                np.testing.assert_array_equal(ids, z['ids'])
                labels = {k:z[k].copy() for k in LABEL_KEYS}
            groups.append(dict(inner_site=inner, producer=group['producer'],
                ids=ids, prediction=prediction, labels=labels))
        inputs, targets = assemble_training_view(site, seed, data['sites'], groups)
        if array_hash(inputs['ids']) != view['training_query_ids_sha256'] or len(inputs['ids']) != view['training_rows']:
            raise ValueError('Head training query alignment changed')
        directory = root/'head_training'/key
        pred_path, target_path = directory/'inputs.npz', directory/'targets.npz'
        if args.verify and not (pred_path.exists() and target_path.exists() and (directory/'manifest.json').exists()):
            raise ValueError('Verify cannot create missing materialized views')
        write_arrays(pred_path, inputs); write_arrays(target_path, targets)
        receipt = dict(source_manifest_sha256=file_digest(root/'cost_views.json'),
            outer_site=site, seed=seed, rows=len(inputs['ids']),
            query_ids_sha256=array_hash(inputs['ids']),
            inputs_path=str(pred_path.relative_to(ROOT)), inputs_sha256=file_digest(pred_path),
            targets_path=str(target_path.relative_to(ROOT)), targets_sha256=file_digest(target_path),
            permitted_prediction_members=['prediction'], alignment_only_members=['ids'],
            past_geometry_reference='bound_parent_geometry_by_global_ids_via_pack_geometry_only',
            forbidden_prediction_members=list(LABEL_KEYS),
            training_producers=[g['producer'] for g in view['groups']],
            outer_prediction=view['outer_producer'],
            independent_confirmation=False, risk_head_fitted=False, risk_calibrated=False)
        immutable_json(directory/'manifest.json', receipt)
        summary.append(dict(outer_site=site, seed=seed, rows=len(inputs['ids']),
            supported_cost_rows=int(np.isfinite(targets['gain']).sum()),
            unknown_cost_rows=int(np.isnan(targets['gain']).sum()),
            outer_rows_in_training=int(np.sum(data['sites'][inputs['ids']] == site)),
            receipt_path=str((directory/'manifest.json').relative_to(ROOT)),
            receipt_sha256=file_digest(directory/'manifest.json')))
        print(json.dumps(summary[-1]), flush=True)
    assert_current(identity)
    result = dict(result_source='fresh_run_partition_export_cached_verified_nested_predictions',
        source_analysis_sha256=file_digest(public/'analysis.json'),
        exporter_sha256=file_digest(Path(__file__)),
        assembler_sha256=file_digest(ROOT/'src/world_model/m3w_native_cost_views.py'),
        views=summary, head_training_views=len(summary),
        total_training_entries=sum(v['rows'] for v in summary),
        unique_source_queries=len(data['sites']), outer_rows_in_training=sum(v['outer_rows_in_training'] for v in summary),
        all_checks_passed=True, held_labels_in_training_files=False,
        independent_confirmation=False, new_training=False, risk_head_fitted=False, deployment=False)
    immutable_json(public/'materialized_views.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'views'}, indent=2))


if __name__ == '__main__':
    main()
