"""Post-fit output-bound diagnosis on already-opened source development only."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_dimensionless_harm_forecast as run
from scripts.analyze_m3w_harm_forecast_postfit import immutable, digest
from scripts import run_m3w_train_feasibility as old
from src.world_model.m3w_native_forecast import pack_geometry
from src.world_model.m3w_european_source_forecast import baseline_torch
from src.world_model.m3w_baseline_relative_forecaster import past_motion_budget
from src.world_model.m3w_source_checkpoint import source_partition
from src.world_model.m3w_source_forest import fingerprint
from src.evaluation.m3w_output_bound_diagnostic import bound_relaxation
from src.evaluation.m3w_native_metrics import native_errors


def summarize(arrays, take, actual, cv):
    lower = arrays['independent_step_ball_ADE_infimum'][take]
    known = np.isfinite(lower)
    reference = cv[take][known]
    denom = float(reference.sum())
    gain = None if denom <= 0 else float(100*(reference-lower[known]).sum()/denom)
    return dict(rows=int(take.sum()), supported=int(known.sum()), unknown=int((~known).sum()),
        finite_CV_geometrically_representable=int(arrays['finite_CV_geometrically_representable'][take].sum()),
        CV_outside_bound=int(arrays['CV_outside_bound'][take].sum()),
        CV_boundary_or_numerically_ambiguous=int(arrays['CV_boundary_or_numerically_ambiguous'][take].sum()),
        exactly_zero_motion_budget=int(arrays['exactly_zero_motion_budget'][take].sum()),
        known_positive_error_infimum=int((lower[known] > 1e-6).sum()),
        geometry_oracle_ADE_gain_CV_percent=gain,
        actual_ADE_above_geometric_infimum_mean=None if not known.any() else
            float((actual[take][known]-lower[known]).mean()),
        label_aware_relaxation_not_realizable_prediction=True,
        coordinate_unit='image_pixel')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    _, _, data, jobs, identity = run.load()
    support_path = run.PUBLIC/'support_diagnostic.json'
    support = json.loads(support_path.read_text())
    verified = json.loads((run.PUBLIC/'support_verification.json').read_text())
    if not verified['exact_replay'] or verified['summary_sha256'] != digest(support_path):
        raise ValueError('Verified unchanged support readout required')
    registration, _, _ = old.prior.admission()
    all_rows, problematic = [], []
    for job in jobs:
        ids = job['design']['held_ids']
        pieces = {}
        for start in range(0, len(ids), 4096):
            local = ids[start:start+4096]
            inputs = pack_geometry(data['geometry'][local])
            inputs['baseline'] = baseline_torch(inputs['history'], job['design']['baseline_index'])
            budget = past_motion_budget(inputs)
            cv = baseline_torch(inputs['history'], 1)
            target = data['target_eval'][local]-data['origin'][local, None]
            result = bound_relaxation(inputs['baseline'].numpy(), budget.numpy(), cv.numpy(),
                                      target, data['valid'][local])
            for key, value in result.items():
                pieces.setdefault(key, []).append(value)
        arrays = {key:np.concatenate(value) for key,value in pieces.items()}
        prediction_path = run.PRIVATE/'predictions'/(job['key']+'.npz')
        receipt = json.loads(prediction_path.with_suffix('.json').read_text())
        if digest(prediction_path) != receipt['sha256'] or receipt['checkpoint'] != run.complete(job, identity)['checkpoint']:
            raise ValueError('Changed frozen forecast')
        with np.load(prediction_path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            actual = native_errors(z['prediction'].astype(float)+data['origin'][ids,None],
                data['target_eval'][ids], data['valid'][ids], np.ones(len(ids)))[0]
        lower = arrays['independent_step_ball_ADE_infimum']
        known = np.isfinite(lower)
        if (actual[known]+1e-3 < lower[known]).any():
            raise ValueError('Actual prediction violates the claimed geometric lower bound')
        cv_cost = data['baseline_ade'][ids, 1]
        all_rows.append(dict(trial=job['key'], prediction_sha256=receipt['sha256'],
            summary=summarize(arrays, np.ones(len(ids), bool), actual, cv_cost)))
        for row in support['rows']:
            if row['old']['budget_feasible'] and row['new']['budget_feasible']:
                continue
            context, source = row['context'], row['source']
            if context.split('_controller')[0] != job['key']:
                continue
            item = next(item for item in registration['contexts'][context]
                        if item['packet']['identity']['source'] == source)
            source_ids = np.flatnonzero(data['sites'] == source)
            train, _, partition = source_partition(data['recordings'][source_ids], data['frames'][source_ids], source)
            train_ids = source_ids[train]
            packet = item['packet']['identity']
            if partition != packet['partition'] or fingerprint(train_ids) != packet['training_ids_sha256']:
                raise ValueError('Frozen source partition differs')
            fit_ids = train_ids[np.isin(data['recordings'][train_ids].astype(str), item['partition']['fit_recordings'])]
            matched = next(a for a in support['alignment'] if a['context'] == context and a['source'] == source)
            import hashlib
            if hashlib.sha256(fit_ids.tobytes()).hexdigest() != matched['ids_sha256']:
                raise ValueError('Verified fitting row alignment changed')
            take = np.isin(ids, fit_ids) & (data['recordings'][ids].astype(str) == row['recording'])
            problematic.append(dict(context=context, source=source, recording=row['recording'],
                old_budget_feasible=row['old']['budget_feasible'], new_budget_feasible=row['new']['budget_feasible'],
                summary=summarize(arrays, take, actual, cv_cost)))
    result = dict(result_source='fresh_run_postfit_geometry_diagnostic',
        support_sha256=digest(support_path), code_sha256=digest(Path(__file__)),
        api_sha256=digest(ROOT/'src/evaluation/m3w_output_bound_diagnostic.py'),
        training_updates=0, model_selection=False, trials=all_rows, formerly_unsupported_views=problematic,
        oracle_uses_labels_only_for_diagnosis=True, inputs_to_trained_model_unchanged=True,
        radial_ball_closure_infimum_not_network_expressivity_guarantee=True,
        independent_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    immutable(run.PUBLIC/'output_bound_diagnosis.json', json.dumps(result, indent=2)+'\n', args.verify)
    lines = ['# Causal Output-Bound Diagnosis', '',
        'Post-fit geometry-only relaxation on previously opened source development.',
        'No new training, model selection, inference features or deployment change.', '',
        'The bound and causal CV are computed from observed history only. Future labels are used ',
        'solely to compute the independent-step closed-ball ADE infimum. This is a label-aware ',
        'oracle, not a neural prediction; finite-network capacity and causal learnability are not proved.', '',
        '| Trial | Rows | CV strictly inside or exact fixed | CV outside | Boundary/ambiguous | Zero-motion budget |',
        '|---|---:|---:|---:|---:|---:|']
    for row in all_rows:
        v = row['summary']
        lines.append(f"| {row['trial']} | {v['rows']} | {v['finite_CV_geometrically_representable']} | "
            f"{v['CV_outside_bound']} | {v['CV_boundary_or_numerically_ambiguous']} | {v['exactly_zero_motion_budget']} |")
    lines += ['', '## Formerly Unsupported TRAIN Views', '',
        '| Context / record | Rows | CV representable | CV outside | ADE lower-bound gain vs CV (%) | New candidate gap to geometric infimum (pixels) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in problematic:
        v = row['summary']
        lines.append(f"| {row['context']} / {row['source']}/{row['recording']} | {v['rows']} | "
            f"{v['finite_CV_geometrically_representable']} | {v['CV_outside_bound']} | "
            f"{v['geometry_oracle_ADE_gain_CV_percent']} | {v['actual_ADE_above_geometric_infimum_mean']} |")
    lines += ['', 'Representability of CV does not establish a safe neural policy. Remaining failure ',
        'may involve learning, context ambiguity or finite model capacity. The geometric oracle is ',
        'neither a model result nor evidence of generalization. Image pixels and raw frames only.', '']
    immutable(run.PUBLIC/'output_bound_diagnosis.md', '\n'.join(lines), args.verify)
    print(json.dumps(dict(exact_replay=args.verify, result_sha256=digest(run.PUBLIC/'output_bound_diagnosis.json'))))


if __name__ == '__main__':
    main()
