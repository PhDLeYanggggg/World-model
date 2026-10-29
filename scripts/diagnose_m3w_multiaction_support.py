"""Full fixed-TRAIN action-bank feasibility; no learned selector or test access."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_reference_coverage as run
from scripts import run_m3w_train_feasibility as old
from scripts.analyze_m3w_reference_coverage_support import fit_packets_cached
from scripts.report_m3w_reference_coverage import bank
from scripts.report_m3w_reference_harm_forecast import write_text
from src.evaluation.m3w_forecast_support import reconstruct_fit_ids, aligned_predictions
from src.evaluation.m3w_multiaction_support import solve_action_bank
from src.evaluation.m3w_support_action_diagnostic import real_intervention_mask, causal_bank_envelope
from src.evaluation.m3w_native_metrics import native_errors
from src.world_model.m3w_european_source_forecast import baseline_numpy, BASELINES
from src.world_model.m3w_fixed_floor_probe import targets


def diagnostic(verify=False):
    cfg, pc, data, jobs, identity = run.load()
    protocol = run.PUBLIC/'multiaction_probe_protocol.md'
    rel = str(protocol.relative_to(ROOT))
    if subprocess.check_output(['git', 'show', 'HEAD:'+rel], cwd=ROOT) != protocol.read_bytes():
        raise ValueError('Commit the fixed bank/protocol before full-roster bank readout')
    support_path = run.PUBLIC/'support_diagnostic.json'
    support = json.loads(support_path.read_text())
    check = json.loads((run.PUBLIC/'support_verification.json').read_text())
    if not check['exact_replay'] or check['summary_sha256'] != run.digest(support_path):
        raise ValueError('Verified existing support required')
    controls = {(r['context'], r['source'], r['recording']):r for r in support['rows']}
    registration, _, ancestor = old.prior.admission()
    rows = []
    for context, items in sorted(registration['contexts'].items()):
        job = next(j for j in jobs if j['key'] == context.split('_controller')[0])
        ids = job['design']['held_ids']; refs = identity['controls'][job['key']]
        banks = dict(original_ADE=bank(refs['ADE_only_prediction'], ids),
                     harm_control=bank(refs['prediction'], ids))
        path = run.PRIVATE/'predictions'/(job['key']+'.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        if run.digest(path) != receipt['sha256'] or receipt['checkpoint'] != run.complete(job, identity)['checkpoint']:
            raise ValueError('Coverage prediction changed')
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids); banks['new'] = z['prediction'].copy()
        packets = fit_packets_cached(items, ancestor, True)
        for item in items:
            source = item['packet']['identity']['source']; args = packets[source]
            row_ids = reconstruct_fit_ids(data, item, args)
            physical = np.stack([baseline_numpy(data['history'][row_ids], k)-data['origin'][row_ids, None]
                                 for k in range(len(BASELINES))], axis=1)
            switch = args[0][:, -1].astype(bool)
            floor = np.where(switch[:, None, None], physical[:, 3], physical[:, 1])
            cost = lambda p:native_errors(p.astype(float)+data['origin'][row_ids, None],
                data['target_eval'][row_ids], data['valid'][row_ids], np.ones(len(row_ids)))[0]
            floor_cost = cost(floor); cv = data['baseline_ade'][row_ids, 1]
            rollouts = [*physical.transpose(1, 0, 2, 3)]
            for arm in ('original_ADE', 'new', 'harm_control'):
                rollouts.append(aligned_predictions(row_ids, ids, banks[arm]))
            labels = np.stack([targets(cv, floor_cost, cost(p), job['design']['easy_cut']) for p in rollouts], axis=1)
            envelope = np.stack([causal_bank_envelope(physical[:, 1], floor, p) for p in rollouts], axis=1)
            eligible = old.base.shared.eligible(args)
            allowed = np.stack([eligible & real_intervention_mask(p, floor) for p in rollouts], axis=1)
            np.testing.assert_array_equal(labels[:, 6], args[2])
            banks_to_solve = dict(physical_only=list(range(6)), physical_plus_original=list(range(7)),
                                 physical_plus_coverage=[*range(6), 7],
                                 original_neural_real_only=[6], coverage_neural_real_only=[7])
            for recording in sorted(set(args[5].astype(str))):
                take = args[5].astype(str) == recording
                frozen = controls[context, source, recording]
                for arm, index in (('original_ADE', 6), ('new', 7), ('harm_control', 8)):
                    reproduced, _ = old.api.solve(labels[take, index], envelope[take, index], eligible[take])
                    if reproduced != frozen['oracles'][arm]:
                        raise ValueError('Original single-candidate control changed')
                outcomes = {}
                for name, indices in banks_to_solve.items():
                    outcomes[name], _ = solve_action_bank(labels[take][:, indices], envelope[take][:, indices],
                                                          allowed[take][:, indices])
                known = np.isfinite(labels[:, 0]).all(1)
                reference_mass = float(floor_cost[take & known].sum())
                rows.append(dict(context=context, source=source, recording=recording, rows=int(take.sum()),
                    known_rows=int((take & known).sum()), unknown_rows=int((take & ~known).sum()),
                    reference_mass_counted_once=reference_mass, banks=outcomes))
            run.beat('multiaction_source_complete', context=context, source=source, record_views=len(rows))
    if len(rows) != 148:
        raise ValueError('Full fixed148-view roster required')
    names = list(rows[0]['banks'])
    summary = {name:dict(budget_feasible=sum(r['banks'][name]['budget_feasible'] for r in rows),
        positive_feasible=sum(r['banks'][name]['positive_feasible'] for r in rows),
        gain_lower_mass=sum(r['banks'][name]['optimal_gain_lower_mass'] or 0. for r in rows)) for name in names}
    comparisons = []
    for row in rows:
        physical = row['banks']['physical_only']; neural = row['banks']['physical_plus_original']
        if physical['budget_feasible'] and not neural['budget_feasible']:
            raise ValueError('Adding real action options cannot remove feasible choices')
        if physical['budget_feasible'] and neural['budget_feasible']:
            if neural['optimal_gain_lower_mass']+1e-8 < physical['optimal_gain_lower_mass']:
                raise ValueError('Neural action bank oracle must include physical alternatives')
            delta = neural['optimal_gain_lower_mass']-physical['optimal_gain_lower_mass']
        else:
            delta = None
        comparisons.append(dict(context=row['context'], source=row['source'], recording=row['recording'],
            incremental_neural_gain_mass=delta,
            incremental_neural_gain_pp=100*delta/row['reference_mass_counted_once']
                if delta is not None and row['reference_mass_counted_once'] > 0 else None))
    result = dict(result_source='fresh_run_label_aware_TRAIN_multiaction_feasibility',
        required_views=148, unique_source_recordings=len({(r['source'], r['recording']) for r in rows}),
        summary=summary, rows=rows, incremental_original_neural=comparisons,
        original_single_action_controls_replayed=3*148,
        action_names=[*BASELINES, 'original_ADE_neural', 'coverage_neural', 'harm_control_neural'],
        source_protocol_sha256=run.digest(protocol), source_support_sha256=run.digest(support_path),
        code_sha256=run.digest(Path(__file__)), api_sha256=run.digest(ROOT/'src/evaluation/m3w_multiaction_support.py'),
        source_forecaster_seed=43, noops_excluded=True, at_most_one_action_per_row=True,
        reference_denominator_not_multiplied_by_bank_size=True,
        new_selector_training=False, independent_roles_opened=False, learned_gain_claim=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    run.immutable_json(run.PUBLIC/'multiaction_support.json', result)
    lines = ['# Full TRAIN Multi-Action Feasibility', '',
        'Label-aware diagnostic only; no new selector, independent confirmation or deployment.',
        'All148fixed record/context views retained;74unique recordings. Source forecaster seed43 is fixed, not selected.',
        'At most one real action per row. Exact analytic floor no-ops are excluded.', '',
        '| Fixed bank | Budget feasible views | Positive feasible views | Completion-lower utility mass (descriptive) |',
        '|---|---:|---:|---:|']
    for name, row in summary.items():
        lines.append(f"| {name} | {row['budget_feasible']} | {row['positive_feasible']} | {row['gain_lower_mass']} |")
    positive = sum(r['incremental_neural_gain_mass'] is not None and r['incremental_neural_gain_mass'] > 1e-8
                   for r in comparisons)
    lines += ['', f'Original neural action adds positive oracle utility in {positive}/148record/context views.',
        'Raw utility masses mix image-coordinate source scales and are bookkeeping only, not a cross-domain metric.',
        'Per-record fixed-reference relative values are in the JSON; no independent CI or learned effect is claimed.',
        'All444single-action controls replay. The new bank does not rewrite their old gates.',
        'Favorable oracle feasibility is not causal learnability, scene-joint interaction lift or safe deployment.',
        'A prospective multi-action learner still needs training/validation and independent calibration.', '']
    write_text(run.PUBLIC/'multiaction_support.md', '\n'.join(lines))
    print(json.dumps(dict(exact_replay=verify, result_sha256=run.digest(run.PUBLIC/'multiaction_support.json'),
                         feasible={k:v['budget_feasible'] for k,v in summary.items()})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    with (run.PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        diagnostic(args.verify)


if __name__ == '__main__':
    main()
