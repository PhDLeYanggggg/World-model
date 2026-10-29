"""Label-aware TRAIN case: real alternative actions, never a deployed selector."""
import argparse
import fcntl
import hashlib
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
from scripts import run_m3w_reference_coverage as run
from scripts import run_m3w_train_feasibility as old
from scripts.analyze_m3w_reference_coverage_support import fit_packets_cached
from scripts.report_m3w_reference_coverage import bank
from scripts.report_m3w_reference_harm_forecast import write_text
from src.evaluation.m3w_forecast_support import reconstruct_fit_ids, aligned_predictions
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_support_action_diagnostic import real_intervention_mask, causal_bank_envelope
from src.world_model.m3w_european_source_forecast import baseline_numpy, BASELINES
from src.world_model.m3w_fixed_floor_probe import targets

CONTEXT = 'single1_seed43_controller2_dimensionless'
SOURCE, RECORDING = 'eu-locality-082', '51'


def diagnostic(verify):
    cfg, pc, data, jobs, identity = run.load()
    support_path = run.PUBLIC/'support_diagnostic.json'
    support = json.loads(support_path.read_text())
    verification = json.loads((run.PUBLIC/'support_verification.json').read_text())
    if not verification['exact_replay'] or verification['summary_sha256'] != run.digest(support_path):
        raise ValueError('Verified fixed-roster support diagnosis required')
    original_case = next(r for r in support['rows'] if (r['context'], r['source'], r['recording']) ==
                         (CONTEXT, SOURCE, RECORDING))
    if original_case['oracles']['new']['budget_feasible']:
        raise ValueError('The prespecified failure case changed')
    registration, _, ancestor = old.prior.admission()
    item = next(item for item in registration['contexts'][CONTEXT] if item['packet']['identity']['source'] == SOURCE)
    args = fit_packets_cached([item], ancestor, True)[SOURCE]
    ids = reconstruct_fit_ids(data, item, args)
    take = args[5].astype(str) == RECORDING
    row_ids = ids[take]
    job = next(j for j in jobs if j['key'] == CONTEXT.split('_controller')[0])
    rolls = {name:baseline_numpy(data['history'][row_ids], index)-data['origin'][row_ids, None]
             for index, name in enumerate(BASELINES)}
    cv_roll, d_roll = rolls[BASELINES[1]], rolls[BASELINES[3]]
    switch = args[0][take, -1].astype(bool)
    floor = np.where(switch[:, None, None], d_roll, cv_roll)
    receipt = json.loads((run.PRIVATE/'predictions'/(job['key']+'.json')).read_text())
    path = ROOT/receipt['path']
    if run.digest(path) != receipt['sha256'] or receipt['checkpoint'] != run.complete(job, identity)['checkpoint']:
        raise ValueError('Frozen candidate changed')
    with np.load(path, allow_pickle=False) as z:
        rolls['coverage_neural'] = aligned_predictions(row_ids, z['ids'], z['prediction'])
    valid, target, origin = data['valid'][row_ids], data['target_eval'][row_ids], data['origin'][row_ids, None]
    cost = lambda p:native_errors(p.astype(float)+origin, target, valid, np.ones(len(row_ids)))[0]
    floor_cost = cost(floor); cv_cost = data['baseline_ade'][row_ids, 1]
    eligible = old.base.shared.eligible(args)[take]
    easy = np.isfinite(cv_cost) & (cv_cost > 0) & (cv_cost <= job['design']['easy_cut'])
    candidates, per_row = {}, []
    for name, prediction in rolls.items():
        errors = cost(prediction)
        labels = targets(cv_cost, floor_cost, errors, job['design']['easy_cut'])
        real = real_intervention_mask(prediction, floor)
        allowed = eligible & real
        envelope = causal_bank_envelope(cv_roll, floor, prediction)
        oracle, _ = old.api.solve(labels, envelope, allowed)
        known_easy = allowed & easy & np.isfinite(labels).all(1) & (labels[:, 3] > 0)
        ratio = labels[known_easy, 4]/labels[known_easy, 3]
        candidates[name] = dict(real_eligible_actions=int(allowed.sum()),
            exact_floor_noops_removed=int((eligible & ~real).sum()),
            known_real_easy_actions=int(known_easy.sum()),
            individually_safe_easy_actions=int((ratio <= .02+1e-12).sum()),
            minimum_real_easy_H_over_R=float(ratio.min()) if len(ratio) else None,
            label_aware_oracle=oracle)
        for i in np.flatnonzero(eligible & easy):
            per_row.append(dict(row_sha256=hashlib.sha256(row_ids[i].tobytes()).hexdigest(),
                candidate=name, real_intervention=bool(real[i]), floor_is_D97=bool(switch[i]),
                reference_ADE=float(floor_cost[i]), candidate_ADE=float(errors[i]),
                positive_harm_over_reference=float(max(errors[i]-floor_cost[i], 0)/floor_cost[i])
                    if floor_cost[i] > 0 else None))
    result = dict(result_source='fresh_run_label_aware_TRAIN_action_bank_diagnostic',
        context=CONTEXT, source=SOURCE, recording=RECORDING, fitting_rows=len(row_ids),
        eligible_rows=int(eligible.sum()), known_eligible_easy_rows=int((eligible & easy).sum()),
        candidates=candidates, easy_row_error_diagnostics=per_row,
        noops_are_not_interventions=True, threshold_search=False, new_selector_training=False,
        no_inference_policy_claim=True, original_support_gate_changed=False,
        source_packet_sha256=item['packet']['sha256'], support_sha256=run.digest(support_path),
        code_sha256=run.digest(Path(__file__)),
        api_sha256=run.digest(ROOT/'src/evaluation/m3w_support_action_diagnostic.py'),
        independent_roles_opened=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    run.immutable_json(run.PUBLIC/'remaining_case_diagnosis.json', result)
    lines = ['# Remaining TRAIN Case: Alternative Action Bank', '',
        'Post-fit label-aware case diagnosis; not a new trained/deployed selector or a changed primary gate.',
        f'Fixed case: {CONTEXT}, {SOURCE}/{RECORDING}.',
        f'Rows: {len(row_ids)}; known eligible easy rows: {int((eligible & easy).sum())}.',
        'All six existing analytic baselines plus the frozen new neural candidate are reported.',
        'Exact analytic equality to the floor is removed, not used as a fake safe intervention.', '',
        '| Candidate | Real eligible | Known real easy | Individually safe easy | Minimum easy H/R | Label-aware feasible |',
        '|---|---:|---:|---:|---:|---|']
    for name, row in candidates.items():
        lines.append(f"| {name} | {row['real_eligible_actions']} | {row['known_real_easy_actions']} | "
            f"{row['individually_safe_easy_actions']} | {row['minimum_real_easy_H_over_R']} | "
            f"{row['label_aware_oracle']['budget_feasible']} |")
    lines += ['', 'Future labels are diagnosis only, never inference inputs. No per-row oracle is a real selector.',
        'This action-bank probe does not rewrite the frozen148-view support diagnostic. Any new learned bank ',
        'would need prospective training/validation and independent calibration. Image pixels/raw frames only.', '']
    write_text(run.PUBLIC/'remaining_case_diagnosis.md', '\n'.join(line.rstrip() for line in lines))
    print(json.dumps(dict(exact_replay=verify, diagnosis_sha256=run.digest(run.PUBLIC/'remaining_case_diagnosis.json'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    with (run.PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        diagnostic(args.verify)


if __name__ == '__main__':
    main()
