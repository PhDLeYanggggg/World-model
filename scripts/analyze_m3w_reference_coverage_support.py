"""Exact original/harm-aware/new candidate support on a fixed TRAIN roster."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time

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
from scripts.report_m3w_reference_coverage import bank, committed_freeze
from scripts.report_m3w_reference_harm_forecast import write_text
from src.evaluation.m3w_forecast_support import reconstruct_fit_ids, aligned_predictions
from src.evaluation.m3w_native_metrics import native_errors
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_floor_relative import matched_features
from src.world_model.m3w_fixed_floor_probe import targets
from src.evaluation.m3w_train_packet_transport import verified_packet_bytes


def fit_packets_cached(items, ancestor, verify):
    # Same frozen bytes, unpacker and fitting-record filter; transport only.
    loader = old.direct.LOADER.census.heads.previous.execution
    from scripts.train_m3w_temporal_auxiliary_portable import unpack
    result = {}
    for item in items:
        ref = item['packet']; source = ref['identity']['source']

        def fetch():
            args = [ref['group'], ancestor['input_manifest_sha256'], ancestor['parent_registration_sha256']]
            process = subprocess.run(loader.base.ssh_args()+[
                shlex.join(['/usr/bin/python3', '-c', loader.PACKET, *args])],
                capture_output=True, timeout=300)
            if process.returncode:
                message = process.stderr.decode(errors='replace')[-2000:]
                run.beat('read_only_packet_stream_failed', source=source,
                         returncode=process.returncode, stderr=message)
                raise RuntimeError('Existing TRAIN stream unavailable; no job or authentication change: '+message)
            return process.stdout

        raw, status = verified_packet_bytes(ref, run.PRIVATE/'verified_TRAIN_packets', fetch,
                                           allow_fetch=not verify)
        run.beat('support_packet_verified', source=source, transport=status, sha256=ref['sha256'])
        result[source] = old.fit_packet(unpack(raw), item['partition'])
    return result


def run_diagnostic(verify=False):
    started = time.monotonic(); committed_freeze()
    cfg, pc, data, jobs, identity = run.load()
    readout = json.loads((run.PUBLIC/'verification.json').read_text())
    if not readout['exact_analysis_replay'] or readout['analysis_sha256'] != run.digest(run.PUBLIC/'analysis.json'):
        raise ValueError('Exact predictive readout replay required')
    previous = json.loads((run.parent.PUBLIC/'support_diagnostic.json').read_text())
    check = json.loads((run.parent.PUBLIC/'support_verification.json').read_text())
    if not check['exact_replay'] or check['summary_sha256'] != run.digest(run.parent.PUBLIC/'support_diagnostic.json'):
        raise ValueError('Verified preceding support required')
    frozen_harm = {(r['context'], r['source'], r['recording']):r['new'] for r in previous['rows']}
    reg, _, ancestor = old.prior.admission()
    completion = json.loads((old.PUBLIC/'complete.json').read_text())
    if completion['summary_sha256'] != run.digest(old.PUBLIC/'summary.json'):
        raise ValueError('Original support summary changed')
    originals = {Path(ref['path']).name:ref for ref in completion['groups']}
    rows, alignment = [], []
    for context, items in sorted(reg['contexts'].items()):
        job = next(j for j in jobs if j['key'] == context.split('_controller')[0])
        ids = job['design']['held_ids']
        path = run.PRIVATE/'predictions'/(job['key']+'.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        if receipt['checkpoint'] != run.complete(job, identity)['checkpoint'] or run.digest(path) != receipt['sha256']:
            raise ValueError('New candidate cache changed')
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(ids, z['ids']); newbank = z['prediction'].copy()
        refs = identity['controls'][job['key']]
        banks = dict(new=newbank, harm_control=bank(refs['prediction'], ids),
                     original_ADE=bank(refs['ADE_only_prediction'], ids))
        packets = fit_packets_cached(items, ancestor, verify)
        for item in items:
            source = item['packet']['identity']['source']; args = packets[source]
            row_ids = reconstruct_fit_ids(data, item, args)
            pred = {arm:aligned_predictions(row_ids, ids, values) for arm,values in banks.items()}
            switch = args[0][:, -1]
            if not np.isin(switch, [0., 1.]).all():
                raise ValueError('Exact frozen causal floor switch required')
            cv_roll = baseline_numpy(data['history'][row_ids], 1)-data['origin'][row_ids, None]
            d_roll = baseline_numpy(data['history'][row_ids], 3)-data['origin'][row_ids, None]
            floor = np.where(switch[:, None, None].astype(bool), d_roll, cv_roll)
            x, env = matched_features(data['geometry'][row_ids], cv_roll, floor, pred['original_ADE'], switch.astype(bool))
            np.testing.assert_array_equal(x, args[0]); np.testing.assert_array_equal(env, args[1])
            cost = lambda p:native_errors(p.astype(float)+data['origin'][row_ids, None], data['target_eval'][row_ids],
                                         data['valid'][row_ids], np.ones(len(row_ids)))[0]
            cv_cost = data['baseline_ade'][row_ids, 1]; floor_cost = cost(floor)
            labels = {arm:targets(cv_cost, floor_cost, cost(p), job['design']['easy_cut']) for arm,p in pred.items()}
            np.testing.assert_array_equal(labels['original_ADE'], args[2])
            envelopes = {arm:matched_features(data['geometry'][row_ids], cv_roll, floor, p, switch.astype(bool))[1]
                         for arm,p in pred.items()}
            eligible = old.base.shared.eligible(args)
            ref = originals[f'{context}_{source}_seed17_fit.json']
            if run.digest(ROOT/ref['path']) != ref['sha256']:
                raise ValueError('Original oracle recording changed')
            original = json.loads((ROOT/ref['path']).read_text())
            for recording in sorted(set(args[5].astype(str))):
                take = args[5].astype(str) == recording
                oracles = {arm:old.api.solve(y[take], envelopes[arm][take], eligible[take])[0]
                           for arm,y in labels.items()}
                if oracles['original_ADE'] != original['result']['recordings'][recording]['oracle']:
                    raise ValueError('Original148-view oracle replay failed')
                if oracles['harm_control'] != frozen_harm[context, source, recording]:
                    raise ValueError('Preceding147-view oracle replay failed')
                support = {}
                for arm, y in labels.items():
                    use = take & eligible & np.isfinite(y).all(1) & (y[:, 3] > 0)
                    ratio = y[use, 4]/y[use, 3]
                    support[arm] = dict(known_eligible_easy_rows=int(use.sum()),
                        individually_within_2percent=int((ratio <= .02+1e-12).sum()),
                        minimum_easy_H_over_R=float(ratio.min()) if len(ratio) else None,
                        minimum_status='defined' if len(ratio) else 'not_estimable')
                rows.append(dict(context=context, source=source, recording=recording, oracles=oracles,
                                 easy_support=support))
            alignment.append(dict(context=context, source=source, rows=len(row_ids),
                packet_sha256=item['packet']['sha256'], original_features_envelopes_labels_exact=True,
                eligibility_frozen=True))
        run.beat('support_context_complete', context=context, record_views=len(rows))
    if len(rows) != 148 or len(frozen_harm) != 148:
        raise ValueError('Every required148record/context view must remain')
    counts = {arm:dict(budget_feasible=sum(r['oracles'][arm]['budget_feasible'] for r in rows),
                      positive_feasible=sum(r['oracles'][arm]['safe_positive_selection_exists'] for r in rows))
              for arm in ('original_ADE', 'harm_control', 'new')}
    transitions = {arm:dict(gained=sum(not r['oracles'][arm]['budget_feasible'] and
                                    r['oracles']['new']['budget_feasible'] for r in rows),
                           lost=sum(r['oracles'][arm]['budget_feasible'] and not
                                    r['oracles']['new']['budget_feasible'] for r in rows))
                   for arm in ('original_ADE', 'harm_control')}
    result = dict(result_source='fresh_run_label_aware_TRAIN_support_cached_verified_controls',
        required_views=148, unique_source_recordings=len({(r['source'], r['recording']) for r in rows}),
        counts=counts, transitions=transitions, rows=rows, alignment=alignment,
        original_and_harm_control_replay_exact=True, original_eligibility_and_floor_frozen=True,
        readout_sha256=run.digest(run.PUBLIC/'analysis.json'),
        preceding_support_sha256=run.digest(run.parent.PUBLIC/'support_diagnostic.json'),
        code_sha256=run.digest(Path(__file__)), oracle_is_not_inference=True,
        transport_api_sha256=run.digest(ROOT/'src/evaluation/m3w_train_packet_transport.py'),
        candidate_dependent_cost_heads_retrained=False, independent_roles_opened=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    run.immutable_json(run.PUBLIC/'support_diagnostic.json', result)
    lines = ['# Fixed-Floor Reference-Coverage Support', '',
        'TRAIN-only label-aware oracle, not deployment or causal learnability evidence.',
        'Exact original floor, eligibility and all148record/context views retained.',
        f'Feasible counts: {counts}.', f'Transitions versus controls: {transitions}.', '',
        '| Context / recording | Original | Harm-aware | Reference coverage | New minimum easy H/R |',
        '|---|---|---|---|---:|']
    for row in rows:
        if all(row['oracles'][arm]['budget_feasible'] for arm in counts):
            continue
        values = [row['oracles'][arm]['budget_feasible'] for arm in counts]
        lines.append(f"| {row['context']} / {row['source']}/{row['recording']} | {values[0]} | {values[1]} | "
                     f"{values[2]} | {row['easy_support']['new']['minimum_easy_H_over_R']} |")
    lines += ['', 'The risk ratio threshold remains0.02, not2. Empty risk is not_estimable.',
        'Six unchanged models cannot repair their preceding failures. New forecasts need new causal costs ',
        'and independent calibration before deployment. No Stage5C/SMC, metric or seconds claim.', '']
    write_text(run.PUBLIC/'support_diagnostic.md', '\n'.join(line.rstrip() for line in lines))
    runtime = dict(pid=os.getpid(), seconds=time.monotonic()-started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, exact_replay=verify,
        summary_sha256=run.digest(run.PUBLIC/'support_diagnostic.json'))
    run.json_write(run.PRIVATE/('support_replay_runtime.json' if verify else 'support_runtime.json'), runtime)
    if verify:
        run.immutable_json(run.PUBLIC/'support_verification.json', runtime)
    run.beat('support_complete', verify=verify, feasible=counts['new']['budget_feasible'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    with (run.PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        run_diagnostic(args.verify)


if __name__ == '__main__':
    main()
