"""Independent accounting and light reports for the frozen CV-reference repair."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/publication_readiness_2026_09'
PUBLIC = BASE/'european_cv_reference_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(m):
    v = m['equal_scene_gain_percent']
    return 'undefined' if v is None else f'{v:.6f}'


def ci(m):
    v = m['scene_bootstrap_ci95']
    return 'undefined' if v is None else f'[{v[0]:.6f}, {v[1]:.6f}]'


def observed_safety_label(m):
    if m['zero_CV']['rows'] == 0:
        return 'zero-event support absent; not certified'
    return str(m['safety_observed_pass'])+'; not certified'


def equal_errors(a, b):
    for field in ('indexed_rows', 'supported_rows', 'unknown_rows', 'expected_scenes'):
        if a[field] != b[field]:
            raise ValueError('Original readout population changed: '+field)
    if set(a['by_scene']) != set(b['by_scene']):
        raise ValueError('Original locality roster changed')
    for site, row in a['by_scene'].items():
        for field in ('model_error', 'reference_error', 'gain_percent', 'rows'):
            if row[field] != b['by_scene'][site][field]:
                raise ValueError('Frozen readout differs: '+site+' '+field)


def require_verification(public, result):
    expected_sha = sha(public/'analysis.json')
    verification = json.loads((public/'verification.json').read_text())
    replay = json.loads((public/'checkpoint_replay.json').read_text())
    if any(v['analysis_sha256'] != expected_sha for v in (verification, replay)):
        raise ValueError('Complete metric and checkpoint verification required')
    if verification.get('metrics_recomputed') is not True or replay.get('all_passed') is not True:
        raise ValueError('Verification must succeed, not merely name the same artifact')
    checks = replay.get('checks', [])
    if (len(checks) != len(result['training'])
            or {c['trial'] for c in checks} != set(result['training'])
            or any(c.get('exact') is not True or c.get('rows', 0) <= 0
                   or c.get('max_difference') != 0 for c in checks)):
        raise ValueError('Every head needs a positive-support exact replay')


def main():
    r = json.loads((PUBLIC/'analysis.json').read_text())
    require_verification(PUBLIC, r)
    old = json.loads((BASE/'european_source_intervention_v1/analysis.json').read_text())
    forecast = json.loads((BASE/'european_source_forecast_v1/analysis.json').read_text())
    packed = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed'
    packed_receipt = json.loads((packed/'receipt.json').read_text())
    expected = r['identity']['previous_identity']['parent_identity']
    if packed_receipt['identity'] != expected:
        raise ValueError('Source population identity differs')
    arrays = {}
    for name in ('recordings', 'frames', 'sites', 'baseline_ade', 'history', 'valid'):
        path = packed/(name+'.npy')
        if sha(path) != packed_receipt['arrays'][name]:
            raise ValueError('Accounting input changed: '+name)
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    matched_checks, neural_checks, previous_checks = [], [], []
    for key, row in r['seeds'].items():
        seed = key.split('_')[0]
        equal_errors(row['full']['neural']['ADE_vs_CV'], forecast['seeds'][seed]['ADE_vs_CV'])
        equal_errors(row['full']['previous_pointwise']['ADE_vs_CV'], old['seeds'][key]['full']['pointwise']['ADE_vs_CV'])
        neural_checks.append(key); previous_checks.append(key)
    for artifact in r['controls']:
        receipt_path = ROOT/artifact['path']
        if sha(receipt_path) != artifact['sha256']:
            raise ValueError('Frozen control receipt changed')
        receipt = json.loads(receipt_path.read_text())
        path = ROOT/receipt['path']
        if sha(path) != receipt['sha256']:
            raise ValueError('Frozen decisions changed')
        with np.load(path, allow_pickle=False) as z:
            expected_rows = sum(q['agents'] for q in receipt['queries'])
            if len(z['ids']) != expected_rows:
                raise ValueError('Query/row accounting differs')
            # Reconstruct query blocks from public count summaries plus private keys.
            rec = arrays['recordings'][z['ids']]
            frame = arrays['frames'][z['ids']]
            keys = np.column_stack((rec, frame))
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            if len(unique) != len(receipt['queries']):
                raise ValueError('Actual query population differs')
            count = 0
            for i, q in enumerate(receipt['queries']):
                use = inverse == i
                if unique[i].tolist() != [q['recording'], q['frame']] or use.sum() != q['agents']:
                    raise ValueError('Actual past query keys differ')
                if q['matched']:
                    counts = [int(z[k][use].sum()) for k in ('independent', 'unary_exact', 'joint_exact')]
                    if len(set(counts)) != 1 or counts[0] != q['reference_count']:
                        raise ValueError('Actual matched decisions violate count equality')
                    if not np.all(z['matched'][use]) or not np.all(z['matched_nonzero'][use] == (counts[0] > 0)):
                        raise ValueError('Matched masks differ from actual decisions')
                    count += 1
            matched_checks.append(dict(path=artifact['path'], queries=len(unique), matched_count_checks=count))
    cv = arrays['baseline_ade'][:, 1]
    zero = np.isfinite(cv) & (cv == 0)
    sites = arrays['sites']
    folds = np.array([r['identity']['folds'][s] for s in sites])
    step_speed = np.linalg.norm(np.diff(arrays['history'][zero], axis=1), axis=2)
    zero_support = dict(total=int(zero.sum()), localities={s: int((zero & (sites == s)).sum()) for s in sorted(set(sites))},
        outer_folds={str(f): dict(fitting_zero_rows=int((zero & (folds != f)).sum()),
            held_zero_rows=int((zero & (folds == f)).sum())) for f in range(3)},
        moving_history_rows=int((step_speed.sum(1) > 0).sum()),
        last_velocity_zero_rows=int((step_speed[:, -1] == 0).sum()),
        observed_future_label_counts=arrays['valid'][zero].sum(1).tolist(),
        requested_final_endpoint_available=int(arrays['valid'][zero, -1].sum()),
        scope='posthoc_source_support_diagnostic_not_inference_filter_or_guarantee')
    audit = dict(result_source='fresh_arithmetic_on_cached_verified_inputs', new_training=False,
        analysis_sha256=sha(PUBLIC/'analysis.json'), reporter_sha256=sha(Path(__file__)),
        original_neural_readouts_unchanged=neural_checks, original_policy_readouts_unchanged=previous_checks,
        actual_query_count_checks=matched_checks, zero_event_support=zero_support,
        all_passed=True, deployment_changed=False)
    (PUBLIC/'accounting_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    lines = ['# CV-Reference Repair Results', '',
        'Fresh cost fitting and fixed control readout on cached_verified neural forecasts. All source-development results; independent reserved data remain closed.',
        'The original neural and previous-policy per-locality readouts reproduce exactly. Actual matched intervention counts are checked from decision arrays.',
        '', '## Full Registered Cohort', '',
        '| Seed/head | Rule | ADE gain vs CV (%) | Conditional CI | ADE vs selected baseline (%) | ADE vs fixed damping097 (%) | Easy gain vs CV (%) | Worst easy gain (%) | Zero-CV harmed | Switch rate |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for key, row in r['seeds'].items():
        for rule, m in row['full'].items():
            lines.append(f"| {key} | {rule} | {value(m['ADE_vs_CV'])} | {ci(m['ADE_vs_CV'])} | {value(m['ADE_vs_training_selected_baseline'])} | {value(m['ADE_vs_fixed_damping097'])} | {value(m['positive_easy_ADE_vs_CV'])} | {m['positive_easy_ADE_vs_CV']['worst_scene_gain_percent']} | {m['zero_CV']['harmed_rows']} | {m['switch_rate']:.6f} |")
    lines += ['', '## Matched Joint Population', '',
        f"{r['joint_rows']} targets, not all {r['source_rows']} forecast targets. Negative easy gain denotes degradation; observed risk is not certified risk.", '',
        '| Seed/head | Rule | ADE vs CV (%) | Conditional CI | Hard vs CV (%) | Easy vs CV (%) | Worst easy gain (%) | Zero-CV harmed | Switch rate | Observed safety |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---|']
    for key, row in r['seeds'].items():
        for rule, m in row['joint_population'].items():
            lines.append(f"| {key} | {rule} | {value(m['ADE_vs_CV'])} | {ci(m['ADE_vs_CV'])} | {value(m['hard_ADE_vs_CV'])} | {value(m['positive_easy_ADE_vs_CV'])} | {m['positive_easy_ADE_vs_CV']['worst_scene_gain_percent']} | {m['zero_CV']['harmed_rows']} | {m['switch_rate']:.6f} | {observed_safety_label(m)} |")
    lines += ['', '## Direct Paired Contrasts', '',
        '| Seed/head | Contrast | ADE gain (%) | Conditional CI |', '|---|---|---:|---|']
    for key, row in r['seeds'].items():
        for name, m in row['comparisons'].items():
            lines.append(f'| {key} | {name} | {value(m)} | {ci(m)} |')
    lines += ['', 'A change of fallback also changes the cost labels, causal disagreement features and fitting cost scale; these necessary derived changes are part of one reference-contract repair.',
        'Matching is within the new policy family. Old/new reference comparisons are on the same population but need not share intervention rates.',
        'No post-readout winner, threshold or checkpoint selection. No change to deployment, risk limits, held roles, metric/time claims, Stage5C or SMC.', '']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    losses = ['# CV-Reference Cost Training', '',
        'Real Torch cost training, not new forecasting training. Three seeds, fixed budgets, no validation or held-out checkpoint selection.', '',
        '| Head | Steps | First loss | Last logged loss | Training seconds | Unknown rows drawn |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in r['training'].items():
        f = row['fit']
        if 'trace' in f:
            losses.append(f"| {name} | {f['step']} | {f['trace'][0]['loss']:.6f} | {f['trace'][-1]['loss']:.6f} | {f['seconds']:.3f} | {f['unknown_rows_sampled']} |")
        else:
            losses.append(f'| {name} | 0 (closed-form ridge) | n/a | n/a | n/a | n/a |')
    losses += ['', 'Training losses use a fitting-only cost scale and asymmetric harm penalty; they are not held-site errors or calibrated harm bounds. The 100-update pilot is included in, not added to, 18,000 neural updates.', '']
    (PUBLIC/'training_losses.md').write_text('\n'.join(losses))
    print(json.dumps(dict(unchanged_neural_checks=len(neural_checks), unchanged_previous_checks=len(previous_checks),
        matched_receipts=len(matched_checks), result='passed')))


if __name__ == '__main__':
    main()
