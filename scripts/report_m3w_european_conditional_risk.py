"""Independent accounting and aggregate reports for event-risk experiments."""
import hashlib
import json
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch checkpoint verification')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.report_m3w_european_cv_reference import equal_errors, require_verification, value, ci

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_conditional_risk_v1'
OLD = ROOT/'outputs/publication_readiness_2026_09/european_cv_reference_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((PUBLIC/'analysis.json').read_text())
    require_verification(PUBLIC, result)
    old = json.loads((OLD/'analysis.json').read_text())
    reference_checks = []
    for seed, refs in result['references'].items():
        previous = old['seeds'][seed+'_neural_underharm4']
        for name in ('CV', 'training_selected_baseline', 'fixed_damping097', 'neural', 'pointwise'):
            for field in ('ADE_vs_CV', 'FDE_vs_CV', 'positive_easy_ADE_vs_CV'):
                equal_errors(refs[name]['full'][field], previous['full'][name][field])
            reference_checks.append(seed+'_'+name)
        for name in ('independent', 'scene_uniform', 'joint', 'unary_exact', 'joint_exact'):
            equal_errors(refs[name]['joint_population']['ADE_vs_CV'], previous['joint_population'][name]['ADE_vs_CV'])
    draw_checks = []
    for fold in range(3):
        for seed in (17, 29, 43):
            cps = []
            for event in ('all', 'easy'):
                name = f'complement{fold}_seed{seed}_{event}_neural_underharm4'
                receipt = result['training'][name]
                p = ROOT/receipt['artifacts']['checkpoint']['path']
                if sha(p) != receipt['artifacts']['checkpoint']['sha256']:
                    raise ValueError('Checkpoint changed during sampler audit')
                cp = torch.load(p, map_location='cpu', weights_only=False)
                if cp['identity']['identity'] != result['identity'] or cp['step'] != 2000:
                    raise ValueError('Wrong checkpoint or update budget')
                cps.append(cp)
            for field in ('draws', 'sampler_rng'):
                np.testing.assert_array_equal(cps[0][field], cps[1][field])
            for field in ('mean', 'std', 'weights', 'known'):
                np.testing.assert_array_equal(cps[0]['preprocess'][field], cps[1]['preprocess'][field])
            if cps[0]['preprocess']['cost_scale'] != cps[1]['preprocess']['cost_scale']:
                raise ValueError('Fitting normalization mismatch')
            draw_checks.append(dict(fold=fold, seed=seed, exact=True,
                total_draws=int(cps[0]['draws'].sum()), unique_rows=int((cps[0]['draws'] > 0).sum())))
    packed = ROOT/'data/stage_cvpr2027_experiments/european_source_forecast_v1/packed'
    packed_receipt = json.loads((packed/'receipt.json').read_text())
    expected = result['identity']['previous_identity']['previous_identity']['parent_identity']
    if packed_receipt['identity'] != expected:
        raise ValueError('Source population identity mismatch')
    arrays = {}
    for name in ('recordings', 'frames'):
        path = packed/(name+'.npy')
        if sha(path) != packed_receipt['arrays'][name]:
            raise ValueError('Source query arrays changed')
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    query_checks = []
    for artifact in result['controls']:
        path = ROOT/artifact['path']
        if sha(path) != artifact['sha256']:
            raise ValueError('Decision receipt changed')
        receipt = json.loads(path.read_text())
        path = ROOT/receipt['path']
        if sha(path) != receipt['sha256']:
            raise ValueError('Decision arrays changed')
        with np.load(path, allow_pickle=False) as z:
            keys = np.column_stack((arrays['recordings'][z['ids']], arrays['frames'][z['ids']]))
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            if len(unique) != len(receipt['queries']):
                raise ValueError('Wrong query count')
            matched = 0
            for i, q in enumerate(receipt['queries']):
                use = inverse == i
                if unique[i].tolist() != [q['recording'], q['frame']] or use.sum() != q['agents']:
                    raise ValueError('Wrong causal query population')
                if q['matched']:
                    counts = [int(z[k][use].sum()) for k in ('independent', 'unary_exact', 'joint_exact')]
                    if len(set(counts)) != 1 or counts[0] != q['reference_count']:
                        raise ValueError('Unmatched actual interventions')
                    if not np.all(z['matched'][use]) or not np.all(z['matched_nonzero'][use] == (counts[0] > 0)):
                        raise ValueError('Wrong matched population mask')
                    matched += 1
            if not receipt['source_support_available']:
                for k in ('pointwise', 'independent', 'scene_uniform', 'joint', 'unary_exact', 'joint_exact'):
                    if z[k].any():
                        raise ValueError('Unsupported fold did not abstain')
            query_checks.append(dict(path=artifact['path'], queries=len(unique), actual_matched_count_checks=matched,
                support_available=receipt['source_support_available']))
    audit = dict(result_source='fresh_accounting_on_cached_verified_artifacts',
        analysis_sha256=sha(PUBLIC/'analysis.json'), reporter_sha256=sha(Path(__file__)),
        unchanged_reference_checks=reference_checks, paired_sampler_checks=draw_checks,
        query_checks=query_checks, all_passed=True, new_training=False, deployment_changed=False)
    (PUBLIC/'accounting_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    lines = ['# Conditional Event-Risk Results', '',
        'Fresh moment-head fitting and fixed controls on cached_verified forecasts/utility. Source development only; all policies retained, no winner selection.',
        '', '## Full Pointwise Population', '',
        '| Policy (seed/event/head/support) | ADE vs CV (%) | Locality CI | ADE vs damping097 (%) | ADE vs previous policy (%) | Hard gain (%) | Easy degradation (%) | Worst easy degradation (%) | Zero-CV harmed | Switch rate |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, p in result['policies'].items():
        m = p['full']; e = m['positive_easy_ADE_vs_CV']
        lines.append(f"| {name} | {value(m['ADE_vs_CV'])} | {ci(m['ADE_vs_CV'])} | {value(m['ADE_vs_fixed_damping097'])} | {value(p['comparisons']['pointwise_vs_previous'])} | {value(m['hard_ADE_vs_CV'])} | {-e['equal_scene_gain_percent']:.6f} | {-e['worst_scene_gain_percent']:.6f} | {m['zero_CV']['harmed_rows']}/{m['zero_CV']['rows']} | {m['switch_rate']:.6f} |")
    lines += ['', '## Joint Pilot', '',
        '1,152 queries / 6,116 targets, not the full cohort. This pilot has no zero-CV cases; none of its safety flags certifies zero-event protection.', '',
        '| Policy | Rule | ADE vs CV (%) | Locality CI | Easy degradation (%) | Worst easy degradation (%) | Switch rate |',
        '|---|---|---:|---|---:|---:|---:|']
    for name, p in result['policies'].items():
        for rule, m in p['joint_population'].items():
            e = m['positive_easy_ADE_vs_CV']
            lines.append(f"| {name} | {rule} | {value(m['ADE_vs_CV'])} | {ci(m['ADE_vs_CV'])} | {-e['equal_scene_gain_percent']:.6f} | {-e['worst_scene_gain_percent']:.6f} | {m['switch_rate']:.6f} |")
    lines += ['', '## Paired Factorial Contrasts', '',
        '| Contrast | Rule | ADE gain (%) | Conditional CI |', '|---|---|---:|---|']
    for name, row in result['factorial_contrasts'].items():
        for rule, m in row.items():
            lines.append(f'| {name} | {rule} | {value(m)} | {ci(m)} |')
    lines += ['', '## Matched Intervention Contrasts', '',
        '| Policy | Contrast | ADE gain (%) | Conditional CI |', '|---|---|---:|---|']
    for name, row in result['policies'].items():
        for contrast, m in row['comparisons'].items():
            if contrast != 'pointwise_vs_previous':
                lines.append(f'| {name} | {contrast} | {value(m)} | {ci(m)} |')
    lines += ['', 'Undefined locality contrasts are retained, never zero-filled or recalculated after dropping unsupported localities.',
        'All intervals use 3,000 locality resamples conditional on these development sources and fitted models. They are not independent calibration or confirmation.',
        'Positive-easy mean/worst <=2% and zero-CV added harm0 are unchanged. A zero-switch policy is fallback, not neural contribution.',
        'No deployment, metric/seconds/physical-safety claim, Stage5C or SMC.', '']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    losses = ['# Event-Moment Training Losses', '',
        'Real Torch risk-head training; no new neural forecast training. First channel is reference-error mass, not benefit. Event masks change target magnitude, so lower raw loss across event arms does not imply better forecasting.', '',
        '| Head | Steps | First loss | Last logged loss | Fit-loop seconds | Unique drawn fitting rows |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in result['training'].items():
        f = row['fit']
        if 'trace' in f:
            losses.append(f"| {name} | {f['step']} | {f['trace'][0]['loss']:.8f} | {f['trace'][-1]['loss']:.8f} | {f['seconds']:.4f} | {f['unique_training_rows']} |")
        else:
            losses.append(f'| {name} | 0 (closed-form) | n/a | n/a | n/a | n/a |')
    losses += ['', 'The 100-update pilot is included in 36,000 neural updates. All/easy arms use exactly matching sampled row counts and sampler states for each seed/fold; this is verified from checkpoints, not inferred from matching seeds.',
        'Losses are training minibatch losses, not held-data errors, convergence proof, calibrated risk or a model-selection objective.', '']
    (PUBLIC/'training_losses.md').write_text('\n'.join(losses))
    print(json.dumps(dict(reference_checks=len(reference_checks), sampler_pairs=len(draw_checks),
                         decision_receipts=len(query_checks), all_passed=True)))


if __name__ == '__main__':
    main()
