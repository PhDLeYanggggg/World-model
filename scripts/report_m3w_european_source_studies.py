"""Deterministic light tables from completed, verified source experiments."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/publication_readiness_2026_09'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return 'undefined' if value is None else f'{value:.4f}'


def interval(metric):
    value = metric.get('scene_bootstrap_ci95')
    return 'undefined' if value is None else f'[{value[0]:.4f}, {value[1]:.4f}]'


def gain(metric):
    return number(metric.get('equal_scene_gain_percent'))


def load(study):
    p = BASE/study
    result = json.loads((p/'analysis.json').read_text())
    for name in ('verification.json','checkpoint_replay.json'):
        verified = json.loads((p/name).read_text())
        if verified['analysis_sha256']!=sha(p/'analysis.json'):
            raise ValueError('Verification does not match current study')
    if result['independent_reserved_readout'] or result['deployment_changed']:
        raise ValueError('Source study boundaries changed')
    return p,result


def forecast():
    p,r = load('european_source_forecast_v1')
    table = ['# Source Forecast Results','',
        'Fresh Torch training and source-excluded inference; cached metric reproduction and separate fresh checkpoint replay pass.',
        'Twelve opened training localities, not reserved confirmation. Gains are equal-locality percentages. All coordinates are image pixels; the task is 8 observed / 12 requested raw-stride-12 steps.',
        'Outcome: average prediction signal, failed easy/zero-reference protection, no deployment promotion. The training-selected baseline is not the retrospective best fixed baseline on the entire readout.',
        '', '## Causal Controls','', '| Baseline | ADE gain vs CV (%) | Conditional 95% locality bootstrap |',
        '|---|---:|---|']
    for name,m in r['baselines'].items():
        table.append(f'| {name} | {gain(m)} | {interval(m)} |')
    table += ['', '## Neural Forecaster','',
        '| Seed | ADE gain vs fit-selected strongest (%) | 95% CI | FDE gain vs strongest (%) | ADE gain vs CV (%) | Easy gain vs CV (%) | Worst easy-locality gain (%) | Zero-CV harmed rows |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for seed,v in r['seeds'].items():
        m = v['ADE_vs_training_selected_strongest']
        table.append(f"| {seed} | {gain(m)} | {interval(m)} | {gain(v['FDE_vs_training_selected_strongest'])} | {gain(v['ADE_vs_CV'])} | {gain(v['positive_easy_ADE_vs_CV'])} | {number(v['positive_easy_ADE_vs_CV']['worst_scene_gain_percent'])} | {v['zero_CV']['harmed_rows']} |")
    table += ['',f"Mean-seed ADE gain vs selected strongest: **{gain(r['mean_seed_ADE_vs_training_selected_strongest'])}%**, conditional CI {interval(r['mean_seed_ADE_vs_training_selected_strongest'])}.",
        'This averages errors from separate seeds, not their predictions. A negative gain is degradation.',
        'On the same equal-locality CV-relative scale, fixed damping 0.97 gives +3.9755%, compared with +2.1576% for mean neural errors. The neural model therefore has not established superiority over every strong fixed control. This observation does not retrospectively change the registered comparison or select a replacement model.',
        'Positive-easy degradation is 13.65--14.39% across seeds, above the unchanged 2% ceiling. All four exact-zero-CV rows are harmed in each neural seed. The corresponding training-selected fallback itself degrades positive-easy errors by 15.48%; falling back to it cannot by itself certify CV-relative safety.',
        '', '## Per Locality','',
        '| Locality | Supported targets | Mean neural ADE (px) | Strongest ADE (px) | Gain (%) | Neural p95 (px) |',
        '|---|---:|---:|---:|---:|---:|']
    for site,v in r['mean_seed_ADE_vs_training_selected_strongest']['by_scene'].items():
        table.append(f"| {site} | {v['rows']} | {number(v['model_error'])} | {number(v['reference_error'])} | {number(v['gain_percent'])} | {number(v.get('model_p95'))} |")
    table += ['', 'The JSON retains every seed, baseline, hard/positive-easy and complete-future sensitivity, zero-reference harms, unknown label coverage and per-site tails.',
        'Bootstrap units are locality groups, never overlapping windows. Shared fitted models and opened source sites make these intervals conditional exploratory evidence.',
        'No cross-camera pixel pooling is interpreted as a common physical scale. Missing FDE endpoints remain unknown rather than using a last-valid endpoint.', '']
    (p/'results.md').write_text('\n'.join(table))
    losses = ['# Training Loss and Compute','',
        'These are training-minibatch losses, not validation or held-site metrics. No loss-based checkpoint selection was used.',
        'The objective is masked native ADE divided by a fitting-site baseline normalizer, with indexed/supported sampling correction.',
        'The inherited trace field `mean_past_normalized_ADE` actually stores the unweighted native-pixel minibatch ADE for this input path; its legacy name must not be interpreted as a normalized score.',
        '', '| Trial | Steps | Parameters | Fitting seconds | First logged loss | Mean last 5 logged losses | Held rows drawn |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v in r['training']:
        t,f = v['identity'],v['fit']
        last = [row['loss'] for row in f['losses'][-5:]]
        losses.append(f"| {t['kind']}{t['fold']}_seed{t['seed']} | {f['step']} | {f['parameters']} | {f['seconds']:.2f} | {f['losses'][0]['loss']:.6f} | {sum(last)/len(last):.6f} | {f['held_rows_sampled']} |")
    losses += ['', f"Total: {r['total_models']} fits, {r['total_optimizer_updates']} updates, {r['summed_fit_seconds']:.2f} summed fitting seconds. This excludes cache creation, inference and verification.",
        'Native arm64 CPU4/inter-op1/workers0. The real first 100 updates were resumed inside the fixed budget. No CUDA/MPS claim or NumPy substitute.', '']
    (p/'training_losses.md').write_text('\n'.join(losses))
    print(json.dumps(dict(study='forecast',gain_vs_strongest=r['mean_seed_ADE_vs_training_selected_strongest']['equal_scene_gain_percent'],
                         CI=r['mean_seed_ADE_vs_training_selected_strongest']['scene_bootstrap_ci95'])))


def intervention():
    p,r = load('european_source_intervention_v1')
    table = ['# Nested Intervention Results','',
        'Fresh nested source-cost fitting and source-excluded controls; cached metric reproduction and fresh cost-checkpoint replay pass.',
        'All results remain source-only method development. The predicted positive-harm cap does not certify the actual easy or exact-zero-reference limits.',
        '', '## Full Indexed Source Cohort','',
        '| Seed / cost head | Rule | ADE gain vs strongest (%) | Conditional 95% CI | FDE gain vs strongest (%) | Easy gain vs CV (%) | Worst easy-locality gain (%) | Zero-CV harmed rows | Switch rate |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for key,v in r['seeds'].items():
        for arm,m in v['full'].items():
            table.append(f"| {key} | {arm} | {gain(m['ADE_vs_floor'])} | {interval(m['ADE_vs_floor'])} | {gain(m['FDE_vs_floor'])} | {gain(m['positive_easy_ADE_vs_CV'])} | {number(m['positive_easy_ADE_vs_CV']['worst_scene_gain_percent'])} | {m['zero_CV']['harmed_rows']} | {number(m['switch_rate'])} |")
    table += ['', '## Fixed Joint-Control Population','',
        f"Exactly {r['joint_query_rows']} target rows at 1,152 past-selected query frames. This is not the full {r['source_rows']}-row forecast benchmark. All controls below share this population.",
        '', '| Seed / head | Rule | ADE gain vs floor (%) | Conditional 95% CI | Worst easy gain vs CV (%) | Zero-CV harmed | Switch rate |',
        '|---|---|---:|---|---:|---:|---:|']
    for key,v in r['seeds'].items():
        for arm,m in v['joint_population'].items():
            table.append(f"| {key} | {arm} | {gain(m['ADE_vs_floor'])} | {interval(m['ADE_vs_floor'])} | {number(m['positive_easy_ADE_vs_CV']['worst_scene_gain_percent'])} | {m['zero_CV']['harmed_rows']} | {number(m['switch_rate'])} |")
    table += ['', '## Matched Nonzero Intervention Queries','',
        'Matching uses the independent decision count before outcomes. Unary geometry removes only nonadditive pair terms. Undefined all-locality ratios are not replaced by zero.',
        '', '| Seed / head | Matched nonzero queries | Rule | ADE gain vs floor (%) | Conditional CI | Switch rate |',
        '|---|---:|---|---:|---|---:|']
    for key,v in r['seeds'].items():
        for arm,m in v['matched_nonzero'].items():
            table.append(f"| {key} | {v['queries']['matched_nonzero']} | {arm} | {gain(m['ADE_vs_floor'])} | {interval(m['ADE_vs_floor'])} | {number(m['switch_rate'])} |")
    table += ['', '## Solver and Learning Diagnostics','',
        '| Seed / head | Queries with edges | Matched queries | Nonzero matched | Solver failure counts |',
        '|---|---:|---:|---:|---|']
    for key,v in r['seeds'].items():
        q = v['queries']
        table.append(f"| {key} | {q['with_edges']} | {q['matched']} | {q['matched_nonzero']} | {q['solver_failures']} |")
    table += ['', 'Per-locality benefit/harm prediction errors, signed-gain correlations and underestimation rates are retained in `analysis.json`. Unknown outcome rows remain in each decision cohort.',
        'The overlap proxy is computed on forecasts in image coordinates, not measured physical collision risk. No calibrated safety, independent confirmation, deployment promotion, Stage5C or SMC claim.', '']
    (p/'results.md').write_text('\n'.join(table))
    losses = ['# Cost-Head Training Loss and Compute','',
        'Nine fixed ridge fits and nine fresh neural fits. Neural loss is continuous benefit/harm regression with underpredicted harm weighted fourfold.',
        '', '| Head | Method | Gradient steps | First loss | Mean last 5 losses | Elapsed seconds |',
        '|---|---|---:|---:|---:|---:|']
    for key,v in r['training'].items():
        f = v['fit']
        trace = f.get('trace',[])
        losses.append(f"| {key} | {'ridge closed form' if not trace else 'neural underharm4'} | {f.get('step',0)} | {number(trace[0]['loss']) if trace else 'not applicable'} | {number(sum(t['loss'] for t in trace[-5:])/len(trace[-5:])) if trace else 'not applicable'} | {v['wall_seconds']:.2f} |")
    losses += ['', 'Weights, checkpoints, arrays and row-level outputs remain private. Light provenance and metrics are public. No head was selected using held outcomes.', '']
    (p/'training_losses.md').write_text('\n'.join(losses))
    print(json.dumps(dict(study='intervention',heads=len(r['training']),joint_rows=r['joint_query_rows'])))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast',action='store_true')
    parser.add_argument('--intervention',action='store_true')
    args = parser.parse_args()
    if args.forecast:
        forecast()
    if args.intervention:
        intervention()
    if not (args.forecast or args.intervention):
        parser.error('Explicit completed study required')
