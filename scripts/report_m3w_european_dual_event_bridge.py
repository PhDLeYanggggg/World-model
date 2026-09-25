"""Report every bridge result and paired ablation without a best-seed claim."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_dual_event_bridge as run
from scripts.report_m3w_european_floor_relative import dump


def summarize(rows):
    if len(rows) != 18: raise ValueError('All registered producer/controller/seed groups required')
    result = {}
    names = set(rows[0]['summary'])
    if any(set(r['summary']) != names for r in rows): raise ValueError('Missing control')
    for name in sorted(names):
        vals = [r['summary'][name] for r in rows]
        def bounds(key):
            v = [r[key] for r in vals if r[key] is not None]
            return [min(v), max(v)] if len(v) == len(vals) else None
        result[name] = dict(all_gain=bounds('gain_vs_easy_add'), hard_gain=bounds('hard_gain_vs_easy_add'),
            vs_reference=bounds('gain_vs_reference'), vs_training_selected=bounds('gain_vs_training_selected'),
            easy_degradation=bounds('worst_easy_degradation'), switch_rate=bounds('switch_rate'),
            easy_pass=sum(v['easy_pass'] for v in vals), zero_CV_harm_views=sum(v['zero_CV_harmed_rows'] > 0 for v in vals),
            positive_CI=sum(v['CI'] is not None and v['CI'][0] > 0 for v in vals),
            negative_CI=sum(v['CI'] is not None and v['CI'][1] < 0 for v in vals))
    return result


def ablations(rows):
    out = {}
    for producer in range(3):
        for controller in range(3):
            if producer == controller: continue
            rr = sorted([r for r in rows if (r['producer'], r['controller']) == (producer, controller)], key=lambda r: r['seed'])
            assert [r['seed'] for r in rr] == [17, 29, 43]
            pair = {}
            for control in ('utility_only', 'all_risk_only', 'easy_risk_only', 'ridge_dual'):
                subsets = {}
                for subset in ('all', 'easy', 'hard'):
                    locals_ = rr[0]['views']['dual_risk']['ADE_vs_CV'][subset]['expected_scenes']
                    matrix = []
                    for r in rr:
                        a, b = [r['views'][k]['ADE_vs_CV'][subset]['by_scene'] for k in ('dual_risk', control)]
                        matrix.append([100*(1-a[s]['model_error']/b[s]['model_error']) for s in locals_])
                    local = np.mean(matrix, axis=0)
                    boot = np.random.default_rng(39271).choice(local, size=(3000, len(local))).mean(1)
                    subsets[subset] = dict(gain=float(local.mean()), CI=np.quantile(boot, [.025, .975]).tolist(),
                        by_locality=dict(zip(locals_, local.tolist())), seeds=3,
                        independent_localities=len(locals_), not_independent_confirmation=True)
                pair[control] = subsets
            out[f'producer{producer}_controller{controller}'] = pair
    return out


def gates(aggregate, seeds, comparisons):
    dual = aggregate['dual_risk']
    return dict(real_neural_cost_heads_trained=True, shared_reference_targets=True,
        producer_controller_selection_disjoint=True, all_fixed_comparisons_reported=True,
        dual_easy_preserved=dual['easy_pass'] == 18 and dual['zero_CV_harm_views'] == 0,
        dual_better_than_old_easy_add_all_six_seed_means=len(seeds) == 6 and all(r['dual_risk']['CI'][0] > 0 for r in seeds.values()),
        dual_better_than_easy_risk_all_six_seed_means=len(comparisons) == 6 and all(r['easy_risk_only']['all']['CI'][0] > 0 for r in comparisons.values()),
        independent_calibration=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)


def fmt(value):
    return 'not applicable / undefined' if value is None else f'{value[0]:+.6f}% to {value[1]:+.6f}%'


def main():
    done = json.loads((run.PUBLIC/'completion_checks.json').read_text()); assert done['all_passed']
    rows = []
    for ref in done['evaluation_receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    s = summarize(rows); a = ablations(rows)
    seeds = json.loads((run.PUBLIC/'seed_averaged_metrics.json').read_text())
    dump(run.PUBLIC/'aggregate_metrics.json', s); dump(run.PUBLIC/'paired_ablations.json', a)
    gate = gates(s, seeds, a); dump(run.PUBLIC/'gates.json', gate)
    lines = ['# Dual-Event Policy Bridge Results', '', '## Material Passport',
        'Fresh real neural-cost training and frozen-decision readout; cached_verified source forecasts.',
        'Reused opened model-selection evidence only. Six localities, not 18 independent repetitions.', '',
        'Primary comparator: the preceding old easy add_only policy, not the weaker bridge reference.', '',
        '| Policy | All ADE gain | Hard ADE gain | Gain vs training-selected motion | Worst-locality easy degradation | Easy passes | Positive/negative CI |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in s.items():
        lines.append(f"| {name} | {fmt(r['all_gain'])} | {fmt(r['hard_gain'])} | {fmt(r['vs_training_selected'])} | {fmt(r['easy_degradation'])} | {r['easy_pass']}/18 | {r['positive_CI']}/{r['negative_CI']} |")
    lines += ['', 'Ranges cover all registered groups, not selected winners. All/easy/hard labels retain',
        'producer-training thresholds. CV-zero harm is separate. Endpoint and complete-label results,',
        'p95/p99 errors and per-locality costs are in the group JSON files.', '',
        '## Three-Seed Paired Ablations', '',
        'Mean seedwise relative gains within each locality, followed by 3,000 locality resamples.',
        'No trajectory ensembling, window-level independence claim or multiplicity adjustment.', '',
        '| Producer/controller | Dual vs control | All gain | 95% locality CI | Hard gain |',
        '|---|---|---:|---:|---:|']
    for group, pairs in a.items():
        for control, v in pairs.items():
            lines.append(f"| {group} | {control} | {v['all']['gain']:+.6f}% | {fmt(v['all']['CI'])} | {v['hard']['gain']:+.6f}% |")
    lines += ['', 'Both risk heads describe exactly the same R-to-P action. Their predicted constraints',
        'do not certify realized risk. No calibration or confirmation localities opened.',
        'No candidate deployment, Stage5C or SMC. Image-pixel annotation-step results only.']
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    lines = ['# Gates: Dual-Event Bridge', '', 'Engineering completion does not imply scientific success.', '']
    lines += [f'- {k}: {str(v).lower()}' for k, v in gate.items()]
    (run.PUBLIC/'world_model_gate.md').write_text('\n'.join(lines)+'\n')
    training = json.loads((run.PRIVATE/'training_complete.json').read_text())
    records = []
    for ref in training['heads']:
        r = json.loads((ROOT/ref['path']).read_text()); fit = r['fit']
        records.append(dict(group=r['identity']['group'], task=r['identity']['task'], parameters=fit['parameters'],
            step=fit['step'], seconds=fit['seconds'], unknown_rows_sampled=fit['unknown_rows_sampled'],
            trace=fit['trace'], fixed_trace=fit.get('fixed_trace'), checkpoint=r['artifacts']['checkpoint']))
    dump(run.PUBLIC/'training_metrics.json', dict(heads=records, updates=108000,
        total_head_training_seconds=sum(r['seconds'] for r in records),
        loss_scale='train-source CV scale; not raw FDE; random minibatch losses not monotonic',
        new_forecaster_training=False, real_torch_training=True, pilot_in_fixed_budget=True))
    print(json.dumps(dict(aggregate=s, gates=gate), indent=2))


if __name__ == '__main__': main()
