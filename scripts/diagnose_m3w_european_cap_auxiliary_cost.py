"""Descriptive fit/held contrasts; never changes registered tests or selects arms."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COMPARISONS = {'aux_vs_control': ('cap_aux', 'cost_only'),
               'aux_vs_shuffled': ('cap_aux', 'shuffled_aux'),
               'aux_vs_original': ('cap_aux', 'original')}


def percent_gain(new, old):
    if new is None or old is None or not np.isfinite([new, old]).all() or old <= 0:
        return None
    return float(100*(old-new)/old)


def distribution(values):
    good = np.asarray([v for v in values if v is not None], float)
    return dict(dependent_views=len(values), missing=len(values)-len(good),
        positive=int((good > 0).sum()), negative=int((good < 0).sum()), equal=int((good == 0).sum()),
        median=float(np.median(good)) if len(good) else None,
        range=[float(good.min()), float(good.max())] if len(good) else None)


def diagnose(rows, fits):
    result = {}
    for pair in ('full', 'motion_only'):
        records = [(r, f) for r in rows if r['pair'] == pair for f in r['folds']]
        result[pair] = {}
        for comparison, (new, old) in COMPARISONS.items():
            train, held, worst = [], [], []
            for row, fold in records:
                def value(arm, fitting):
                    metrics = fold['training'][arm]['training'] if fitting else fold['metrics'][arm]
                    return metrics.get('envelope_positive', {}).get('harm_MSE')
                a = percent_gain(value(new, True), value(old, True))
                b = percent_gain(value(new, False), value(old, False))
                train.append(a); held.append(b)
                if b is not None:
                    worst.append(dict(group=row['group'], held=fold['held'], gain_percent=b))
            result[pair][comparison] = dict(fitting=distribution(train), held=distribution(held),
                fit_positive_held_negative=sum(a is not None and b is not None and a > 0 > b for a, b in zip(train, held)),
                fit_negative_held_negative=sum(a is not None and b is not None and a < 0 and b < 0 for a, b in zip(train, held)),
                worst_held=sorted(worst, key=lambda r: r['gain_percent'])[:3],
                missing_fit_reason='legacy_original_has_all_rows_only_not_same_subset' if old == 'original' else None)
    losses = {}
    for arm in ('cost_only', 'cap_aux', 'shuffled_aux'):
        traces = [r['fit']['trace'] for r in fits if r['arm'] == arm]
        losses[arm] = dict(models=len(traces),
            cost_loss_decreased=sum(t[-1]['cost_loss'] < t[0]['cost_loss'] for t in traces),
            first_cost_loss=distribution([t[0]['cost_loss'] for t in traces]),
            last_cost_loss=distribution([t[-1]['cost_loss'] for t in traces]))
    return dict(result_source='fresh_run_descriptive_diagnostic_from_frozen_aggregates',
        comparisons=result, fixed_batch_losses=losses, independent_samples=False,
        multiple_comparisons_adjusted=False, causal_mechanism_identified=False,
        gates_changed=False, model_or_threshold_selected=False)


def main():
    from scripts import run_m3w_european_cap_auxiliary_cost as run
    run.registration(); run.check_freeze()
    rows = json.loads((run.PUBLIC/'readout.json').read_text())['rows']
    fits = json.loads((run.PUBLIC/'training_metrics.json').read_text())
    assert len(rows) == 36 and len(fits) == 432
    result = diagnose(rows, fits)
    result['source_bindings'] = {name: run.artifact(run.PUBLIC/name)
                               for name in ('readout.json', 'training_metrics.json')}
    run.immutable_json(run.PUBLIC/'fit_held_diagnosis.json', result)
    lines = ['# Descriptive Fitting and Held Error Diagnosis', '',
        'Secondary diagnostic, not a change to the registered hypothesis or gates.',
        'Each entry summarizes72 dependent seed/locality/assignment views. Counts are not independent trials.', '',
        '| Inputs / contrast | Fitting median gain (%) | Held median gain (%) | Missing fitting comparisons | Fit positive / held negative | Both negative |',
        '|---|---:|---:|---:|---:|---:|']
    for pair, comparisons in result['comparisons'].items():
        for name, item in comparisons.items():
            lines.append(f"| {pair}/{name} | {item['fitting']['median']} | {item['held']['median']} | "
                         f"{item['fitting']['missing']} | {item['fit_positive_held_negative']} | {item['fit_negative_held_negative']} |")
    lines += ['', 'Fit/held divergence can be consistent with generalization failure but does not identify its cause.',
        'The original estimator differs in architecture, inputs and objective; only the three new arms are matched.',
        'Legacy original fitting diagnostics have all rows only. Its positive-envelope fitting comparison is not_estimable; denominators are not mixed.',
        'Fixed-batch loss is an optimization diagnostic, not whole-fitting MSE or held performance.',
        'Worst-view details and all supported/missing counts are retained in fit_held_diagnosis.json.',
        'No arm, loss weight, threshold or independent data role is selected by this diagnostic.',
        'No policy/trajectory gain, metric/seconds, true3D or foundation claim. Stage5C/SMC remain off.']
    (run.PUBLIC/'fit_held_diagnosis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['comparisons'], indent=2))


if __name__ == '__main__': main()
