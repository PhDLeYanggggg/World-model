"""Secondary descriptive counts; no threshold, model selection or gate changes."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_aux_gradient as run


def main():
    freeze = json.loads((run.PUBLIC/'diagnostic_freeze.json').read_text())
    primary = json.loads((run.PUBLIC/'aggregate_metrics.json').read_text())
    summaries = []
    for pair in ('full', 'motion_only'):
        for arm in ('cost_only', 'cap_aux', 'shuffled_aux'):
            for contrast in ('true_minus_cost', 'true_minus_shuffled', 'projection_minus_true', 'projected_true_minus_shuffled'):
                for metric in ('cost4', 'easy_harm_positive'):
                    selected = [r for r in primary['comparisons'] if (r['pair'], r['arm'], r['contrast'], r['metric']) ==
                                (pair, arm, contrast, metric)]
                    known = [r for r in selected if r['status'] != 'not_estimable']
                    pos = sum(r['CI'][0] > 0 for r in known); neg = sum(r['CI'][1] < 0 for r in known)
                    summaries.append(dict(pair=pair, arm=arm, contrast=contrast, metric=metric,
                        positive=pos, negative=neg, overlap=len(known)-pos-neg, unsupported=len(selected)-len(known),
                        point_range=[min(r['point'] for r in known), max(r['point'] for r in known)] if known else None))
    cells = defaultdict(list)
    for ref in freeze['receipts']:
        assert run.artifact(ROOT/ref['path']) == ref
        row = json.loads((ROOT/ref['path']).read_text())
        for state in row['states']:
            cells[(row['pair'], state['arm'])].extend(state['repeats'])
    diagnostic = []
    for (pair, arm), rows in sorted(cells.items()):
        local = dict(pair=pair, arm=arm, dependent_batches=len(rows))
        for metric in ('cost4', 'easy_harm_positive'):
            g = [r['gradients']['true'][metric+'_shared'] for r in rows]
            valid = [v for v in g if v['cosine'] is not None]
            conflict = [r for r in rows if r['gradients']['true']['cost4_shared']['conflict'] is True]
            eligible = [r for r in conflict if r['after']['true_aux']['joined'][metric] is not None]
            improved = sum(r['after']['projected_true']['joined'][metric] < r['after']['true_aux']['joined'][metric] for r in eligible)
            worsened = sum(r['after']['projected_true']['joined'][metric] > r['after']['true_aux']['joined'][metric] for r in eligible)
            local[metric] = dict(shared_gradient_estimable=len(valid), shared_gradient_conflicts=sum(v['conflict'] for v in valid),
                cosine_median=float(np.median([v['cosine'] for v in valid])) if valid else None,
                norm_ratio_median=float(np.median([v['norm_ratio'] for v in valid])) if valid else None,
                cost4_conflict_batches=len(conflict), actual_projection_improved=improved,
                actual_projection_worsened=worsened, actual_projection_equal=len(eligible)-improved-worsened)
        local['clipped_updates'] = {name: sum(r['after'][name]['gradient_norm_before_clip'] > 5 for r in rows)
                                    for name in run.method.VARIANTS}
        diagnostic.append(local)
    result = dict(result_source='fresh_secondary_descriptive_analysis_of_frozen_diagnostics',
        primary_contrast_summary=summaries, gradient_adamw_counts=diagnostic,
        gate_changes=False, independent_sample_counts=False, additional_training=False,
        explanation='Counts include repeated batches/models and overlapping contexts; not independent tests or causal training attribution.')
    run.immutable_json(run.PUBLIC/'secondary_diagnosis.json', result)
    print(json.dumps([r for r in diagnostic if r['pair'] == 'full' and r['arm'] == 'cap_aux'], indent=2))


if __name__ == '__main__': main()
