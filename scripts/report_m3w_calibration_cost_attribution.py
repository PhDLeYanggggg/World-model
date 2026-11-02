"""Post-readout selected-cost accounting; does not fit or change any policy."""
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 Python required before importing model modules')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from src.world_model.m3w_european_source_forecast import baseline_numpy
from src.world_model.m3w_independent_risk_calibration import load_record
from src.world_model.m3w_selection_controller import labels as cost_labels
from src.evaluation.m3w_selected_cost_diagnosis import subset_summary, MOMENTS

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'
RATIOS = ('harm_prediction_ratio', 'easy_harm_prediction_ratio',
          'reference_inflation_ratio', 'easy_reference_inflation_ratio')


def checked(ref):
    path = ROOT/ref['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('Frozen diagnostic source changed')
    return path


def reconcile(diagnosis, view):
    for field, original in (('observed_all_risk_on_known', 'known_all_risk'),
                            ('observed_easy_risk_on_known', 'known_easy_risk')):
        a, b = diagnosis[field], view[original]
        if a is None or b is None:
            if a is not b:
                raise ValueError('Undefined reference cannot be replaced by a numeric risk')
        elif not np.isclose(a, b, rtol=1e-10, atol=1e-12):
            raise ValueError('Selected cost mass must reconcile to the original metric')


def main():
    started = time.monotonic()
    read = lambda p: json.loads(Path(p).read_text())
    complete = read(PUBLIC/'eval.json'); registration = read(PUBLIC/'registration.json')
    roles = read(checked(registration['roles'])); label_refs = read(PUBLIC/'labels.json')['label_receipts']
    if len(complete['private_metric_receipts']) != 918 or len(label_refs) != 102:
        raise ValueError('Complete original readout required before descriptive attribution')
    label_map = {str(Path(r['path']).parent):r for r in label_refs}
    producers = {r['trial']:r for r in registration['frozen_models']}
    regsha = hashlib.sha256((PUBLIC/'registration.json').read_bytes()).hexdigest()
    buckets, seen = {}, set()
    for ref in complete['private_metric_receipts']:
        metric_path = checked(ref); metric = read(metric_path); directory = metric_path.parent
        record = directory.parent; key = (metric['source_member'], metric['trial'])
        if key in seen:
            raise ValueError('Repeated recording/pipeline receipt')
        seen.add(key); producer = producers[metric['trial']]
        _, data, _ = load_record(record, roles, metric['source_member'], regsha)
        lr = read(checked(label_map[str(record.relative_to(ROOT))]))
        target = np.load(checked(lr['target']), allow_pickle=False)
        valid = np.load(checked(lr['valid']), allow_pickle=False)
        policy = read(checked(metric['policy'])); inf = read(checked(policy['inference']))
        choices = np.load(checked(policy['choices']), mmap_mode='r', allow_pickle=False)
        neural = np.load(checked(inf['neural']), mmap_mode='r', allow_pickle=False)
        moments = np.load(checked(inf['moments']), mmap_mode='r', allow_pickle=False)
        physical = np.stack([baseline_numpy(data['history'], k) for k in range(6)], 1)
        paths = np.concatenate((physical, neural[:, None]), 1)
        truth, _ = cost_labels(paths, target, valid, producer['baseline_index'], producer['easy_cut'])
        floor = paths[:, producer['baseline_index']]; ix = np.arange(len(paths))
        scores = {(r['multiplier'], r['arm']):r['metrics'] for r in metric['scores']}
        for j, multiplier in enumerate(policy['multipliers']):
            for arm in ('independent', 'joint_matched'):
                c = choices[j, policy['policy_names'].index(arm)]
                action = np.where(c < 0, producer['baseline_index'], c)
                changed = np.linalg.norm(paths[ix, action]-floor, axis=-1).max(1) > 0
                # The known-label mask is for diagnosis only, never for allocation.
                p = moments[ix[changed], action[changed], None, :]
                y = truth[ix[changed], action[changed], None, :]
                result = subset_summary(p, y, np.ones(p.shape[:2], bool), np.ones(len(p)))
                original = next(iter(scores[multiplier, arm]['recordings'].values()))
                reconcile(result, original)
                b = buckets.setdefault((multiplier, arm), dict(views=0, known=0, unknown=0,
                    predicted={k:0. for k in MOMENTS}, observed={k:0. for k in MOMENTS},
                    ratios={k:[] for k in RATIOS}, violations={k:0 for k in ('all', 'easy')},
                    harm_dominates={k:0 for k in ('all', 'easy')}))
                b['views'] += 1; b['known'] += result['known_action_views']; b['unknown'] += result['unknown_action_views']
                for k in MOMENTS:
                    b['predicted'][k] += result['predicted_known_mass'][k]
                    b['observed'][k] += result['observed_known_mass'][k]
                for k in RATIOS:
                    if result[k] is not None:
                        b['ratios'][k].append(result[k])
                for event in ('all', 'easy'):
                    value = result['observed_'+event+'_risk_on_known']
                    if value is not None and value > .02:
                        d = result[event+'_excess_decomposition']
                        b['violations'][event] += 1
                        b['harm_dominates'][event] += int(d['harm_underprediction'] >= d['reference_inflation'])
    if len(seen) != 918 or len(buckets) != 26 or any(b['views'] != 918 for b in buckets.values()):
        raise ValueError('Every recording/pipeline and both required policies/grid points required')
    rows = []
    for (multiplier, arm), b in sorted(buckets.items()):
        ratios = {k:dict(median=statistics.median(v) if v else None, defined_views=len(v))
                  for k, v in b.pop('ratios').items()}
        rows.append(dict(multiplier=multiplier, arm=arm, **b, record_ratio_summaries=ratios))
    out = dict(result_source='fresh_run_post_readout_descriptive_selected_cost_attribution',
        scope='known_label_costs_at_frozen_actual_switches_two_required_policies_full_grid',
        rows=rows, reconciled_original_metric_views=23868, reconciled_known_risk_ratios=47736,
        repeated_pipeline_record_views=918,
        predicted_costs_are_native_not_lambda_coefficient_view=True,
        MSE_scale='one_pixel_no_new_normalizer_fit',
        unknown_rows_retained_in_counts_not_imputed_cost_labels=True,
        medians_conditional_on_defined_record_ratios=True, causal_training_cause_proven=False,
        new_optimizer_updates=0, threshold_or_policy_change=False, confirmation_access=False,
        empirical_calibration_not_independent_confirmation=True, deployment_changed=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        eval_receipt_sha256=hashlib.sha256((PUBLIC/'eval.json').read_bytes()).hexdigest(),
        labels_receipt_sha256=hashlib.sha256((PUBLIC/'labels.json').read_bytes()).hexdigest(),
        diagnostic_helper_sha256=hashlib.sha256((ROOT/'src/evaluation/m3w_selected_cost_diagnosis.py').read_bytes()).hexdigest(),
        execution=dict(pid=os.getpid(), seconds=time.monotonic()-started,
                       peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    (PUBLIC/'selected_cost_attribution.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')
    lines = ['# Post-Readout Selected Cost Attribution', '',
        'This diagnosis was added after the completed calibration readout. It does not',
        'choose a multiplier, fit a model, alter decisions or open confirmation.',
        'Native predicted cost moments are compared on exactly the same known-label',
        'actual-switch rows as the original risk metric. The lambda-scaled optimizer',
        'coefficient view is not treated as a predicted cost distribution.', '',
        '|Lambda|Policy|Median predicted/true H|Median predicted/true EH|Median predicted/true R|Median predicted/true ER|All violations (harm-dominant)|Easy violations (harm-dominant)|',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'N/A' if v is None else f'{v:.6g}'
    for r in rows:
        values = '|'.join(fmt(r['record_ratio_summaries'][k]['median']) for k in RATIOS)
        lines.append(f'|{r["multiplier"]}|{r["arm"]}|{values}|'
            + '|'.join(f'{r["violations"][k]} ({r["harm_dominates"][k]})' for k in ('all', 'easy'))+'|')
    lines += ['', 'The exact identity is observed excess minus predicted excess =',
        'harm underprediction + 0.02 times reference overprediction, separately for',
        'H/R and EH/ER. Negative components alleviate the excess. A dominant component',
        'is an accounting diagnosis, not proof of the causal training failure.',
        'Medians exclude undefined ratios and are explicitly conditional; the JSON',
        'retains finite counts, unknown selected rows and native cost masses. Views',
        'repeat nine pipelines and are not independent observations. Missing outcomes',
        'are not imputed. Available-label cost is not full12 ADE. No physical safety,',
        'metric/seconds, true3D, foundation, Stage5C or SMC claim is made.', '']
    (PUBLIC/'selected_cost_attribution.md').write_text('\n'.join(lines))
    print(json.dumps(dict(record_views=918, reconciled_original_metric_views=23868,
                          reconciled_known_risk_ratios=47736, policies_changed=False)))


if __name__ == '__main__':
    main()
