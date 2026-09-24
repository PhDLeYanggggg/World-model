"""Post-decision sensitivity bounds; never substitutes an oracle policy."""
import json
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_m3w_native_forecast import file_digest, immutable_json
import numpy as np


def main():
    public = ROOT/'outputs/publication_readiness_2026_09/net_easy_moment_guarded_v1'
    root = ROOT/'data/stage_cvpr2027_experiments/net_easy_moment_guarded_v1'
    a = json.loads((public/'analysis.json').read_text())
    parent_path = ROOT/'outputs/publication_readiness_2026_09/easy_moment_v1/analysis.json'
    context_path = ROOT/'outputs/publication_readiness_2026_09/native_scene_context_v2/analysis.json'
    parent, context = json.loads(parent_path.read_text()), json.loads(context_path.read_text())
    identity = json.loads((root/'identity.json').read_text())
    for path in (parent_path, context_path):
        assert file_digest(path) == identity['source_bindings'][str(path.relative_to(ROOT))]
    dm = json.loads((root/'decisions_complete.json').read_text())
    failed = []
    for ref in dm['receipts']:
        assert file_digest(ROOT/ref['path']) == ref['sha256']
        r = json.loads((ROOT/ref['path']).read_text())
        for q in r['queries']:
            for name in ('positive', 'net', 'matched'):
                if q[name]['optimal']:
                    continue
                site, seed = r['view'].rsplit('_seed', 1)
                cr = next(v for v in context['records'] if v['recording'] == q['recording'])
                assert file_digest(ROOT/cr['cache']['path']) == cr['cache']['sha256']
                with np.load(ROOT/cr['cache']['path'], allow_pickle=False) as z:
                    rows = z['context_target_rows'][z['context_frame_ids'] == q['frame']]
                    rows = rows[rows >= 0]
                assert file_digest(ROOT/r['path']) == r['sha256']
                with np.load(ROOT/r['path'], allow_pickle=False) as z:
                    loc = np.flatnonzero(np.isin(z['ids'], rows))
                    np.testing.assert_array_equal(np.sort(z['ids'][loc]), np.sort(rows))
                    active = z['ids'][loc][z['support'][loc]]
                    wanted_count = int(z['choices'][loc, 6].sum()) if name == 'matched' else None
                source = next(v for v in parent['outcome_archives'] if v['path'].endswith(f"{r['action']}_seed{seed}.npz"))
                assert file_digest(ROOT/source['path']) == source['sha256']
                with np.load(ROOT/source['path'], allow_pickle=False) as z:
                    cv, candidate = z['cv'], z['candidate_ade']
                    bounds = {}
                    policy = dict(positive='positive_population', net='net_population', matched='net_matched_positive')[name]
                    for subset in ('all', 'hard', 'positive_easy'):
                        mask = np.ones(len(cv), bool) if subset == 'all' else z[subset]
                        used = active[mask[active] & np.isfinite(cv[active])]
                        absolute = math.fsum(np.abs(candidate[used]-cv[used]))
                        sm = a['summary'][r['action']+'__'+policy]['seeds'][seed]
                        metric = sm['ADE'] if subset == 'all' else sm['subsets'][subset]
                        row = metric['by_scene'][site]
                        denominator = row['rows'] * row['reference_error']
                        bounds[subset] = dict(max_absolute_native_error_sum=absolute,
                            equal_site_seed_mean_gain_shift_bound_pp=100*absolute/(4*3*denominator) if denominator else None,
                            single_site_seed_gain_shift_bound_pp=100*absolute/denominator if denominator else None)
                failed.append(dict(view=r['view'], action=r['action'], recording=q['recording'], frame=q['frame'],
                    policy=name, solver=q[name], target_count=len(rows), eligible_count=len(active),
                    intended_count=wanted_count, bounds=bounds,
                    downstream_count_control_affected=name == 'positive'))
    result = dict(result_source='fresh_run_postdecision_sensitivity_no_policy_change',
        analysis_sha256=file_digest(public/'analysis.json'),
        source_bindings={'scripts/audit_m3w_net_risk_numerics.py': file_digest(Path(__file__))},
        failed_query_arm_instances=len(failed), records=failed,
        interpretation='Upper bound on possible metric shift if these query decisions changed; not a repaired optimum or proposed policy.',
        no_forecast_retraining=True, no_threshold_search=True, deployment=False)
    immutable_json(public/'numerical_sensitivity.json', result)
    lines = ['# Numerical Fallback Sensitivity', '',
        'Post-decision bounds only. No action is chosen from observed costs, and no original decision is changed.',
        'For the affected query, sum absolute candidate-minus-CV error on eligible observed rows. Divide by the',
        'site CV sum and by four sites and three seeds to bound any primary mean-gain change.',
        'This deliberately overbounds an exact-count repair and retains all existing outcomes.', '',
        '| View / predictor / policy | Eligible | Desired count | Max mean ADE gain shift pp |', '|---|---:|---:|---:|']
    for r in failed:
        lines.append(f"| {r['view']} / {r['action']} / {r['policy']} | {r['eligible_count']} | {r['intended_count']} | {r['bounds']['all']['equal_site_seed_mean_gain_shift_bound_pp']:.9f} |")
    lines += ['', 'The positive-policy failure also sets its matched-control reference count to zero at that query.',
        'The matched Transformer arm is not globally exact-count matched because its own failed query falls to zero.',
        'These limitations must accompany matched comparisons. Bounds do not certify risk or repair the solver.']
    (public/'numerical_sensitivity.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
