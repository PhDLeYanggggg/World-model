"""Descriptive readout after the registered paired-quality experiment completes."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/european_past_quality_auxiliary_v1'


def mean(values):
    return math.fsum(values) / len(values) if values else None


def describe(rows):
    out = {'groups': len(rows), 'arms': {}, 'transitions': {}}
    for arm in ('original', 'quality', 'placebo'):
        policies = [r['result']['policies'][arm] for r in rows]
        risks = [p['easy_selected_risk_upper'] for p in policies]
        out['arms'][arm] = {
            'signed_score_MSE_mean_across_heads': mean([r['result']['scores'][arm] for r in rows]),
            'selected_occurrences': sum(p['selected_count'] for p in policies),
            'unknown_selected_occurrences': sum(p['selected_unknown'] for p in policies),
            'complete_support': sum(p['finite_completion_supported'] for p in policies),
            'undefined_easy_risk': sum(v is None for v in risks),
            'easy_risk_upper_violations': sum(v is not None and v > .02 + 1e-12 for v in risks),
            'worst_easy_risk_upper': max((v for v in risks if v is not None), default=None),
        }
        if arm != 'original':
            fits = [r['training'][arm] for r in rows]
            out['arms'][arm].update(
                mean_leaf_training_loss_before=mean([f['mean_leaf_training_loss_before'] for f in fits]),
                mean_leaf_training_loss_after=mean([f['mean_leaf_training_loss_after'] for f in fits]),
                zero_mean_error_max=max(f['training_zero_mean_max'] for f in fits),
                negative_raw_coordinates=sum(r['diagnostic'][arm]['negative_raw_coordinates'] for r in rows),
                validation_MSE_better_than_original_groups=sum(
                    r['result']['scores'][arm] < r['result']['scores']['original'] for r in rows),
            )
            lost, gained, new_violations, resolved, undefined = 0, 0, 0, 0, 0
            for r in rows:
                old, new = (r['result']['policies'][k] for k in ('original', arm))
                lost += old['finite_completion_supported'] and not new['finite_completion_supported']
                gained += not old['finite_completion_supported'] and new['finite_completion_supported']
                a, b = old['easy_selected_risk_upper'], new['easy_selected_risk_upper']
                old_bad = a is not None and a > .02 + 1e-12
                new_bad = b is not None and b > .02 + 1e-12
                new_violations += new_bad and not old_bad
                resolved += old_bad and b is not None and not new_bad
                undefined += old_bad and b is None
            out['transitions'][arm] = dict(
                complete_support_lost=lost, complete_support_gained=gained,
                new_upper_violations=new_violations,
                old_upper_violations_now_defined_below_budget=resolved,
                old_upper_violations_now_undefined_not_passes=undefined)
    return out


def risk_decomposition(rows):
    out = {}
    for arm in rows[0]['result']['policies']:
        cases = []
        for r in rows:
            p = r['result']['policies'][arm]
            ref = p['selected_known_easy_reference_mass']
            known = p['selected_known_easy_harm_mass'] / ref if ref > 0 else None
            upper = p['easy_selected_risk_upper']
            cases.append(dict(group=r['group'], head_seed=r['head_seed'],
                              known_easy_harm_mass=p['selected_known_easy_harm_mass'],
                              selected_easy_reference_mass=ref,
                              unknown_envelope_mass=p['selected_unknown_envelope_mass'],
                              known_easy_risk=known, completion_easy_risk_upper=upper,
                              selected_occurrences=p['selected_count'],
                              selected_unknown=p['selected_unknown'],
                              whole_easy_degradation_upper=p['easy_degradation_upper']))
        upper_bad = [c for c in cases if c['completion_easy_risk_upper'] is not None
                     and c['completion_easy_risk_upper'] > .02 + 1e-12]
        known_bad = [c for c in cases if c['known_easy_risk'] is not None
                     and c['known_easy_risk'] > .02 + 1e-12]
        out[arm] = dict(
            known_label_violations=len(known_bad), upper_violations=len(upper_bad),
            completion_only_violations=len(upper_bad) - len(known_bad),
            worst_upper_case=max(cases, key=lambda c: c['completion_easy_risk_upper']
                                if c['completion_easy_risk_upper'] is not None else -1),
            violating_cases=upper_bad)
    return out


def persist(path, value):
    payload = json.dumps(value, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == payload
    else:
        with path.open('x') as f:
            f.write(payload)


def main():
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    verification = json.loads((PUBLIC / 'verification.json').read_text())
    assert sha(PUBLIC / 'complete.json') == verification['complete_sha256']
    assert sha(PUBLIC / 'summary.json') == verification['summary_sha256']
    complete = json.loads((PUBLIC / 'complete.json').read_text())
    rows = []
    for ref in complete['groups']:
        path = ROOT / ref['path']
        assert sha(path) == ref['sha256']
        rows.append(json.loads(path.read_text()))
    assert len(rows) == 72
    result = dict(
        status='fresh_run_descriptive_posthoc_no_model_selection',
        checkpoint_training='fresh_run_preceding_registered_experiment',
        inputs='cached_verified_completed_group_reports',
        verification_sha256=sha(PUBLIC / 'verification.json'),
        independent_confirmation=False, model_or_threshold_changed=False,
        group_counts_are_repeated_heads_not_independent_scenes=True,
        training_loss_is_per_tree_weighted_five_moment_MSE_not_trajectory_loss=True,
        all_groups=describe(rows),
        by_locality={s: describe([r for r in rows if r['source'] == s])
                     for s in sorted({r['source'] for r in rows})},
        by_head_seed={str(s): describe([r for r in rows if r['head_seed'] == s])
                      for s in sorted({r['head_seed'] for r in rows})},
    )
    persist(PUBLIC / 'diagnostic_readout.json', result)
    persist(PUBLIC / 'risk_decomposition.json', dict(
        status='fresh_run_descriptive_posthoc_no_model_selection',
        verification_sha256=sha(PUBLIC / 'verification.json'),
        unknown_outcomes_are_not_observed_harm=True,
        risk_budget_unchanged=.02, policies=risk_decomposition(rows)))
    print(json.dumps(result['all_groups'], indent=2))


if __name__ == '__main__':
    main()
