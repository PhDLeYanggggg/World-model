"""Post-fit summaries of fixed forecast readouts, not policy certification."""
from collections import defaultdict
import math


def recording_budget_counts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['arm'], row['subset'], row['reference']].append(row)
    result = {}
    for key, group in sorted(groups.items()):
        passed = failed = undefined = net_fail = unknown = 0
        for row in group:
            ratio = row['known_positive_harm_percent']
            flag = row['known_budget_pass']
            if ratio == '':
                if flag != '' or row['support_status'] != 'not_estimable':
                    raise ValueError('Undefined risk cannot be a safety pass')
                undefined += 1
            else:
                value = float(ratio)
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Finite nonnegative known risk required')
                expected = value <= 2.
                if flag != str(expected):
                    raise ValueError('Budget flag disagrees with the frozen 2% rule')
                passed += expected
                failed += not expected
            net = row['known_net_degradation_percent']
            net_fail += net != '' and float(net) > 2.
            unknown += int(row['unknown_rows']) > 0
            if row['complete_population_budget_claim'] != 'False':
                raise ValueError('Partial-label diagnostic cannot certify complete risk')
        if passed + failed + undefined != len(group):
            raise ValueError('All required views must be retained')
        result['/'.join(key)] = dict(required_views=len(group), known_budget_passes=passed,
            known_budget_failures=failed, not_estimable=undefined,
            known_net_degradation_gt2=net_fail, views_with_unknown_labels=unknown,
            complete_policy_safety_certified=False)
    return result


def paired_fit_losses(freeze):
    """Compare only the shared ADE component on identical scheduled minibatches."""
    rows = []
    for key, trial in sorted(freeze['fits'].items()):
        new = trial['fit']
        old = freeze['identity']['controls'][key]['fit']
        for field in ('step', 'total_draws', 'unique_training_rows', 'held_rows_sampled'):
            if new[field] != old[field]:
                raise ValueError('Matched fitting budget differs')
        if new['step'] != 4000 or new['held_rows_sampled'] != 0:
            raise ValueError('Complete train-only endpoint required')
        newlog, oldlog = new['losses'], old['losses']
        if [r['step'] for r in newlog] != [r['step'] for r in oldlog]:
            raise ValueError('Matched loss logging steps differ')
        # Last ten logged minibatches are a diagnostic, not a whole-TRAIN loss.
        newtail, oldtail = newlog[-10:], oldlog[-10:]
        mean = lambda values: sum(values) / len(values)
        rows.append(dict(trial=key, final_step=4000, tail_logged_batches=len(newtail),
            new_native_ADE=mean([r['native_loss'] for r in newtail]),
            old_native_ADE=mean([r['loss'] for r in oldtail]),
            new_selected_positive_harm=mean([r['selected_positive_harm'] for r in newtail]),
            new_CV_positive_harm=mean([r['cv_positive_harm'] for r in newtail]),
            same_draw_counts=True, whole_training_loss_claim=False))
    return rows
