"""Check which Athens conclusions survive weighting the four observed dates."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
SOURCE = PUBLIC/'pneuma_confirmation_v1'
OUT = PUBLIC/'pneuma_date_robustness_v1/readout.json'
REGISTRATION_COMMIT = '84b9e944e'
FILES = ('registration', 'freeze', 'confirmation', 'stepwise_budget_sensitivity',
         'recipient_path_attribution')


def envelope(metrics, counts):
    fields = ('lower_pp', 'observed_pp', 'upper_pp')
    dates = sorted(counts)
    for field in fields:
        m = metrics[field]
        if (m['member_count_by_date'] != dict(counts) or set(m['by_date']) != set(dates)
                or m['status'] != 'descriptive_date_bootstrap'
                or m['bootstrap_unit'] != 'acquisition_date_after_seed_then_member_average'
                or not m['one_city']
                or not all(isfinite(x) for x in m['by_date'].values())
                or not isclose(mean(m['by_date'].values()), m['point'], abs_tol=1e-12)):
            raise ValueError(f'Mismatched acquisition-date aggregation: {field}')
    by_date = {d: [metrics[f]['by_date'][d] for f in fields] for d in dates}
    if any(not a <= b <= c for a, b, c in by_date.values()):
        raise ValueError('Invalid sample envelope')
    leave_one_out = {d: [mean(by_date[k][i] for k in dates if d != k) for i in range(3)] for d in dates}
    equal = [mean(v[i] for v in by_date.values()) for i in range(3)]
    # A nonnegative weighted average cannot leave the convex hull of its date endpoints.
    return dict(by_date_pp=by_date, equal_date_pp=equal, leave_one_date_out_pp=leave_one_out,
                any_date_mixture_envelope_pp=[min(v[0] for v in by_date.values()),
                                              max(v[2] for v in by_date.values())],
                dates_definitely_positive=sum(v[0] > 0 for v in by_date.values()),
                dates_definitely_negative=sum(v[2] < 0 for v in by_date.values()),
                dates_including_zero=sum(v[0] <= 0 <= v[2] for v in by_date.values()),
                all_leave_one_out_positive=all(v[0] > 0 for v in leave_one_out.values()),
                all_leave_one_out_include_zero=all(v[0] <= 0 <= v[2] for v in leave_one_out.values()))


def replay(raw):
    h = {k: sha256(v).hexdigest() for k, v in raw.items()}
    registration, freeze, primary, stepwise, attribution = (json.loads(raw[k]) for k in FILES)
    spec = registration['protocol']
    dates = registration['dates']
    counts = Counter(dates.values())
    if (len(counts) != 4 or len(dates) != 6 or dates != primary['dates']
            or set(primary['members']) != set(dates)
            or registration['member_roles'] != dict.fromkeys(dates, 'confirmation')
            or spec['date_roles'] != dict.fromkeys(counts, 'confirmation')
            or (spec['observed_steps'], spec['prediction_steps']) != (8, 12)
            or spec['mode'] != 'zero_shot' or spec['risk_budget'] != .02
            or freeze['registration'] != h['registration'] or freeze['thresholds'] is not None
            or freeze['label_access'] or not primary['roles_frozen_before_read']
            or primary['binding']['registration'] != h['registration']
            or primary['binding']['freeze'] != h['freeze']
            or primary['binding']['role'] != 'confirmation' or primary['mode'] != 'zero_shot'
            or primary['official_risk_budget'] != .02 or primary['new_fit'] or primary['deployment_changed']
            or primary['release'] != 'not_run_zero_shot_fixed_count_only'):
        raise ValueError('Expected the approved zero-shot readout with no calibrated release')
    if (stepwise['primary_confirmation'] != h['confirmation']
            or attribution['primary_confirmation'] != h['confirmation']
            or stepwise['attribution_receipt'] != h['recipient_path_attribution']
            or stepwise['source_date_count'] != len(counts)
            or attribution['source_date_count'] != len(counts)
            or not stepwise['primary_unchanged'] or stepwise['release_recomputed']
            or stepwise['thresholds_changed'] or not attribution['no_threshold_changes']):
        raise ValueError('Secondary analyses must use the same frozen primary readout')

    arms = primary['matched']['by_arm']
    contrasts = {f'{a}_vs_cv': arms[a]['vs_reference']['cv']
                 for a in ('joint', 'same_family', 'independent', 'additive')}
    contrasts.update({f'joint_vs_{a}': metric for a, metric in primary['matched']['joint_minus'].items()})
    contrasts['joint_vs_independent_equal_step_movement'] = stepwise['by_comparator']['independent']['matched_steps']
    contrasts['joint_vs_independent_path_term'] = attribution['by_comparator']['independent']['path']
    return dict(status='fresh_aggregate_diagnostic', targets_opened_this_run=False,
                new_fit=False, release_recomputed=False, deployment_changed=False,
                registration_commit=REGISTRATION_COMMIT, input_sha256=h,
                date_counts=dict(counts), columns=['lower', 'observed', 'upper'],
                joint_risk_on_fixed_proposals=arms['joint']['risk'],
                contrasts={k: envelope(v, counts) for k, v in contrasts.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    raw = {k: (SOURCE/f'{k}.json').read_bytes() for k in FILES}
    for k in ('registration', 'freeze'):
        p = (SOURCE/f'{k}.json').relative_to(ROOT)
        registered = subprocess.check_output(['git', 'show', f'{REGISTRATION_COMMIT}:{p}'], cwd=ROOT)
        if registered != raw[k]:
            raise ValueError(f'Changed published {k}')
    encoded = (json.dumps(replay(raw), indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_bytes() != encoded:
            raise FileExistsError(f'Refusing to replace {args.output}')
    else:
        with args.output.open('xb') as f:
            f.write(encoded)
    print(args.output)


if __name__ == '__main__':
    main()
