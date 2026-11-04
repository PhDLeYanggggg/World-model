"""Verify a reference-support ceiling from existing complete scalar evidence."""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_shared_motion_cost as source
from scripts.report_m3w_moving_harm_refinement import text_once

PARENT = ROOT / 'outputs/publication_readiness_2026_09/moving_record_feasibility_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/moving_support_ceiling_v1'


def support_ceiling(rows):
    if not rows or len({(r['job'], r['recording']) for r in rows}) != len(rows):
        raise ValueError('Complete nonempty uniquely identified recording views required')
    empty = []
    for row in rows:
        mass, count = row['full_known_easy_reference_mass'], row['positive_easy_reference_rows']
        if not math.isfinite(mass) or mass < 0 or count < 0 or (mass == 0) != (count == 0):
            raise ValueError('Nonnegative finite full-population reference and event counts must agree')
        if mass == 0:
            empty.append(dict(job=row['job'], recording=row['recording'],
                known_easy_rows=row['known_easy_rows'], unknown_rows=row['unknown_rows'],
                full_known_easy_reference_mass=mass))
    return dict(required_record_views=len(rows), zero_full_easy_reference_views=len(empty),
        absolute_upper_bound_on_supported_views=len(rows)-len(empty),
        upper_bound_not_achievable_support_estimate=True, zero_mass_views=empty,
        whole_roster_positive_reference_support_possible=not empty,
        dependence='unchanged_floor_CV_easy_event_observed_labels_and_positive_reference_contract',
        unavailable_full_risk_not_reclassified_as_pass=True)


def main():
    path = PARENT / 'analysis.json'
    parent, verification = source.read(path), source.read(PARENT / 'verify_execution.json')
    if not verification['all_scalars_exact'] or not parent['exact_witness_gate_agreement_all_views']:
        raise ValueError('Original complete scalar replay required')
    for filename, expected in parent['bindings'].items():
        if source.sha(ROOT / filename) != expected:
            raise ValueError('Original feasibility schema/code binding changed')
    source.checked(parent['parent_summary'])
    source.checked(parent['parent_verification'])
    result = support_ceiling(parent['rows'])
    if result['required_record_views'] != 348 or len({r['job'] for r in parent['rows']}) != 18:
        raise ValueError('Do not use a filtered recording or model population')
    PUBLIC.mkdir(exist_ok=True)
    source.once(PUBLIC / 'analysis.json', source.stable(dict(**result,
        input=source.artifact(path), original_replay=source.artifact(PARENT / 'verify_execution.json'),
        result_source='cached_verified_parent_scalars_fresh_arithmetic_not_fresh_raw_inference',
        new_model_inference=False, new_training=False, new_decisions=False, population_changed=False,
        risk_contract_changed=False, independent_confirmation=False, new_deployment=False,
        original24group_primary='not_estimable_not_passed', stage5c_executed=False, smc_enabled=False)))
    n, missing = result['required_record_views'], result['zero_full_easy_reference_views']
    text_once(PUBLIC / 'readout.md', f'''# Separate Structural Support From Prediction Error

All {n} original repeated recording views are retained. {missing} have exactly
zero full observed easy-reference mass; even before considering a proposal,
there is no positive easy reference anywhere in that recording view. Their
maximum selected easy-reference mass is therefore zero for every subset.
At most {n-missing}/{n} views can have positive easy-reference support under
these unchanged data/event/floor definitions. This is an upper bound, not a
claim that {n-missing} are achievable or that all other views are safe.

This conclusion uses the verified old complete scalar evidence and fresh
arithmetic, not new raw-data inference. It is separate from the earlier
330/348 single-row candidate witness diagnostic. It does not infer general
joint impossibility from failure to find a witness: the zero-mass ceiling
follows directly from nonnegative selected mass being at most full mass.

For a recording v, let r_i be its nonnegative observed easy-reference cost.
For any selection S, sum(i in S) r_i <= sum(i in v) r_i = 0. Unknown rows add
no observed/lower reference credit under the existing evaluator. A changed
cost head, proposal, joint solver or longer training cannot make this fixed
denominator positive. This is about the released-label benchmark, not proof
that latent physical easy events or unobserved outcomes do not exist.

No failing view is dropped and no unavailable risk becomes a pass. The
original full-support gate remains failed/unavailable, including for models
whose utility or positive-harm estimates improve elsewhere. The two new
neural studies still test those controlled decision mechanisms; they cannot
repair this structural component by training alone. A future change to data,
event or primary population needs a prospective scientific decision, not a
post-hoc declaration that these old results passed. Independent new-source
roles remain unassigned and reserved data remain closed.

This does not establish safety for the remaining views, change the 2% budget,
train a new model, prove a new theorem, or certify submission readiness.
Pixels and native annotation steps are not verified metres or seconds.
Stage5C and SMC remain disabled.
''')
    print(source.json.dumps({k: v for k, v in result.items() if k != 'zero_mass_views'}, sort_keys=True))


if __name__ == '__main__':
    main()
