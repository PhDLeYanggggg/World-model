"""Render completed engineering evidence; never refit or select a predictor."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest


def main():
    out = ROOT/'outputs/publication_readiness_2026_09/interaction_controls_v1'
    path = out/'checks.json'
    r = json.loads(path.read_text())
    for name,digest in r['identity'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Changed checked dependency: '+name)
    s,q,n = r['synthetic'],r['real_inputs'],r['numerical_regression']
    rows = q['cases']
    report = f'''# Matched Interaction Controls: Engineering Evidence

Result source: fresh computation on constructed problems and hash-verified
past inputs from existing SDD source annotations. No new fitting, future-label
readout, deployment or primary-metric change. Not a predictive success result.
Completed report SHA256: `{file_digest(path)}`.

## Why Another Control Is Necessary

The full pair objective contains both single-agent geometry terms and genuinely
non-additive terms. Joint selection beating a risk-only independent policy does
not by itself establish a coordination benefit. The new independent-geometry
control keeps the single-agent geometry terms and removes only binary products.
All arms receive the same issued forecasts, gain/harm estimates, support and
original risk cap. Unary geometry and joint selection use the causal independent
reference count. No future label determines membership or the count.

This is an opt-in diagnostic adapter, not a silent change to the frozen policy.
The risk budget is not recomputed from geometry-adjusted gains. All three new
arms use the same versioned numerical solver. Historical runs and the legacy
solver remain intact. See [derivation and limits](method_and_limits.md).

## Completed Checks

| Check | Result |
| --- | ---: |
| Constructed problems | {s['problems']} |
| Enumerated assignments | {s['enumerated_assignments']} |
| Verified unary/full optima | {s['exact_optima_verified']} |
| Maximum decomposition discrepancy | {s['max_decomposition_error']:.3g} |
| Original recordings / physical sites | {q['recordings']} / 4 |
| Past scene queries / agent queries | {q['queries']} / {q['agent_queries']} |
| Matched / nonzero matched / unmatched queries | {q['matched_queries']} / {q['nonzero_matches']} / {q['unmatched_queries']} |
| Future-array poison checks | {q['poison_checks']} |
| Calls to future-label APIs | {q['label_api_calls']} |
| Queries with potentially active non-additive edges | {q['potential_product_queries']} |
| Queries with different unary/joint switch identities | {q['controls_differing_in_switch_identity']} |
| Queries with full-objective advantage >1e-10 | {q['constructed_score_proxy_advantage_queries']} |

The source check uses fixed CV and damped-velocity forecasts and explicitly
constructed past-only scores, not trained neural scores. Three deterministic
timestamps per recording are engineering probes, not representative accuracy
samples or independent calibration units. Future-array poisoning confirms this
code does not consume later target values; supplied historical annotations may
themselves be offline interpolations, as already disclosed.

The three favorable constructed-objective cases are hyang/video0 frame5724,
hyang/video4 frame4068 and hyang/video7 frame84. No future error was measured.
Different switch identities can be tied optima, and do not establish superiority.
All arms count selected agent identities; a selected forecast can equal its
baseline. Changed-forecast counts are recorded separately and happen to match
in these probes. That is not a general equality or a matched-realized-risk claim.

## Numerical Failure And Repair

The first real-input run stopped at hyang/video3 frame84, after 78 completed
queries. Both solutions were called optimal by the legacy solver, but the joint
objective was worse than a feasible unary-control witness. No completed report
was written by that failed run. A read-only diagnostic initially passed an array
where coordinate restoration required an agent-keyed mapping; that diagnostic
call was corrected before the numerical replay.

| Quantity | Value |
| --- | ---: |
| Agents / graph edges / exact count | {n['agents']} / {n['edges']} / {n['exact_count']} |
| Enumerated feasible assignments | {n['enumerated_feasible_assignments']} |
| Legacy direct objective | {n['legacy_direct_objective']:.17g} |
| Enumerated optimum | {n['enumerated_optimum']:.17g} |
| Legacy error above optimum | {n['legacy_error_to_enumerated_optimum']:.12g} |
| Legacy reported relative gap despite success | {n['legacy_reported_relative_gap']:.12g} |
| Corrected direct objective | {n['corrected_objective']:.17g} |

The observed failure is scale-sensitive numerical termination. Setting forwarded
`mip_abs_gap=0` alone did **not** fix this instance on SciPy1.17.1; it is therefore
not evidence that one default absolute-gap setting is the sole root cause.
Positive global objective scaling by 1e6 did fix it. The versioned control solver
scales coefficients, retains original constraints and checks binary products,
recomputed objective, returned primal value and dual bound in original units.
Original-unit tolerance is 1e-9*(1+abs(objective)); unverifiable outputs fall back
and remain unmatched. This tolerance is numerical, not a risk guarantee.
All completed probes have primal/dual discrepancies at most about3.5e-18.

The upstream [SciPy MILP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
describes the solver result and gap fields; the [HiGHS options reference](https://ergo-code.github.io/HiGHS/dev/options/definitions/)
documents numerical stopping options. Those references motivate checks, while
the local controlled replay is the evidence for this particular repair.

## Limits And Next Experiment

Real predictive lift: **not_run**. Independent risk calibration: **not_run**.
Same-version semantic replay of the completed engineering report passes.
There is no new ADE/FDE score, confidence interval, model promotion, metric/seconds
claim, Stage5C execution or SMC. Bookstore, primary and external readouts stay
closed. Existing negative forecasts remain negative.

After the pending primary-evaluation decision, register the same-predictor
risk-only/unary-geometry/full-joint comparison before fitting or reading new
evaluation results. Report accuracy, simple-case harm, tail/worst-site errors,
changed-forecast counts and proximity together. A lower proxy without reliable
predictive utility will not establish the proposed contribution.
'''
    (out/'conclusions.md').write_text(report)
    table = ['# All Past-Input Engineering Queries','',
        'Constructed scores and proxy objectives only; no future accuracy readout.','',
        '| Recording | Frame | Agents | Count | Non-additive edges | Joint proxy advantage | Changed forecasts I/U/J |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        changed = row['selected_nonidentical_forecasts']
        table.append(f"| {row['recording']} | {row['frame']} | {row['agents']} | {row['reference_count']} | "
            f"{row['potential_nonadditive_edges_at_count']} | {row['predicted_full_objective_advantage']:.12g} | "
            f"{changed['independent']}/{changed['unary_geometry']}/{changed['joint']} |")
    (out/'complete_cases.md').write_text('\n'.join(table)+'\n')
    print('Rendered completed report and all 99 query summaries; no new model or scoring')


if __name__ == '__main__':
    main()
