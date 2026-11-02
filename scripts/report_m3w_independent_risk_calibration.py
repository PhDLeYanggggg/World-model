"""Public scalar report; never reads private trajectories, models or labels."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_independent_risk_calibration_v1'


def render(summary, phases):
    selection = summary['selection']; chosen = selection['selected_multiplier']
    lines = ['# Independent Risk Calibration', '',
        'Result source: fresh_run dedicated-role conversion, frozen Torch inference,',
        'allocation and scoring; cached_verified SOURCE-trained weights and source-only',
        'normalization. No new optimizer updates. The selection scenes are not reused',
        'as calibration scenes. Final confirmation and DroneCrowd remain closed.', '',
        '## Complete Cohort And Decision', '',
        '102 recordings in 12 locality groups; nine frozen pipelines, hence 918',
        'repeated pipeline-recording views, not 918 independent recordings.',
        'All 13 multipliers and four allocation policies freeze before labels.',
        'Original all/easy positive-harm and easy-net budgets remain 2%.', '',
        f'Selected common multiplier: **{chosen if chosen is not None else "none"}**.',
        f'All required calibration views supported: **{selection["calibration_supported"]}**.',
        'If no multiplier qualifies, the global physical floor is the separate fallback,',
        'not a calibrated learned success. No deployment promotion is made here.', '',
        '## Fixed-Grid Readout', '',
        '|Lambda|Policy|Known ADE gain % [conditional locality CI]|Completion-lower gain %|Supported / 918|Undefined views|Worst easy net %|',
        '|---:|---|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'N/A' if v is None else f'{v:.6f}'
    for curve in summary['curves']:
        for arm, value in curve['policies'].items():
            stat = value['known_gain_percent']; ci = stat['CI95']
            estimate = fmt(stat['point']) + (f' [{fmt(ci[0])}, {fmt(ci[1])}]' if ci is not None else '')
            lines.append(f'|{curve["multiplier"]}|{arm}|{estimate}|{fmt(value["lower_gain_percent"]["point"])}|'
                         f'{value["supported_record_views"]}/{value["required_record_views"]}|'
                         f'{value["unsupported_views"]}|{fmt(value["worst_easy_net_percent"])}|')
    lines += ['', '## Interpretation', '',
        'These are in-sample calibration outcomes after fitting the common multiplier.',
        'The locality bootstrap describes conditional uncertainty; it does not turn',
        'calibration fitting into independent efficacy or a population risk certificate.',
        'Twelve localities, correlated recordings, ratio denominators, partial labels',
        'and reselected/nonmonotone actions do not justify applying a general conformal',
        'risk guarantee without its assumptions and an appropriate loss/test procedure.',
        'See [Learn then Test](https://arxiv.org/abs/2110.01052) for hypothesis-testing',
        'risk control and [Conformal Risk Control](https://arxiv.org/abs/2208.02814)',
        'for monotone-loss expected-risk calibration; neither theorem is claimed here.', '',
        'Available-label ADE is not full12 ADE. Completion bounds cover wholly unknown',
        'rows only, not missing steps inside partial trajectories. Endpoint FDE is',
        'diagnostic if missing. Released tracker/detector silver is not human gold.',
        'Inputs are retrospective past-only, not a proof of online sensor causality.',
        'Local image pixels and raw stride12 are not metric units or seconds.',
        'This is not true3D, a foundation model, Stage5C execution or SMC.', '',
        '## Execution Receipts', '',
        '|Phase|PID|Seconds|Peak RSS bytes|', '|---|---:|---:|---:|']
    for name, phase in phases.items():
        lines.append(f'|{name}|{phase["pid"]}|{phase["seconds"]:.6f}|{phase["peak_RSS_bytes"]}|')
    lines += ['', 'Registration, phase receipts and scalar curves are public. All weights,',
        'input/feature arrays, labels and policy matrices remain private.',
        'Fresh raw/input replay, fresh model/cost replay, metric/feasibility replay',
        'and MILP reoptimization are distinct checks. See verify.json for exact scope.',
        'Original negative SOURCE/model-selection experiments remain unchanged.', '']
    return '\n'.join(lines)


def render_failure(summary, taxonomy, attribution):
    chosen = summary['selection']['selected_multiplier']
    lines = ['# Calibration Limitations And Failure Analysis', '',
        'Complete dedicated-role calibration, not independent confirmation. No model,',
        'policy, risk budget or threshold grid is changed by this diagnosis.', '',
        ('No common multiplier passes the original full-record contract. Global-floor'
         ' fallback is separate, not learned safe success.' if chosen is None else
         'A common multiplier fits this calibration cohort; confirmation is still required.'), '',
        '## Observed Error Harm Versus Support', '',
        '|Lambda|Policy|Observed-harm exclusive|Undefined reference|Unknown-bound only|Supported|No actual switches|',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in taxonomy['rows']:
        if r['arm'] not in ('independent', 'joint_matched'):
            continue
        c = r['exclusive']
        lines.append(f'|{r["multiplier"]}|{r["arm"]}|{c["observed_positive_harm"]}|'
            f'{c["unsupported_reference"]}|{c["unknown_completion_bound_only"]}|'
            f'{c["supported"]}|{r["overlapping"]["selected_none"]}|')
    lines += ['', 'Exclusive categories prioritize undefined reference before observed harm;',
        'overlapping flags are retained in risk_failure_taxonomy.json. Harm here means',
        'increased forecast error, not physical damage. Empty selected reference is',
        'not a measured safe zero. Full-roster easy-reference unavailable views:',
        f'{taxonomy["full_roster_easy_reference_unavailable_views"]}/918.', '',
        'The low-multiplier failures are not explained solely by unobserved outcomes.',
        'Increasing the multiplier suppresses actions, but finite selected support is',
        'lost and average gains approach zero. Passing signed easy net alone would',
        'not pass the separate original positive-harm/reference constraints.', '',
        '## Selected Cost Diagnosis', '',
        'This attribution was added after the completed readout, not preregistered as',
        'another primary result. It uses native costs, never lambda-scaled optimizer',
        'coefficients. Medians condition on defined record ratios; nine pipelines',
        'repeat recordings and do not create independent samples.', '',
        '|Lambda|Policy|Median predicted/true H|Median predicted/true EH|',
        '|---:|---|---:|---:|']
    for r in attribution['rows']:
        if r['multiplier'] not in (1, 4096):
            continue
        values = [r['record_ratio_summaries'][k]['median'] for k in
                  ('harm_prediction_ratio', 'easy_harm_prediction_ratio')]
        text = '|'.join('N/A' if v is None else f'{v:.6g}' for v in values)
        lines.append(f'|{r["multiplier"]}|{r["arm"]}|{text}|')
    lines += ['', 'The cost accounting separates harm underprediction from reference',
        'overprediction; it does not identify the causal training mechanism. A new',
        'SOURCE-side repair should address the selected easy-harm tail and verify',
        'SOURCE generalization, not tune repeatedly on these calibration labels.',
        'Future reuse of this cohort must disclose its outcome exposure.', '',
        'The complete paired policy effects are in paired_calibration_effects.md.',
        'Some grid points show positive descriptive effects, but inspecting the grid',
        'does not supply multiplicity-controlled superiority or repair failed risk.',
        'No favorable lambda is substituted for the registered unsupported selection.', '',
        'Available-label ADE is not full12 ADE. Unknown-row completion bounds do not',
        'recover missing steps inside partial trajectories. Pixels/raw stride12 and',
        'released tracker/detector silver do not support metric, seconds, human-gold,',
        'true3D or foundation claims. Confirmation and DroneCrowd remain closed.',
        'Stage5C and SMC remain off. No new controller is deployed.', '']
    return '\n'.join(lines)


def main():
    summary = json.loads((PUBLIC/'summary.json').read_text())
    phases = {name: json.loads((PUBLIC/(name+'.json')).read_text())
              for name in ('pilot', 'build', 'verify_inputs', 'infer', 'verify_inference', 'policy', 'labels', 'eval', 'verify')}
    (PUBLIC/'readout.md').write_text(render(summary, phases))
    chosen = summary['selection']['selected_multiplier']
    gates = dict(complete_calibration=True, labels_after_freeze=True,
        original_record_risk_supported=chosen is not None, population_certificate=False,
        confirmation_efficacy=False, deployment_promoted=False, stage5c=False, smc=False)
    (PUBLIC/'gates.md').write_text('# Calibration Gates\n\n'+'\n'.join(f'- {k}: {v}' for k,v in gates.items())+'\n')
    taxonomy = json.loads((PUBLIC/'risk_failure_taxonomy.json').read_text())
    attribution = json.loads((PUBLIC/'selected_cost_attribution.json').read_text())
    (PUBLIC/'failure_analysis.md').write_text(render_failure(summary, taxonomy, attribution))
    print(json.dumps(dict(readout=str(PUBLIC/'readout.md'), selected_multiplier=chosen)))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
