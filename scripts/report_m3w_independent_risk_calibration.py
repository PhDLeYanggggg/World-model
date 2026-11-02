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
    (PUBLIC/'failure_analysis.md').write_text(
        '# Calibration Limitations And Failure Analysis\n\n'
        + ('No multiplier supports every required recording view. Shrinking intervention '
           'does not turn empty or undefined denominators into evidence of safety.\n\n' if chosen is None else
           'A common multiplier fits this dedicated calibration cohort. Independent '
           'confirmation is still unrun; passing calibration is not deployment approval.\n\n')
        + 'The frozen score family may underestimate conditional harm under domain shift. '
        'A stricter scalar can suppress useful actions and lose finite support; it cannot '
        'relearn the ranking or create missing labels. The full fixed grid and per-locality '
        'statistics are retained. No locality is removed and no failed historical protocol '
        'is relabeled. Next research must separate score accuracy, conditional support and '
        'calibration limitations instead of optimizing these calibration labels repeatedly.\n')
    print(json.dumps(dict(readout=str(PUBLIC/'readout.md'), selected_multiplier=chosen)))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
