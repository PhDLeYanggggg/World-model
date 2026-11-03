"""Describe the complete fourfold fitting probe without claiming held utility."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_feature_duration_probe_v1'


def shown_median(values, moment):
    quantiles = values['selected_actual_interpolation'][moment]['quantiles_0_25_50_75_100']
    return 'not estimable' if quantiles is None else f'{quantiles[2]:.6f}'


def main():
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    summary, verify, complete, reg = [read(PUBLIC/f) for f in ('summary.json','verification.json','complete.json','registration.json')]
    if not verify['all18_fitting_decompositions_exact'] or verify['summary_sha256'] != sha(PUBLIC/'summary.json'):
        raise ValueError('Require complete separate fitting diagnostic replay')
    paths = sorted((PUBLIC/'fits').glob('*.json'))
    fits = [read(p) for p in paths]
    if len(fits) != 18 or {v['job'] for v in fits} != {j['key'] for j in reg['jobs']}:
        raise ValueError('All18 original jobs required')
    if sum(v['updates'] for v in fits) != reg['planned_primary_optimizer_updates']:
        raise ValueError('Complete fourfold update coverage required')
    for fit in fits:
        a, b = fit['sets']['original'], fit['sets']['extended_fixed_original_selection']
        if set(a) != set(b):
            raise ValueError('Fixed selected recording cohort required')
        for rec in a:
            if (a[rec]['selected'],a[rec]['unknown_selected']) != (b[rec]['selected'],b[rec]['unknown_selected']):
                raise ValueError('Original selected rows must remain fixed')
            for k in ('B','H','R','ER','EH'):
                if a[rec]['selected_actual_interpolation'][k]['target_mass'] != b[rec]['selected_actual_interpolation'][k]['target_mass']:
                    raise ValueError('Fixed original paths and target contributions required')
    objectives = {label:{arm:float(np.mean([v[label+'_objectives'][arm]['total'] for v in fits]))
                        for arm in ('known_only','bound_aware')} for label in ('original','extended')}
    result = dict(result_source='verified_scalar_readout_not_new_training',
        objectives=objectives, summary_sha256=sha(PUBLIC/'summary.json'),
        verification_sha256=sha(PUBLIC/'verification.json'),
        fixed_original_selected_rows_and_actual_targets_exact=True,
        held_policy_evaluation=False, deployment_changed=False)
    lines = ['# Fourfold Training Adequacy: Fitting-Only Results', '', '## Material Passport', '',
        'Fresh_run: all18 real Torch extended fits, including original-prefix replay.',
        'Cached_verified: original frozen controls. Fresh separate final fitting',
        'diagnostic replay, without new updates. This report reads public scalars.', '',
        f'Actual primary updates: {summary["primary_optimizer_updates"]:,}; first full-refit updates: {summary["first_full_refit_updates"]:,}.',
        'The three actual seeds and original G-role roster are unchanged. Every',
        'old-budget scientific prefix is exact. Capacity, loss and sampling are',
        'unchanged; compute/exposure are deliberately fourfold, not matched.', '',
        '## Mean Whole-Fitting Objective', '',
        'Equal mean across18 original jobs. Known-only is a diagnostic objective',
        'on the same fitted model, not another separately trained arm.', '',
        '| Objective | Original budget | Fourfold budget | Reduction |',
        '|---|---:|---:|---:|---:|']
    for arm in ('bound_aware','known_only'):
        a, b = objectives['original'][arm], objectives['extended'][arm]
        lines.append(f'| {arm} | {a:.6f} | {b:.6f} | {100*(a-b)/a:.2f}% |')
    lines += ['', '## Same-Set Calibration', '',
        'These are median predicted/target mass ratios over444 repeated fitting',
        'recording/pipeline views, not independent-scene inference. Fixed-original',
        'selection holds rows, amplitudes, actual paths and target masses exactly.',
        'Thus any change there is prediction fitting, not less intervention.', '',
        '| Prediction / selected set | H ratio | R ratio | EH ratio | ER ratio | Actual all/easy violations |',
        '|---|---:|---:|---:|---:|---:|']
    for name, values in summary['results'].items():
        lines.append(f'| {name} | '+ ' | '.join(shown_median(values,k) for k in ('H','R','EH','ER'))
            +f' | {values["all_risk"]["actual_violations"]}/{values["easy_risk"]["actual_violations"]} |')
    lines += ['', 'Actual violations on the fixed original set cannot change: the paths and',
        'labels are identical. Own-selection fitting risk can change but is not',
        'held generalization, an original complete-risk pass or deployability.',
        'Undefined support stays undefined; no zero-risk substitution.', '',
        '## Execution And Reproduction', '',
        f'Training/refit/diagnostics PID{complete["pid"]}: {complete["seconds"]:.3f}s, peak RSS{complete["peak_RSS_bytes"]:,}bytes.',
        f'Separate fitting replay PID{verify["pid"]}: {verify["seconds"]:.3f}s, peak RSS{verify["peak_RSS_bytes"]:,}bytes.',
        'The full-refit log intentionally repeats step numbers under the same job',
        'name: its checkpoint is first_full_refit.pt.gz, not a budget rollback.', '',
        '```sh', '.venv-pytorch/bin/python scripts/run_m3w_feature_duration_probe.py register',
        '.venv-pytorch/bin/python scripts/run_m3w_feature_duration_probe.py train',
        '.venv-pytorch/bin/python scripts/run_m3w_feature_duration_probe.py verify',
        '.venv-pytorch/bin/python scripts/report_m3w_feature_duration_probe.py', '```', '',
        'Use train --resume for an interrupted same-identity run only. Completed',
        'phases are immutable. Private data/checkpoints are required; scalar',
        'reconstruction is not anonymous cold-start reproduction.', '',
        'No new H policy scoring or threshold selection. A positive fitting result',
        'requires a separate frozen held evaluation before any generalization',
        'claim. The original feature primary remains inconclusive and risk failed.',
        'SOURCE is exposed development. Pixels/raw frames/silver are not metric,',
        'seconds or human gold. Stage5C execution/SMC remain off; goal remains unmet.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    result.update(reporter_sha256=sha(Path(__file__)),
        fit_sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths])
    (PUBLIC/'report_sources.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(readout=str(PUBLIC/'readout.md'),objectives=objectives)))


if __name__ == '__main__':
    main()
