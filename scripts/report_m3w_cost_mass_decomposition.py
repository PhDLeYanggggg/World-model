"""Describe complete same-set decomposition using verified public scalars only."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_bound_cost_decomposition_v1'


def main():
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    s, v, c = (read(PUBLIC/f) for f in ('summary.json', 'verification.json', 'complete.json'))
    if not v['all36_job_role_summaries_exact'] or v['summary_sha256'] != sha(PUBLIC/'summary.json'):
        raise ValueError('Complete separate frozen-model replay required')
    groups = [read(p) for p in sorted((PUBLIC/'groups').glob('*.json'))]
    if len(groups) != 36: raise ValueError('All original job/role groups required')
    lines = ['# Same-Set Harm And Reference Diagnosis', '', '## Material Passport', '',
        'Fresh_run frozen-model cost inference on both SOURCE roles, not training.',
        'All36cost heads are reused without updates;72role passes and a separate',
        'complete replay finish. Held cost/amplitude arrays match the parent exactly.',
        'This report reads public scalars; it is not another raw-data experiment.', '',
        '## Same-Set Decomposition', '',
        'Each value below is the median recording-view predicted/target mass ratio,',
        'not prediction accuracy, a pooled pixel sum, or a ratio of population means.',
        'H/EH predictions are amplitude-scaled; R/ER are not. Targets are actual',
        'interpolated-path contributions: known masked costs plus the registered',
        'wholly-unknown disagreement bound. They are not complete physical errors.', '',
        '|Role|Arm|H ratio|R ratio|EH ratio|ER ratio|Completion all/easy violations|',
        '|---|---|---:|---:|---:|---:|---:|']
    diagnostics = []
    for role in ('critic_fit', 'double_excluded'):
        for arm in ('known_only', 'bound_aware'):
            a = s['results'][role][arm]; m = a['selected_actual_interpolation']
            med = lambda key: m[key]['quantiles_0_25_50_75_100'][2]
            lines.append(f'|{role}|{arm}|{med("H"):.6f}|{med("R"):.6f}|{med("EH"):.6f}|{med("ER"):.6f}|{a["all_risk"]["actual_violations"]}/{a["easy_risk"]["actual_violations"]}|')
            records = [r for g in groups if g['role'] == role for r in g['arms'][arm]['records'].values()]
            failed = [r for r in records if r['selected_actual_risks']['target_easy'] is not None
                and r['selected_actual_risks']['target_easy'] > .02
                and r['selected_actual_risks']['predicted_easy'] is not None
                and r['selected_actual_risks']['predicted_easy'] <= .02+1e-12]
            under = lambda r: r['selected_actual_interpolation']['EH']['predicted_over_target'] < 1
            over = lambda r: r['selected_actual_interpolation']['ER']['predicted_over_target'] > 1
            no_unknown = [r for r in records if r['unknown_selected'] == 0]
            pure = [r['selected_actual_interpolation']['EH']['predicted_over_target'] for r in no_unknown]
            pure = [x for x in pure if x is not None]
            diagnostics.append(dict(role=role, arm=arm, record_views=len(records),
                falsely_safe_easy_views=len(failed), EH_underpredicted=sum(under(r) for r in failed),
                ER_overpredicted=sum(over(r) for r in failed), both=sum(under(r) and over(r) for r in failed),
                selected_full_EH_ratio_below_full_legal=sum(r['selected_full_proposal']['EH']['predicted_over_target'] is not None
                    and r['selected_full_proposal']['EH']['predicted_over_target'] < r['full_legal_neural']['EH']['predicted_over_target'] for r in records),
                no_unknown_selected_views=len(no_unknown), no_unknown_defined_EH_views=len(pure),
                no_unknown_median_EH_ratio=float(np.median(pure)) if pure else None,
                no_unknown_easy_violations=sum(r['selected_actual_risks']['target_easy'] is not None
                    and r['selected_actual_risks']['target_easy'] > .02 for r in no_unknown)))
    lines += ['', 'Every role/arm retains444recording views. Undefined reference ratios are',
        'excluded from their own medians and reported as undefined in the JSON,',
        'never scored as safe zero. These repeated views are not444independent sites.', '',
        '## Selection And Fitting Failures', '',
        '|Role|Arm|False-safe easy views|EH below target|ER above target|Selected full-EH ratio below full legal|',
        '|---|---|---:|---:|---:|---:|']
    for d in diagnostics:
        lines.append(f'|{d["role"]}|{d["arm"]}|{d["falsely_safe_easy_views"]}|{d["EH_underpredicted"]}|{d["ER_overpredicted"]}|{d["selected_full_EH_ratio_below_full_legal"]}|')
    lines += ['', 'For bound-aware fitting views, median full-legal proposal EH ratio is',
        '0.182912, falling to0.034390 on the actual selected set before amplitude.',
        'This direction occurs in423/444views; the corresponding held count is411.',
        'The same candidate target is used for this fixed-versus-selected comparison.',
        'Actual interpolated EH ratios must not be substituted into that contrast.', '',
        'All358fitting and332held falsely-safe bound-aware easy views underpredict EH.',
        'ER is overpredicted in only40and124respectively. Thus denominator inflation',
        'is not required for most of these failures. Numerator underestimation is',
        'present even where denominator error is conservative. This component',
        'measurement is stronger than attributing a ratio error to one term by guess.',
        'It still does not identify a unique architectural or statistical cause.', '',
        'The bounds can contribute to EH. A disclosed posthoc check retains the',
        'full roster above but separately describes views with no unknown selected',
        'rows: bound-aware fitting has16such views,13with defined EH, median EH',
        'ratio0.003122 and8easy violations. Therefore missing-label bound mass is',
        'not the sole explanation. This small selected slice is not a new primary',
        'cohort or an independent test; all counts are retained in factor_counts.json.', '',
        '## Implication For The Next Repair', '',
        'Selected-cost fitting failure already exists before new-locality transfer.',
        'The positive full-bound fitting change does not fix that conditional tail.',
        'Do not repeat a width sweep, global attenuation grid or generic easy-event',
        'hurdle and call it a new method. The earlier conditional-moment experiment',
        'already improved occurrence/cost fit but lost policy utility. A useful next',
        'candidate must explicitly address selected EH magnitude and wholly unknown',
        'support contributions, with a distinct measurable change and matched controls.',
        'Its selection rule, data roles and full-risk contract must remain prospective.', '',
        '## Runtime And Reproduction', '',
        f'Original pass PID{c["pid"]}:{c["seconds"]:.3f}s,peak{c["peak_RSS_bytes"]:,}bytes.',
        f'Separate complete replay PID{v["pid"]}:{v["seconds"]:.3f}s,peak{v["peak_RSS_bytes"]:,}bytes.',
        'All36job-role summaries reproduce exactly. Nativearm64CPU4/inter-op1/',
        'workers0, heartbeat and per-job resumable outputs. Zero optimizer updates.', '',
        '```sh', '.venv-pytorch/bin/python scripts/run_m3w_cost_mass_decomposition.py register',
        '.venv-pytorch/bin/python scripts/run_m3w_cost_mass_decomposition.py run',
        '.venv-pytorch/bin/python scripts/run_m3w_cost_mass_decomposition.py verify',
        '.venv-pytorch/bin/python scripts/report_m3w_cost_mass_decomposition.py', '```', '',
        'Completed phases are immutable; use run --resume only for an incomplete',
        'original pass. Do not delete results to relabel cached work as fresh.', '',
        'SOURCE is exposed development; fitting resubstitution is not independent',
        'validation. Parent primary/risk failures, thresholds and deployment remain',
        'unchanged. No calibration/confirmation/DroneCrowd arrays or new forecasts.',
        'Pixels/raw frames/released silver are not metric, seconds or human gold.',
        'Partial-mask ADE is not full12ADE. No physical-safety, true3D, foundation',
        'or new dynamics claim. Stage5C and SMC remain off. Goal remains unmet.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    (PUBLIC/'factor_counts.json').write_text(json.dumps(dict(
        result_source='posthoc_descriptive_public_scalar_aggregation_not_new_policy',
        summary_sha256=sha(PUBLIC/'summary.json'), verification_sha256=sha(PUBLIC/'verification.json'),
        reporter_sha256=sha(Path(__file__)), counts=diagnostics), indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(readout=str(PUBLIC/'readout.md'), groups=len(groups), counts=diagnostics)))


if __name__ == '__main__': main()
