"""Reconstruct the frozen feature-head diagnosis from verified scalar groups."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_feature_cost_diagnosis import ARMS, ROLES, summarize

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_feature_cost_diagnosis_v1'


def main():
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    reg = read(PUBLIC/'registration.json')
    summary = read(PUBLIC/'summary.json')
    verified = read(PUBLIC/'verification.json')
    complete = read(PUBLIC/'complete.json')
    if not verified['all36_job_role_summaries_exact'] or verified['summary_sha256'] != sha(PUBLIC/'summary.json'):
        raise ValueError('Require separate complete frozen-head replay')
    paths = sorted((PUBLIC/'groups').glob('*.json'))
    groups = [read(p) for p in paths]
    if summarize(groups, reg['jobs']) != summary:
        raise ValueError('Scalar groups must reconstruct the complete summary')
    lines = ['# Frozen Interaction-Feature Cost Diagnosis', '', '## Material Passport', '',
        'Fresh_run frozen-model inference and target diagnosis on both SOURCE roles.',
        'This readout reconstructs verified public scalars, not another training run.',
        'All 36 models remain unchanged. H costs/amplitudes and completion-risk',
        'ratios match the parent exactly; all 36 role groups replay exactly.', '',
        '## Same-Selected-Set Cost Ratios', '',
        'Median recording/pipeline-view predicted/target mass ratios are descriptive.',
        'Each role/arm retains 444 repeated views, not 444 independent scenes.',
        'H/EH predictions are amplitude-scaled; R/ER are not. Actual targets use',
        'interpolated paths, observed labels and the original wholly-unknown bound.',
        'Missing steps inside partially observed futures are not recovered.', '',
        '| Role | Arm | H ratio | R ratio | EH ratio | ER ratio | All/easy violations |',
        '|---|---|---:|---:|---:|---:|---:|']
    med = lambda a, scope, k: a[scope][k]['quantiles_0_25_50_75_100'][2]
    for role in ROLES:
        for arm in ARMS:
            a = summary['results'][role][arm]
            ratios = [med(a, 'selected_actual_interpolation', k) for k in ('H', 'R', 'EH', 'ER')]
            lines.append(f'| {role} | {arm} | '+ ' | '.join(f'{v:.6f}' for v in ratios)
                         +f' | {a["all_risk"]["actual_violations"]}/{a["easy_risk"]["actual_violations"]} |')
    lines += ['', '## Numerator And Denominator Errors', '',
        '| Role | Arm | False-safe easy | EH below target | ER above target | Full legal EH ratio | Selected full EH ratio | Selected-ratio drop views |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for role in ROLES:
        for arm in ARMS:
            a = summary['results'][role][arm]
            e = a['easy_risk']
            lines.append(f'| {role} | {arm} | {e["predicted_safe_actual_violations"]} | '
                f'{e["false_safe_numerator_underpredicted"]} | {e["false_safe_denominator_overpredicted"]} | '
                f'{med(a,"full_legal_neural","EH"):.6f} | {med(a,"selected_full_proposal","EH"):.6f} | '
                f'{a["selected_full_EH_below_full_legal_views"]} |')
    lines += ['', 'The full-versus-selected comparison holds proposal targets fixed before',
        'amplitude. It must not be replaced by the different actual-interpolation',
        'ratio to attribute a selection effect. False-safe means predicted risk',
        '<=2% and actual diagnostic risk >2% on exactly the same selected set.',
        'This is a measurement of cost error, not proof of its unique cause.', '',
        '## Missing-Label Sensitivity (Posthoc Descriptive)', '',
        '| Role | Arm | Nonempty views with no unknown selected rows | Defined EH views | Median actual EH ratio | Easy violations |',
        '|---|---|---:|---:|---:|---:|']
    for role in ROLES:
        for arm in ARMS:
            a = summary['results'][role][arm]['posthoc_no_unknown_selected']
            val = a['median_EH_prediction_target_ratio']
            shown = 'not estimable' if val is None else f'{val:.6f}'
            lines.append(f'| {role} | {arm} | {a["record_views"]} | {a["defined_EH_views"]} | {shown} | {a["easy_violations"]} |')
    lines += ['', 'These views do not replace the full roster, repair partial labels or',
        'establish independent inference. Undefined reference ratios stay undefined.', '',
        '## Execution And Reproduction', '',
        f'First pass PID {complete["pid"]}: {complete["seconds"]:.3f} s, peak RSS {complete["peak_RSS_bytes"]:,} bytes.',
        f'Full replay PID {verified["pid"]}: {verified["seconds"]:.3f} s, peak RSS {verified["peak_RSS_bytes"]:,} bytes.',
        'Native arm64 Torch, CPU4/inter-op1, workers0. Zero optimizer updates.', '',
        '```sh', '.venv-pytorch/bin/python scripts/run_m3w_feature_cost_diagnosis.py register',
        '.venv-pytorch/bin/python scripts/run_m3w_feature_cost_diagnosis.py run',
        '.venv-pytorch/bin/python scripts/run_m3w_feature_cost_diagnosis.py verify',
        '.venv-pytorch/bin/python scripts/report_m3w_feature_cost_diagnosis.py', '```', '',
        'Completed phases are immutable. Use run --resume only for an incomplete',
        'same-definition pass; do not delete a result to relabel cached work fresh.', '',
        'SOURCE remains exposed development. The parent primary and complete risk',
        'remain failed/unsupported; this diagnosis changes no threshold, reference',
        'or deployment. No calibration/confirmation/DroneCrowd input was opened.',
        'Pixels/raw frames/silver are not metric, seconds or human gold. No new',
        'neural dynamics, physical safety, true 3D or foundation claim. Stage5C and',
        'SMC remain off. The research goal remains unmet.']
    (PUBLIC/'readout.md').write_text('\n'.join(lines)+'\n')
    sources = dict(result_source='verified_scalar_reconstruction_not_new_inference',
        reporter_sha256=sha(Path(__file__)), summary_sha256=sha(PUBLIC/'summary.json'),
        verification_sha256=sha(PUBLIC/'verification.json'),
        group_sources=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths])
    (PUBLIC/'report_sources.json').write_text(json.dumps(sources, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(readout=str(PUBLIC/'readout.md'), groups=len(groups))))


if __name__ == '__main__':
    main()
