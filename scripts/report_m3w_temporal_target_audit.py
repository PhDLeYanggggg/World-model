"""Render verified temporal-probe results without choosing a model or threshold."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_m3w_temporal_target_audit import PUBLIC, sha


def main():
    receipt = json.loads((PUBLIC/'verification.json').read_text())
    assert sha(PUBLIC/'summary.json') == receipt['summary_sha256']
    assert sha(PUBLIC/'complete.json') == receipt['complete_sha256']
    s = json.loads((PUBLIC/'summary.json').read_text()); rows = []
    for ref in json.loads((PUBLIC/'complete.json').read_text())['groups']:
        assert sha(ROOT/ref['path']) == ref['sha256']
        rows.append(json.loads((ROOT/ref['path']).read_text()))
    lines = ['# Temporal Target Probe Results', '',
        'fresh_run: TRAIN-only analytic probe fitting, exact refitting and validation readout.',
        'cached_verified: original forests, upstream forecasts, source labels and action hashes.',
        'not_run: neural auxiliary training, changed policy, independent calibration/confirmation.', '',
        'No future masks enter probe inference. No primary risk definition or deployment changes.', '',
        '| Cohort / temporal-leaf contrast | Mean normalized MSE change | Nominal 95% locality CI | Localities |',
        '|---|---:|---|---:|']
    for k, v in s['contrasts'].items():
        value = 'undefined' if v['mean'] is None else f"{v['mean']:+.6f}"
        interval = 'undefined' if v['CI95'] is None else f"[{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"
        lines.append(f"| {k} | {value} | {interval} | {len(v['localities'])} |")
    p = s['pooled_validation']
    lines += ['', '## Frozen Selected Cohort', '',
        'Counts below are repeated head-view occurrences, not independent samples.', '',
        '| Quantity | Value |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in p.items()]
    if p['selected_step_harm'] > 0:
        lines += ['', f"Cancellation share of pooled gross step harm: {100*p['selected_cancellation']/p['selected_step_harm']:.4f}%."]
    lines += ['', 'Gross step harm minus cancellation reconstructs the original whole-trajectory harm.',
        'These image-local masses are descriptive. They are neither risk percentages nor ADE/FDE gains.', '',
        '## Per-Locality Signed-Error Contrast', '',
        '| Locality | Temporal minus row-mean leaf | Temporal minus global time mean |', '|---|---:|---:|']
    for site in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['source'] == site]
        means = [math.fsum(r['probes']['validation']['temporal_leaf'][0]-r['probes']['validation'][k][0] for r in group)/len(group)
                 for k in ('rowmean_leaf', 'global_temporal')]
        lines.append(f'| {site} | {means[0]:+.6f} | {means[1]:+.6f} |')
    lines += ['', '## Registered Diagnostic Screen', '',
        f"Advance to separately specified auxiliary-training design: {s['advance_to_auxiliary_design']}.",
        'This screen cannot certify a deployment policy or a world-model improvement.', '',
        *['- '+k for k in s['failure_reasons']], '',
        'The 3,000-resample intervals cluster by locality and are nominal exposed-development',
        'evidence, not search-adjusted independent confirmation. Scores retain the observed-label',
        'task; a complete-label stratum is diagnostic, never a future-based inference filter.', '',
        'Image-local detector-silver, rawstride12 obs8/pred12. No seconds, metric, physical-safety,',
        'true-3D or foundation claim. Stage5C and SMC remain off.', '']
    path = PUBLIC/'results.md'; text = '\n'.join(lines)
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)
    print(json.dumps(dict(groups=len(rows),advance_to_auxiliary_design=s['advance_to_auxiliary_design'],
        result_report=str(path.relative_to(ROOT))), indent=2))


if __name__ == '__main__': main()
