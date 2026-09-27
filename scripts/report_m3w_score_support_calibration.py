"""Public aggregate views of frozen calibration/readout outputs."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_score_support_calibration as run


def interval(r, multiplier=1):
    if r['point'] is None or r['ci95'] is None: return 'undefined'
    return f"{r['point']*multiplier:.4f} [{r['ci95'][0]*multiplier:.4f}, {r['ci95'][1]*multiplier:.4f}]"


def report(doc):
    lines = ['# Source-Separated Calibration: Matched Readout', '',
        'Fresh_run: causal inference, empirical two-source calibration and held-source scoring.',
        'Cached_verified: frozen forecast banks and risk/utility/easy checkpoints. New training updates: 0.',
        'All results use development-exposed source localities, not independent confirmation.', '',
        '## Primary Comparison', '',
        'Signed-score calibrated-supported neural versus equally protected damping ADE gain (%): **'+
        interval(doc['paired_neural_vs_damping']['excess_calibrated_supported']['neural_vs_damping_ADE_gain_percent'])+'**.', '',
        '## All Prespecified Controls', '',
        '| Candidate | Policy | ADE gain vs CV % | Easy gain % | Hard gain % | Switch % | Risk-violating views | Worst view easy gain % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm, policies in doc['summary'].items():
        for p, m in policies.items():
            w = doc['worst_views'][arm][p]
            lines.append(f"| {arm} | {p} | {interval(m['all_gain_percent'])} | {interval(m['easy_gain_percent'])} | {interval(m['hard_gain_percent'])} | {interval(m['switch_rate'],100)} | {w['risk_violating_views']} | {w['worst_easy_gain_percent']:.4f} |")
    lines += ['', '## Neural Versus Matched Damping', '', '| Policy | Paired ADE gain % |', '|---|---:|']
    for p, m in doc['paired_neural_vs_damping'].items():
        lines.append(f"| {p} | {interval(m['neural_vs_damping_ADE_gain_percent'])} |")
    lines += ['', '## Primary By Seed', '', '| Seed | Paired ADE gain % |', '|---|---:|']
    for s, m in doc['primary_by_seed'].items(): lines.append(f"| {s} | {interval(m['neural_vs_damping_ADE_gain_percent'])} |")
    lines += ['', '## Checks and Claim Limits', '', *[f'- {k}: {v}' for k,v in doc['gates'].items()], '',
        'Intervals resample twelve locality means after averaging dependent producer/seed/calibration views.',
        'Each candidate/policy has216 held locality views. Overlapping windows are not independent samples.',
        'Positive-harm/reference is separate from net ADE or easy degradation. Empty selected ratios remain undefined.',
        'The complete static guard is present, but two-source empirical calibration is not a conformal certificate.',
        'Both objective families remain controls; no held-result model/threshold winner was chosen.',
        'No new independent source, scene-joint benefit or deployment promotion follows.',
        'Silver image-local obs8/pred12 at raw-frame stride12; not Stage37t50, metric, seconds, human gold,',
        'physical safety, true3D or foundation evidence. Stage5C/SMC remain disabled.']
    return '\n'.join(lines)+'\n'


def main():
    d = json.loads((run.PUBLIC/'summary.json').read_text())
    (run.PUBLIC/'results.md').write_text(report(d))
    frozen = json.loads((run.PUBLIC/'decision_freeze.json').read_text())
    groups = {}
    for ref in frozen['groups']:
        assert run.artifact(ROOT/ref['path']) == ref
        rec = json.loads((ROOT/ref['path']).read_text())
        arm = 'dimensionless' if '_dimensionless_' in rec['group'] else 'damped'
        for name, s in rec['selections'].items():
            g = groups.setdefault(arm+'_'+name, dict(views=0, all_fallback=0, threshold_counts={}, eligible_thresholds=0))
            g['views'] += 1; g['all_fallback'] += s['threshold'] is None
            key = str(s['threshold']); g['threshold_counts'][key] = g['threshold_counts'].get(key,0)+1
            g['eligible_thresholds'] += sum(x['eligible'] for x in s['candidates'])
    run.immutable_json(run.PUBLIC/'calibration_summary.json', groups)
    rows = ['# Calibration Availability', '',
        'Only the two declared calibration sites contribute to each threshold. No held scores select a model.', '',
        '| Candidate/objective/support | Calibration views | All-fallback views | Selected cutoffs |', '|---|---:|---:|---|']
    for name,g in groups.items(): rows.append(f"| {name} | {g['views']} | {g['all_fallback']} | {json.dumps(g['threshold_counts'],sort_keys=True)} |")
    rows += ['', 'Counts are dependent views, not independent calibration datasets.32 selected rows are an',
        'operational support floor, not32 independent trajectories. The99% fitting support limit is a heuristic',
        'diagonal-distance detector; passing it does not establish in-domain support or risk validity.']
    (run.PUBLIC/'calibration_summary.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps(dict(primary=d['paired_neural_vs_damping']['excess_calibrated_supported'], gates=d['gates'])))


if __name__ == '__main__': main()
