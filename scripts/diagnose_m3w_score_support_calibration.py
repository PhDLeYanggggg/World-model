"""Post-freeze attribution of calibration/support costs; no policy refitting."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_score_support_calibration as run


def main():
    doc = json.loads((run.PRIVATE/'summary_detail.json').read_text()); rows = doc['rows']
    index = {(r['group'],r['site'],r['policy']):r for r in rows}
    sites = sorted({r['site'] for r in rows}); out = {}
    for arm in ('dimensionless','damped'):
        for obj in ('moments','excess'):
            for policy,control in [('calibrated','guarded'),('supported','guarded'),
                                   ('calibrated_supported','supported'),('calibrated_supported','calibrated')]:
                paired = []
                for r in rows:
                    if r['candidate'] != arm or r['policy'] != obj+'_'+policy: continue
                    c = index[(r['group'],r['site'],obj+'_'+control)]['metric']; m = r['metric']
                    paired.append(dict(site=r['site'],metric=dict(
                        paired_ADE_gain_percent=100*(1-m['selected_error_sum']/c['selected_error_sum']) if c['selected_error_sum'] > 0 else None,
                        ADE_gain_change_points=m['all_gain_percent']-c['all_gain_percent'],
                        intervention_change_points=100*(m['switch_rate']-c['switch_rate']))))
                out[arm+'_'+obj+'_'+policy+'_vs_'+control] = {k:run.inter.paired_localities(paired,sites,k)
                    for k in paired[0]['metric']}
    violations = []
    for r in rows:
        if r['policy']=='excess_calibrated_supported' and r['metric']['risk_budget_violation']:
            violations.append({k:r[k] for k in ('site','group','candidate','seed')}|dict(
                positive_harm_ratio=r['metric']['selected_positive_harm_ratio'],
                all_gain_percent=r['metric']['all_gain_percent'],easy_gain_percent=r['metric']['easy_gain_percent'],
                switches=r['metric']['switches']))
    artifact = dict(contrasts=out,violations=violations,post_freeze=True,policy_changes=False,source=run.artifact(run.PRIVATE/'summary_detail.json'))
    run.immutable_json(run.PUBLIC/'factor_diagnosis.json',artifact)
    lines = ['# Frozen-Policy Factor Diagnosis', '',
        'Secondary paired contrasts from registered controls. No refit, winner selection or threshold change.', '',
        '| Contrast | Paired ADE gain % | 95% locality interval | Intervention change points |', '|---|---:|---|---:|']
    for k,v in out.items():
        r=v['paired_ADE_gain_percent']; cov=v['intervention_change_points']
        lines.append(f"| {k} | {r['point']:.6f} | {r['ci95']} | {cov['point']:.4f} |")
    lines += ['', '## Remaining Signed-Calibrated-Supported Risk Violations', '',
        '| Site | Seed | Candidate | Harm/reference % | Net ADE gain % | Easy gain % | Switches |', '|---|---:|---|---:|---:|---:|---:|']
    for r in violations:
        lines.append(f"| {r['site']} | {r['seed']} | {r['candidate']} | {100*r['positive_harm_ratio']:.4f} | {r['all_gain_percent']:.4f} | {r['easy_gain_percent']:.4f} | {r['switches']} |")
    lines += ['', 'Repeated source/seed rows can belong to different calibration pairs. Positive harm is not net',
        'degradation. Overlapping views are averaged at the locality level, not treated as independent samples.']
    (run.PUBLIC/'factor_diagnosis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(contrasts=len(out),remaining_risk_violations=len(violations))))


if __name__=='__main__': main()
