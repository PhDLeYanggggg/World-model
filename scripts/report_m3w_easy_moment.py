"""Lightweight tables from verified fixed easy-moment analyses."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/easy_moment_v1'


def main():
    raw=(PUBLIC/'analysis.json').read_bytes();a=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
    for name in ('verification.json','independent_verification.json'):
        r=json.loads((PUBLIC/name).read_text())
        assert r['all_checks_passed'] and r['analysis_sha256']==sha
    rows=[]
    for name,s in a['summary'].items():
        worst=max(0.,max(-r['subsets']['positive_easy']['worst_scene_gain_percent'] for r in s['seeds'].values()))
        rows.append(dict(policy=name,ADE_gain_percent=s['ADE']['equal_scene_gain_percent'],
            ADE_CI_low=s['ADE']['scene_bootstrap_ci95'][0],ADE_CI_high=s['ADE']['scene_bootstrap_ci95'][1],
            FDE_gain_percent=s['FDE']['equal_scene_gain_percent'],hard_gain_percent=s['subsets']['hard']['equal_scene_gain_percent'],
            worst_site_seed_easy_degradation_percent=worst,
            mean_switch_rate_percent=100*sum(r['selected'] for r in s['seeds'].values())/(3*a['rows']),
            zero_CV_harmed_total=sum(r['zero_CV_harmed'] for r in s['seeds'].values())))
    with (PUBLIC/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Conditional Easy-Moment Results','','Development only; equal-site relative ADE gain over CV. Three seeds, four exposed sites.',
        'No metric/seconds, independent safety, confirmation or deployment claim. All fixed policies retained.','',
        '| Policy | ADE gain % [CI95] | Hard gain % | Worst easy degradation % | Switch % | Zero-CV harmed, summed seeds |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['policy']} | {r['ADE_gain_percent']:.4f} [{r['ADE_CI_low']:.4f}, {r['ADE_CI_high']:.4f}] | {r['hard_gain_percent']:.4f} | {r['worst_site_seed_easy_degradation_percent']:.4f} | {r['mean_switch_rate_percent']:.3f} | {r['zero_CV_harmed_total']} |")
    lines+=['','## Paired Contrasts','','Nominal exploratory physical-site bootstrap intervals; no multiple-comparison claim.','',
        '| Contrast | ADE difference pp | CI95 |','|---|---:|---|']
    for key,r in a['contrasts'].items():lines.append(f"| {key} | {r['all']['mean_gain_difference_pp']:.4f} | {r['all']['ci95_pp']} |")
    (PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')
    loss=['# Fitting Losses','','Four bounded moment targets; draw-weighted fitting MSE, not validation loss or forecast ADE.',
        'No early stopping or outer-outcome selection. All 36 fits use 128 trees.','',
        '| View | Action | First MSE (16 trees) | Final MSE (128 trees) | Fit seconds |','|---|---|---:|---:|---:|']
    for r in a['fits']:
        f=r['fit'];assert f['complete'] and f['trees']==128 and f['sampled_rows']==768000
        loss.append(f"| {r['view']} | {r['action']} | {f['trace'][0]['fitting_mean_mse']:.8f} | {f['trace'][-1]['fitting_mean_mse']:.8f} | {f['seconds']:.3f} |")
    seconds=sum(r['fit']['seconds'] for r in a['fits'])
    loss+=['',f'Summed fitting-loop time: {seconds:.3f} seconds, not total wall time. Per-target traces are in analysis.json.']
    (PUBLIC/'training_losses.md').write_text('\n'.join(loss)+'\n')
    compact=dict(analysis_sha256=sha,result_source=a['result_source'],rows=a['rows'],complete_rows=a['complete_rows'],
        new_risk_forests=36,new_forecasters=0,seed_count=3,physical_site_count=4,fitting_seconds=seconds,
        fitting_seconds_are_not_wall_time=True,policies=rows,contrasts=a['contrasts'],
        independent_confirmation=False,deployment=False,stage5c_executed=False,smc_enabled=False)
    (PUBLIC/'compact_results.json').write_text(json.dumps(compact,indent=2)+'\n')
    print(json.dumps(compact,indent=2))


if __name__=='__main__':main()
