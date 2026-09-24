"""Render descriptive support evidence, not new outcome-selected thresholds."""
from pathlib import Path
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_zero_support_v1'


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a=json.loads((PUBLIC/'analysis.json').read_text()); rr=[]; percentiles=[]
    for ref in a['views']:
        p=ROOT/ref['path']; assert digest(p)==ref['sha256']
        v=json.loads(p.read_text()); rr.extend(v['records'])
        for arm in ('history','full_risk'):
            c=[r for r in v['records'] if r['arm']==arm and r['metadata_control']]
            for r in v['records']:
                if r['arm']==arm and r['zero_case']:
                    percentiles.append(dict(view=r['view'],action=r['action'],arm=arm,row=r['row'],
                        comparison_controls=len(c),fraction_controls_at_least_as_distant=float(np.mean([q['nearest_distance']>=r['nearest_distance'] for q in c]))))
    cases=a['raw_cases']; summary=dict(result_source='fresh_aggregation_of_verified_diagnostic',
        analysis_sha256=digest(PUBLIC/'analysis.json'),unique_zero_windows=len(cases),
        unique_zero_tracks=len({r['scoped_track'] for r in cases}),unique_zero_recordings=len({r['recording'] for r in cases}),
        raw_sampled_zero_matches=sum(r['raw_sampled_cv_ADE']==0 for r in cases),
        dense_raw_zero_matches=sum(r['dense_raw_cv_ADE']==0 for r in cases),
        histories_with_post_query_control=sum(r['past_rows_with_post_query_control']>0 for r in cases),
        all_future_samples_generated=all(r['future_generated_count']==12 for r in cases),
        strict_online_causality_established=False,main_protocol_changed=False,
        neighbor_comparisons=len(a['case_neighborhoods']),case_exact_alias_rows=sum(r['exact_rows'] for r in a['case_neighborhoods']),
        case_zero_events_in_512=sum(r['neighborhoods']['512']['zero_rows'] for r in a['case_neighborhoods']),
        control_distance_comparisons=percentiles,independent_confirmation=False,deployment_changed=False)
    text=json.dumps(summary,indent=2)+'\n'; path=PUBLIC/'interpretation.json'
    if path.exists(): assert json.loads(path.read_text())==summary
    else: path.write_text(text)
    lines=['# Moving Zero-CV Support: Complete Diagnostic Tables','',
        'Fresh distances and raw-annotation checks; frozen verified forecasts/forests. No new training or policy.',
        'Four exposed source sites, three seeds, obs8/pred12 stride12 annotation pixels.','',
        '## Seven Cases, Three Scoped Tracks','',
        '| Row | Site / track | Last step px | Past CV max residual px | Sampled future CV ADE px | Dense raw future CV ADE px | Past generated / 8 | Future generated / 12 | Past rows bracketed by later control |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in cases:
        lines.append(f"| {r['row']} | {r['scoped_track']} | {r['last_step_distance']:.3f} | {r['max_backcast_error']:.3f} | {r['raw_sampled_cv_ADE']:.3f} | {r['dense_raw_cv_ADE']:.4f} | {r['past_generated_count']} | {r['future_generated_count']} | {r['past_rows_with_post_query_control']} |")
    lines+=['','Dense raw-frame errors are a provenance diagnostic, not a replacement main metric.',
        'Generated flags and following controls are retrospective labels, not inference features.',
        'Five coupa rows overlap on one track; the three tracks are not three independent datasets.','',
        '## All Case/Feature/Action/Seed Comparisons','',
        '| View | Action | Features | Row | Closest RMS feature distance | Nearest zero-event rank | Source rank fraction | Exact matches | Zero events / 512 | Tracks / 512 | Mean relative gain / 512 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in a['case_neighborhoods']:
        n=r['neighborhoods']['512']
        lines.append(f"| {r['view']} | {r['action']} | {r['arm']} | {r['row']} | {r['nearest_distance']:.8g} | {r['nearest_zero_rank']} | {r['nearest_zero_rank_fraction']:.6f} | {r['exact_rows']} | {n['zero_rows']} | {n['tracks']} | {n['mean_relative_gain']:.6f} |")
    lines+=['','## Outcome-Blind Control Comparison','',
        '32 fixed metadata-hash controls per site, reused across seeds/actions. The 384 records per action/arm are not 384 independent sites or new tracks.',
        'At-least-as-distant fractions compare each case only with its 32 same-view controls; they are descriptive, not calibrated p-values or thresholds.','',
        '| Action | Features | Case fraction of controls at least as distant, min / median / max |',
        '|---|---|---:|']
    for action in ('damped_velocity_005','transformer','eqmotion'):
        for arm in ('history','full_risk'):
            v=[r['fraction_controls_at_least_as_distant'] for r in percentiles if r['action']==action and r['arm']==arm]
            lines.append(f"| {action} | {arm} | "+' / '.join(f'{q:.4f}' for q in np.quantile(v,[0,.5,1]))+' |')
    lines+=['','All control quantiles, neighborhood counts at 32/128/512, source counts and identities remain in analysis.json.',
        'Near contexts are not identical observations; these results do not prove Bayes irreducibility.',
        'No deployment, external readout, independent calibration or Stage5C/SMC execution.','']
    (PUBLIC/'tables.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    tracks=list(dict.fromkeys(r['scoped_track'] for r in cases))
    y=[np.mean([r['dense_raw_cv_ADE'] for r in cases if r['scoped_track']==t]) for t in tracks]
    axes[0].bar(range(3),y,color=['#286e97','#379879','#95628a'])
    axes[0].set_xticks(range(3),['coupa:87\n5 overlapping windows','hyang:93\n1 window','hyang:158\n1 window'])
    axes[0].set_ylabel('Dense raw-frame CV ADE (annotation pixels)')
    axes[0].set_title('Sampled-grid ADE is zero in all seven cases')
    for i,v in enumerate(y): axes[0].text(i,v+.02,f'{v:.4f}',ha='center')
    axes[0].set_ylim(0,1)
    labels=[]
    for i,(action,arm) in enumerate((a,b) for a in ('transformer','eqmotion','damped_velocity_005') for b in ('history','full_risk')):
        values=[r['nearest_zero_rank_fraction'] for r in rr if r['action']==action and r['arm']==arm and r['metadata_control']]
        axes[1].boxplot([values],positions=[i],widths=.45,showfliers=False,
            medianprops=dict(color='#333333'),whiskerprops=dict(color='#999999'),boxprops=dict(color='#999999'))
        for j,row in enumerate(cases):
            vals=[r['nearest_zero_rank_fraction'] for r in a['case_neighborhoods'] if r['action']==action and r['arm']==arm and r['row']==row['row']]
            axes[1].scatter(i+(j-3)*.045,np.median(vals),s=22,color='#a32c39',zorder=3)
        labels.append(action.replace('damped_velocity_005','damping')+'\n'+arm.replace('_',' '))
    axes[1].set_xticks(range(6),labels,rotation=25,ha='right',fontsize=9)
    axes[1].set_ylabel('Nearest zero-event rank / source bank size')
    axes[1].set_title('Case medians (red); metadata controls (boxes)')
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Moving zero-CV support: descriptive source evidence, no new model',fontsize=13)
    svg = PUBLIC/'support_diagnostic.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(ROOT/'data/stage_cvpr2027_experiments/moving_zero_support_v1/support_preview.png',dpi=140)
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='control_distance_comparisons'}))


if __name__=='__main__': main()
