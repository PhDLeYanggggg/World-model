"""Fixed-roster descriptive accounting; no selection or new inference."""
import argparse
import json
from pathlib import Path
import statistics

from scripts import run_m3w_local_cost_analogue as run
from scripts.verify_m3w_selected_analogue import require


def aggregate(rows):
    output={}
    for arm in run.api.ARMS:
        values=[r['scores'][arm] for r in rows]
        out={k:sum(v[k] for v in values) for k in ('rows','known','unknown','supported','unsupported','benefit_harm_envelope_exceedances')}
        out['easy_exceeds_all']=sum(r['diagnostics']['arms'][arm]['easy_exceeds_all'] for r in rows)
        for target in ('all','easy'):
            part={k:sum(v[target][k] for v in values) for k in ('events','observed_harm','predicted_harm','false_safe_rows')}
            for metric in ('full_cohort_MSE','AUROC','AP'):
                nums=[v[target][metric] for v in values]
                part[metric]=statistics.fmean(nums) if None not in nums else None
                part['undefined_'+metric+'_views']=nums.count(None)
            for k in ('event_harm_SSE','non_event_harm_SSE','predicted_harm_on_events','predicted_harm_on_non_events'):
                part[k]=sum(r['diagnostics']['arms'][arm][target][k] for r in rows)
            out[target]=part
        output[arm]=out
    return output


def describe(rows):
    out=dict(all=aggregate(rows),posthoc_descriptive_only=True,whole_pipeline_OOF=False,
             held_inference=False,independent_roles_read=False,screen_changed=False,deployment_changed=False)
    fmt=lambda x:'undefined' if x is None else f'{x:.9g}'
    lines=['# Local Quadratic Cost: Fixed-Cohort Readout','','72 exposed-TRAIN views; repeated occurrences, not independent samples.',
        'Risk MSE is source-normalized and averaged equally across views, not ADE/FDE.',
        'Masses/SSE sum known normalized occurrences. Unknown queries remain in coverage.','',
        '| Arm | All risk MSE | Easy risk MSE | Predicted easy harm | Observed easy harm | B+H>E | EH>H |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,a in out['all'].items():
        lines.append(f"| {arm} | {fmt(a['all']['full_cohort_MSE'])} | {fmt(a['easy']['full_cohort_MSE'])} | {fmt(a['easy']['predicted_harm'])} | {fmt(a['easy']['observed_harm'])} | {a['benefit_harm_envelope_exceedances']} | {a['easy_exceeds_all']} |")
    lines+=['','## Fixed Source, Context and Seed Slices','',
            '| Slice | Views | Original easy MSE | Uniform nearest | Local signed |',
            '|---|---:|---:|---:|---:|']
    for key in ('source','context','seed'):
        out[key]={}
        for value in sorted({r[key] for r in rows}):
            g=[r for r in rows if r[key]==value];a=aggregate(g)
            out[key][str(value)]=dict(views=len(g),arms=a)
            lines.append(f"| {key}:{value} | {len(g)} | {fmt(a['origin']['easy']['full_cohort_MSE'])} | {fmt(a['selected_nearest32']['easy']['full_cohort_MSE'])} | {fmt(a['local_signed32']['easy']['full_cohort_MSE'])} |")
    effective=[r['diagnostics']['effective_donor_quantiles'] for r in rows if r['diagnostics']['effective_donor_quantiles'] is not None]
    leverage=[r['diagnostics']['max_weight_quantiles'] for r in rows if r['diagnostics']['max_weight_quantiles'] is not None]
    out['leverage']=dict(min_effective_donors=min(q[0] for q in effective) if effective else None,
        median_of_view_medians=statistics.median(q[2] for q in effective) if effective else None,
        max_single_donor_weight=max(q[4] for q in leverage) if leverage else None,
        not_a_cohort_quantile=True)
    out['donor_loss']={a:[sum(r['diagnostics']['donor_loss'][a][j] for r in rows) for j in range(2)]
                       for a in ('uniform','optimum')}
    lines+=['','## Leverage and Donor Fit','',
        f"Minimum effective donors: {fmt(out['leverage']['min_effective_donors'])}.",
        f"Median of view medians (not overall query median): {fmt(out['leverage']['median_of_view_medians'])}.",
        f"Largest single donor weight: {fmt(out['leverage']['max_single_donor_weight'])}.",
        'Lower donor loss is guaranteed by the constrained local fit, not evidence of prediction lift.',
        'Non-event harm mass is not a classification false-positive rate. Moment violations are not projected away.',
        'All negative slices retained; no posthoc primary substitution or threshold tuning.',
        'No independent CI, held readout or policy execution. obs8/pred12 rawstride12; pixel/local detector-silver.',
        'No metric/seconds/gold/physical-safety/true3D/foundation claim. Stage5C/SMC remain off.']
    return out,'\n'.join(lines)+'\n'


def execute(check):
    done=json.loads((run.PUBLIC/'complete.json').read_text());rows=[]
    for ref in done['groups']:
        p=run.ROOT/ref['path'];require(run.sha(p)==ref['sha256'],'Group changed')
        rows.append(json.loads(p.read_text()))
    data,md=describe(rows)
    data.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'),reporter_sha256=run.sha(Path(__file__)),
                result_source='cached_verified_descriptive_receipt_accounting')
    p,table=run.PUBLIC/'slice_diagnostics.json',run.PUBLIC/'slice_diagnostics.md'
    if check:
        require(data==json.loads(p.read_text()) and md==table.read_text(),'Readout replay differs')
        r=json.loads((run.PUBLIC/'figure_receipt.json').read_text())
        require(run.sha(run.ROOT/r['path'])==r['sha256'] and run.sha(p)==r['slice_sha256'],'Figure changed')
        print('All21 descriptive slices, diagnostics and figure hash verified');return
    run.once(p,data)
    if table.exists():require(table.read_text()==md,'Existing table changed')
    else:table.write_text(md)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    names=('Original','Uniform nearest','Weighted harm','Local signed','Global signed','Shuffled signed')
    colors=('#687580','#a95166','#5485aa','#007d78','#b47942','#847593')
    for ax,target,metric,title in zip(axes,('all','easy','easy'),
        ('full_cohort_MSE','full_cohort_MSE','predicted_harm'),
        ('All signed-risk MSE','Easy signed-risk MSE','Predicted easy-harm mass')):
        nums=[data['all'][a][target][metric] for a in run.api.ARMS]
        for i,v in enumerate(nums):
            if v is None:ax.text(i,0,'undefined',rotation=90)
            else:ax.bar(i,v,color=colors[i]);ax.text(i,v,f'{v:.3g}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(len(names)),names,rotation=30,ha='right');ax.set_title(title,fontsize=11)
        ax.set_ylim(0,max([v for v in nums if v is not None]+[1e-12])*1.3)
        if metric=='predicted_harm':
            obs=data['all']['origin']['easy']['observed_harm'];ax.axhline(obs,color='#222222',ls='--',label=f'observed {obs:.4g}')
            ax.legend(fontsize=8);ax.set_ylim(0,max(ax.get_ylim()[1],obs*1.2))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Fixed selected donors and neighbors, different local risk objectives',fontsize=14)
    fig.supxlabel('72 exposed-TRAIN fitting views; no independent test or policy deployment',fontsize=10)
    image=run.PRIVATE/'local_cost.png';fig.savefig(image,dpi=160);plt.close(fig)
    run.once(run.PUBLIC/'figure_receipt.json',dict(path=str(image.relative_to(run.ROOT)),sha256=run.sha(image),slice_sha256=run.sha(p),raw_data_in_figure=False))
    print(json.dumps(dict(slices=21,figure=str(image))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');execute(p.parse_args().check)
