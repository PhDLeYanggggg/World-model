"""Fixed-roster support diagnostics; no selection or new inference."""
import argparse
import json
from pathlib import Path
import statistics
from scripts import run_m3w_coupled_support as run
from scripts.verify_m3w_selected_analogue import require


def aggregate(rows):
    output={}
    for a in run.api.ARMS:
        s=[r['scores'][a] for r in rows]
        result={k:sum(v[k] for v in s) for k in ('rows','known','unknown','supported','unsupported','benefit_harm_envelope_exceedances')}
        result['easy_exceeds_all']=sum(r['diagnostics']['arms'][a]['easy_exceeds_all'] for r in rows)
        for t in ('all','easy'):
            d={k:sum(v[t][k] for v in s) for k in ('events','predicted_harm','observed_harm','false_safe_rows')}
            for k in ('full_cohort_MSE','AUROC','AP'):
                values=[v[t][k] for v in s];d[k]=statistics.fmean(values) if None not in values else None
                d['undefined_'+k+'_views']=values.count(None)
            for k in ('event_harm_SSE','non_event_harm_SSE','predicted_harm_on_events','predicted_harm_on_non_events'):
                d[k]=sum(r['diagnostics']['arms'][a][t][k] for r in rows)
            result[t]=d
        output[a]=result
    return output


def describe(rows):
    data={'all':aggregate(rows)};fmt=lambda v:'undefined' if v is None else f'{v:.9g}'
    lines=['# Coupled Risk and Donor Support: Fixed-Roster Readout','','## Material Passport','',
        '72 exposed-TRAIN fitting views. Repeated queries, not independent observations.',
        'Equal-view normalized signed-risk MSE, not trajectory ADE/FDE. Query unknownness retained.',
        'No whole-pipeline OOF, held/independent role opening, policy change or safety certification.','',
        '| Arm | All MSE | Easy MSE | Predicted easy H | Observed easy H | EH>H | B+H>E |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a,s in data['all'].items():
        lines.append(f"| {a} | {fmt(s['all']['full_cohort_MSE'])} | {fmt(s['easy']['full_cohort_MSE'])} | {fmt(s['easy']['predicted_harm'])} | {fmt(s['easy']['observed_harm'])} | {s['easy_exceeds_all']} | {s['benefit_harm_envelope_exceedances']} |")
    data['support']={k:sum(r['diagnostics'][k] for r in rows) for k in
        ('local_supported','local_unsupported','nested_supported','alpha_zero','alpha_one','alpha_sum','nested_loss_before','nested_loss_after')}
    for name in ('overlap','row_leverage_overlap'):
        data[name]={k:sum(r['diagnostics'][name][k] for r in rows) for k in rows[0]['diagnostics'][name]}
    data['concentration']={}
    for k in rows[0]['diagnostics']['quantiles']:
        q=[r['diagnostics']['quantiles'][k] for r in rows if r['diagnostics']['quantiles'][k] is not None]
        data['concentration'][k]=dict(minimum=min(v[0] for v in q),median_of_view_medians=statistics.median(v[2] for v in q),
            maximum=max(v[-1] for v in q),not_pooled_quantiles=True)
    lines+=['','## Support and Blend','', 'ESS measures weight concentration, not independent sample count.',
        'Nested donor loss improvement is guaranteed by fitting alpha, not query generalization.','',
        '```json',json.dumps(data['support'],indent=2),'```','',
        'Old inconsistency versus low recording ESS:','```json',json.dumps(data['overlap'],indent=2),'```','',
        'Old inconsistency versus low row ESS:','```json',json.dumps(data['row_leverage_overlap'],indent=2),'```']
    lines+=['','## Prespecified Support Strata','',
        '| Stratum | Rows | Known | Original easy SSE/known | Coupled | Nested blend |','|---|---:|---:|---:|---:|---:|']
    data['strata']={}
    for name in rows[0]['diagnostics']['slices']:
        parts=[r['diagnostics']['slices'][name] for r in rows]
        s={k:sum(p[k] for p in parts) for k in ('rows','known','unknown','envelope_over_scale_sum','original_easy_fraction_sum')}
        s['signed_MSE']={a:{t:sum(p['signed_SSE'][a][t] for p in parts)/s['known'] if s['known'] else None
            for t in ('all','easy')} for a in run.api.ARMS}
        data['strata'][name]=s
        lines.append(f"| {name} | {s['rows']} | {s['known']} | {fmt(s['signed_MSE']['origin']['easy'])} | {fmt(s['signed_MSE']['coupled_cost']['easy'])} | {fmt(s['signed_MSE'][run.api.PRIMARY]['easy'])} |")
    lines+=['','Stratum MSE pools normalized SSE/known occurrences; it is NOT the equal-view primary metric.',
        'Zero-alpha and one-record strata do not establish calibrated safety.','',
        '## All Source, Context and Head-Seed Slices','',
        '| Slice | Views | Original easy MSE | Independent | Coupled | Nested blend |',
        '|---|---:|---:|---:|---:|---:|']
    for key in ('source','context','seed'):
        data[key]={}
        for value in sorted({r[key] for r in rows}):
            g=[r for r in rows if r[key]==value];a=aggregate(g);data[key][str(value)]=dict(views=len(g),arms=a)
            lines.append(f"| {key}:{value} | {len(g)} | "+' | '.join(fmt(a[n]['easy']['full_cohort_MSE']) for n in
                ('origin','independent_cost','coupled_cost',run.api.PRIMARY))+' |')
    lines+=['','No posthoc primary substitution or threshold search. No confirmatory CI/p-value from fitting repeats.',
        'Head seeds are not new forecaster seeds. obs8/pred12 rawstride12, pixel/local detector-silver.',
        'No metric/seconds/human-gold/physical-safety/true3D/foundation claim. Stage5C/SMC remain off.']
    return data,'\n'.join(lines)+'\n'


def execute(check=False):
    done=json.loads((run.PUBLIC/'complete.json').read_text());rows=[]
    for ref in done['groups']:
        p=run.ROOT/ref['path'];require(run.sha(p)==ref['sha256'],'Group changed');rows.append(json.loads(p.read_text()))
    data,md=describe(rows)
    data.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'),reporter_sha256=run.sha(Path(__file__)),
        result_source='cached_verified_descriptive_arithmetic',new_inference=False,independent_replication=False)
    p,table=run.PUBLIC/'slice_diagnostics.json',run.PUBLIC/'slice_diagnostics.md'
    if check:
        require(data==json.loads(p.read_text()) and md==table.read_text(),'Readout replay differs')
        receipt=json.loads((run.PUBLIC/'figure_receipt.json').read_text())
        require(run.sha(run.ROOT/receipt['path'])==receipt['sha256'] and run.sha(p)==receipt['slice_sha256'],'Figure changed')
        print('21 slices,6 support strata,overlap and figure receipt checked');return
    run.once(p,data)
    if table.exists():require(table.read_text()==md,'Table changed')
    else:table.write_text(md)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    names=('Original','Independent','Coupled','Mean alpha','Shuffle alpha','Nested alpha')
    colors=('#657582','#a95768','#45879e','#b48c47','#847394','#008577')
    for ax,t,title in zip(axes[:2],('all','easy'),('All signed-risk MSE','Easy signed-risk MSE')):
        vals=[data['all'][a][t]['full_cohort_MSE'] for a in run.api.ARMS]
        for i,v in enumerate(vals):
            if v is not None:ax.bar(i,v,color=colors[i]);ax.text(i,v,f'{v:.4g}',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(6),names,rotation=30,ha='right');ax.set_title(title,fontsize=11)
        ax.set_ylim(0,max(v for v in vals if v is not None)*1.25)
    s=data['support'];labels=('alpha = 0','0 < alpha < 1','alpha = 1')
    vals=(s['alpha_zero'],s['local_supported']+s['local_unsupported']-s['alpha_zero']-s['alpha_one'],s['alpha_one'])
    axes[2].bar(range(3),vals,color=('#657582','#45879e','#008577'))
    for i,v in enumerate(vals):axes[2].text(i,v,str(v),ha='center',va='bottom')
    axes[2].set_xticks(range(3),labels,rotation=25,ha='right');axes[2].set_title('Fitted blend occurrences',fontsize=11)
    axes[2].set_ylim(0,max(vals)*1.25)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Joint constraints and donor-record-supported blending',fontsize=14)
    fig.supxlabel('Exposed TRAIN fitting only; no independent test or policy deployment',fontsize=10)
    image=run.PRIVATE/'coupled_support.png';fig.savefig(image,dpi=160);plt.close(fig)
    run.once(run.PUBLIC/'figure_receipt.json',dict(path=str(image.relative_to(run.ROOT)),sha256=run.sha(image),slice_sha256=run.sha(p)))
    print(json.dumps(dict(slices=21,strata=6,figure=str(image))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');execute(p.parse_args().check)
