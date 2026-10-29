"""Descriptive pooled-reference summaries; no new policy or screen selection."""
import argparse
import json
from pathlib import Path
import statistics

from scripts import run_m3w_pooled_analogue as run
from scripts import verify_m3w_pooled_analogue as verifier
from scripts import report_m3w_selected_analogue as previous


def aggregate(rows):
    old=previous.aggregate(verifier.compatible_rows(rows))
    return {new:old[name] for new,name in verifier.NAMES.items()}


def describe(rows,oldrows):
    current=aggregate(rows)
    supported=[r for r in rows if oldrows[r['context'],r['source'],r['seed']]['scores']['nearest32']['unsupported']==0]
    oldmean=statistics.fmean(oldrows[r['context'],r['source'],r['seed']]['scores']['nearest32']['easy']['full_cohort_MSE'] for r in supported) if supported else None
    out=dict(all=current,previous_complete_support_views=len(supported),
        previous_complete_support_old_nearest_easy_MSE=oldmean,
        previous_complete_support_current=aggregate(supported) if supported else None,
        whole_pipeline_OOF=False,posthoc_descriptive_only=True,held_inference=False,
        independent_roles_read=False,screen_changed=False,deployment_changed=False)
    lines=['# Pooled-Reference Fitting Slices','',
        'Source-normalized signed-risk MSE, equal-view averages. Repeated occurrences, not independent samples.',
        'No source/seed deletion or scientific-screen change. Previous-support subset is descriptive only.','',
        '| Slice | Views | Original easy MSE | Estimation easy MSE | Pooled mean | Pooled nearest32 | Pooled shuffled32 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in ('context','seed','source'):
        out[key]={}
        for value in sorted({r[key] for r in rows}):
            group=[r for r in rows if r[key]==value];a=aggregate(group)
            out[key][str(value)]=dict(views=len(group),arms=a)
            values=[a[k]['easy']['full_cohort_MSE'] for k in run.api.ARMS]
            lines.append(f'| {key}:{value} | {len(group)} | '+' | '.join('undefined' if v is None else f'{v:.9g}' for v in values)+' |')
    return out,'\n'.join(lines)+'\n'


def execute(check):
    done=json.loads((run.PUBLIC/'complete.json').read_text());rows=[];oldrows={}
    oldcomplete=json.loads((run.previous.PUBLIC/'complete.json').read_text())
    oldrefs={Path(r['path']).name:r for r in oldcomplete['groups']}
    for receipt in done['groups']:
        p=run.ROOT/receipt['path'];verifier.previous.require(run.sha(p)==receipt['sha256'],'Group changed')
        r=json.loads(p.read_text());rows.append(r)
        old=run.previous.PUBLIC/'groups'/p.name
        verifier.previous.require(run.sha(old)==oldrefs[p.name]['sha256'],'Source-local readout changed')
        oldrows[r['context'],r['source'],r['seed']]=json.loads(old.read_text())
    out,table=describe(rows,oldrows)
    out.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'),reporter_sha256=run.sha(Path(__file__)),
               result_source='cached_verified_fitting_slice_arithmetic')
    path=run.PUBLIC/'slice_diagnostics.json';md=run.PUBLIC/'slice_diagnostics.md'
    if check:
        verifier.previous.require(out==json.loads(path.read_text()) and table==md.read_text(),'Slice replay differs')
        receipt=json.loads((run.PUBLIC/'figure_receipt.json').read_text())
        verifier.previous.require(run.sha(run.ROOT/receipt['path'])==receipt['sha256'] and
                                  run.sha(path)==receipt['slice_sha256'],'Figure binding changed')
        print('21 descriptive slices and local figure hash verified; no new inference');return
    run.once(path,out)
    if md.exists():verifier.previous.require(md.read_text()==table,'Table changed')
    else:md.write_text(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    labels=['Original','Estimation','Pooled mean','Nearest32','Shuffled32']
    colors=['#66717b','#b76833','#84715c','#007d78','#a64f64']
    for ax,target,metric,title in zip(axes,('all','easy','easy'),
            ('full_cohort_MSE','full_cohort_MSE','predicted_harm'),
            ('All signed-risk MSE','Easy signed-risk MSE','Predicted easy-harm mass')):
        vals=[out['all'][a][target][metric] for a in run.api.ARMS]
        for i,v in enumerate(vals):
            if v is not None:
                ax.bar(i,v,color=colors[i]);ax.text(i,v,f'{v:.4g}',ha='center',va='bottom',fontsize=8)
            else:ax.text(i,0,'undefined',ha='center',va='bottom',fontsize=8)
        ax.set_xticks(range(5),labels,rotation=22,ha='right');ax.set_title(title,fontsize=11)
        ax.set_ylim(0,max([v for v in vals if v is not None]+[1e-12])*1.25)
        if metric=='predicted_harm':
            obs=out['all']['pooled_nearest32']['easy']['observed_harm'];ax.axhline(obs,color='#222222',ls='--',lw=1,label=f'observed {obs:.4g}')
            ax.set_ylim(0,max(ax.get_ylim()[1],obs*1.2));ax.legend(fontsize=8)
            ax.set_ylabel('Sum after source normalization')
        else:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.set_ylabel('Equal-view normalized MSE')
        ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Pooled reference support does not by itself establish calibrated risk',fontsize=14)
    fig.supxlabel('72 frozen fitting views; exposed TRAIN only, no policy or independent test',fontsize=10)
    p=run.PRIVATE/'pooled_analogue.png';fig.savefig(p,dpi=160);plt.close(fig)
    run.once(run.PUBLIC/'figure_receipt.json',dict(path=str(p.relative_to(run.ROOT)),sha256=run.sha(p),
        slice_sha256=run.sha(path),reporter_sha256=run.sha(Path(__file__)),raw_data_in_figure=False))
    print(json.dumps(dict(slices=21,figure=str(p))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');execute(p.parse_args().check)
