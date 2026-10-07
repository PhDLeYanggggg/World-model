"""Deterministic scientific figures from pinned public aggregates only."""
import argparse
import io
from pathlib import Path

from scripts import build_m3w_evidence_manuscript_v4 as source


def figures(e):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','svg.hashsalt':'m3w_evidence_v4',
                         'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
    outputs={};rows=e['temporal_comparisons']
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,metric,title,label in zip(axes,
        ('all_MSE','full_paired_lower_percent','matched_paired_lower_percent'),
        ('A. Global prediction error','B. Full paired utility','C. Same-count paired utility'),
        ('Normalized signed-score MSE\nNegative favours temporal',
         'Percentage points of reference cost\nPositive favours temporal',
         'Percentage points of reference cost\nPositive favours temporal')):
        for i,r in enumerate(rows):
            v=r[metric];color='#167868' if r['comparator'] in ('none','rowmean') else '#355d9a'
            ax.errorbar(v['estimate'],i,xerr=[[v['estimate']-v['ci_low']],[v['ci_high']-v['estimate']]],
                        color=color,fmt='o',capsize=3,markersize=5)
        ax.set_yticks(range(len(rows)),[r['comparator'] for r in rows]);ax.invert_yaxis()
        ax.axvline(0,color='#777777',lw=.8);ax.grid(axis='x',alpha=.15)
        ax.set_title(title,fontsize=11);ax.set_xlabel(label)
    fig.suptitle('Temporal minus comparator: 12 exposed localities, nominal 95% paired-bootstrap intervals',fontsize=12)
    outputs['temporal_prediction_vs_utility']=fig
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    rows=e['train_diagnostic'];labels=[r['arm'] for r in rows]
    for i,r in enumerate(rows):
        axes[0].scatter(i,r['median_known_risk_percent'],color='#b74750',s=40)
        axes[0].annotate(str(r['violations'])+'/72 violations',(i,r['median_known_risk_percent']),
                         xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        for offset,key,color,label in [(-.15,'median_all_harm_ratio','#355d9a','All known TRAIN'),
                                       (.15,'median_selected_harm_ratio','#b74750','Selected known TRAIN')]:
            axes[1].bar(i+offset,r[key],width=.27,color=color,label=label if i==0 else None)
    axes[0].axhline(2,color='#777777',ls='--',label='2% risk budget')
    axes[0].set_ylim(0,6.4);axes[0].set_xlim(-.4,2.4)
    axes[0].set_ylabel('Median positive easy-harm/reference (%)')
    axes[0].set_title('A. Risk fails on TRAIN',fontsize=11);axes[0].legend(fontsize=8,loc='lower right')
    axes[1].set_yscale('log');axes[1].axhline(1,color='#777777',ls='--');axes[1].set_ylim(.01,2.6)
    axes[1].set_ylabel('Median predicted / actual easy harm (log scale)')
    axes[1].set_title('B. Selection exposes underprediction',fontsize=11);axes[1].legend(fontsize=8)
    for ax in axes[:2]:ax.set_xticks(range(3),labels);ax.grid(axis='y',alpha=.15)
    c=rows[2]['components'];names=['Predicted\nslack','Harm\nunderprediction','Reference\nerror','Realized\nexcess']
    values=[c[k] for k in ('predicted_slack_mass','harm_underprediction_mass','reference_overprediction_budget_mass','realized_excess_mass')]
    axes[2].bar(range(4),values,color=['#167868','#b74750','#167868','#355d9a'])
    for i,v in enumerate(values):
        axes[2].annotate(f'{v:+.2f}',(i,v),xytext=(0,4 if v>0 else -12),textcoords='offset points',ha='center',fontsize=9)
    axes[2].set_xticks(range(4),names,fontsize=8);axes[2].axhline(0,color='#777777',lw=.8)
    axes[2].set_ylim(-2.7,10.5);axes[2].set_ylabel('Percentage points of selected reference cost')
    axes[2].set_title('C. Temporal: budget-excess accounting',fontsize=11)
    fig.suptitle('TRAIN resubstitution: descriptive summaries, not independent generalization',fontsize=12)
    outputs['selected_harm_train_diagnosis']=fig
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true')
    p.add_argument('--preview-dir',type=Path);a=p.parse_args()
    evidence=source.build(*source.load_sources());folder=source.ROOT/source.OUTPUT/'figures'
    folder.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt
    for name,fig in figures(evidence).items():
        buf=io.StringIO();fig.savefig(buf,format='svg',metadata={'Date':None})
        text='\n'.join(s.rstrip() for s in buf.getvalue().splitlines())+'\n';path=folder/(name+'.svg')
        if a.check:source.require(path.read_text()==text,'Changed scientific figure: '+name)
        else:path.write_text(text)
        if a.preview_dir:
            a.preview_dir.mkdir(parents=True,exist_ok=True);fig.savefig(a.preview_dir/(name+'.png'),dpi=140)
        plt.close(fig)
    print('Verified public aggregates; two figures; no new inference or bootstrap.')


if __name__=='__main__':
    main()
