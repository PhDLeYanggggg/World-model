"""Plot verified aggregate development evidence; keep binary figures out of Git."""
import json
from pathlib import Path

from scripts import report_m3w_scaled_risk as report


def main():
    _,complete,summary,_=report.validate()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names=['unit_control','quadratic','task_utility','original']
    labels=['Unscaled risk','Quadratic','Utility only','Original forest']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,mode,title in zip(axes[0],['full','matched'],['Full policy','Same intervention count']):
        for i,name in enumerate(names):
            r=summary['contrasts'][name+'_'+mode+'_paired_lower_percent']
            if r['CI95'] is None:
                ax.text(0,i,'Undefined support',fontsize=9)
            else:
                lo,hi=r['CI95'];m=r['mean']
                ax.errorbar(m,i,xerr=np.array([[m-lo],[hi-m]]),fmt='o',
                    color='#b03935' if hi<0 else '#17635b' if lo>0 else '#666666',capsize=3)
        ax.axvline(0,color='#666666',linewidth=.8,linestyle='--')
        ax.set_yticks(range(len(names)),labels);ax.invert_yaxis();ax.set_title(title,fontsize=12)
        ax.set_xlabel('Scaled minus control: paired lower-utility percentage points')
        ax.grid(axis='x',alpha=.2)
    policy=['original','quadratic','task_utility','unit_control','budget_scaled']
    plabel=['Original','Quadratic','Utility','Unscaled','Scaled']
    xpos=np.arange(len(policy));a=axes[1,0]
    a.bar(xpos-.17,[summary['safety'][n]['known_easy_violations'] for n in policy],.34,
          label='Known outcomes',color='#bb4742')
    a.bar(xpos+.17,[summary['safety'][n]['easy_upper_violations'] for n in policy],.34,
          label='Completion upper bound',color='#794880')
    a.set_xticks(xpos,plabel,rotation=15);a.set_ylim(0,76);a.set_ylabel('Views violating 2% selected easy-risk budget /72')
    a.set_title('Safety does not follow from utility',fontsize=12);a.legend(fontsize=9)
    b=axes[1,1]
    b.bar(xpos,[summary['safety'][n]['selected_occurrences'] for n in policy],color='#397a96')
    b.set_xticks(xpos,plabel,rotation=15);b.set_ylabel('Selected row occurrences (not independent samples)')
    b.set_title('Intervention volume',fontsize=12)
    fig.suptitle('Matched TRAIN Budget Scaling: Exposed Development Only',fontsize=15)
    fig.supxlabel('Nominal 95% paired-locality bootstrap, 3,000 draws; 12 localities. Risk is positive harm/reference cost, not net ADE.',fontsize=9)
    destination=report.run.manager.PRIVATE/'figures';destination.mkdir(parents=True,exist_ok=True)
    png=destination/'decision_evidence.png';pdf=destination/'decision_evidence.pdf'
    fig.savefig(png,dpi=180);fig.savefig(pdf,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    record=dict(summary_sha256=report.run.sha(report.HOME/'summary.json'),
        figure_data='cached_verified_aggregate_metrics',new_training=False,new_predictions=False,
        independent_confirmation=False,files={str(p.relative_to(report.run.ROOT)):report.run.sha(p) for p in (png,pdf)})
    (destination/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(png=str(png),pdf=str(pdf),advance=summary['advance_to_transfer_design'])))


if __name__=='__main__':main()
