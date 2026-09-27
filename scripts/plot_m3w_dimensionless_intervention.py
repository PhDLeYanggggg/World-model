"""Descriptive aggregate figure; no prediction or decision changes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_dimensionless_intervention_v1'


def main():
    data=json.loads((PUBLIC/'summary_metrics.json').read_text())
    risks=json.loads((PUBLIC/'risk_diagnosis.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    labels=['Neural raw','Neural protected','Damping protected']
    colors=['#575b61','#197e82','#ae522b']
    for ax,subset,title in zip(axes[:2],('all','positive_easy'),('All ADE gain vs CV','Easy ADE gain vs CV')):
        for i,(candidate,policy) in enumerate([('dimensionless','raw'),('dimensionless','point'),('damped','point')]):
            m=next(r['metrics']['gain_vs_CV_percent'] for r in data['summaries'] if r['scope']=='full' and
                r['candidate']==candidate and r['policy']==policy and r['endpoint']=='ADE' and r['subset']==subset)
            point=m['point'];lo,hi=m['ci95']
            ax.errorbar(point,i,xerr=np.array([[point-lo],[hi-point]]),fmt='o',color=colors[i],capsize=4)
        ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.axvline(0,color='#888888',linewidth=.8)
        ax.set_title(title);ax.set_xlabel('Percent gain (locality 95% interval)')
        ax.spines[['top','right']].set_visible(False)
    ax=axes[2]
    for candidate,color in [('dimensionless',colors[1]),('damped',colors[2])]:
        rows=[r for r in risks['views'] if r['candidate']==candidate and r['selected_actual_all_ratio'] is not None]
        x=[100*r['selected_predicted_all_ratio'] for r in rows]
        y=[100*r['selected_actual_all_ratio'] for r in rows]
        ax.scatter(x,y,s=14,alpha=.65,color=color,label='Neural' if candidate=='dimensionless' else 'Damping')
    ax.plot([0,2],[0,2],color='#888888',linewidth=.8,linestyle='--')
    ax.set_xlabel('Predicted selected harm ratio (%)');ax.set_ylabel('Observed selected harm ratio (%)')
    ax.set_title('Selected-set magnitude mismatch');ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Opened-source development: better forecasts do not yet yield better protected intervention',fontsize=12)
    fig.savefig(PUBLIC/'intervention_evidence.png',dpi=150,metadata={'Software':'M3W aggregate evidence'})
    plt.close(fig)


if __name__=='__main__':main()
