"""All registered parameterization contrasts and matched training loss traces."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.report_m3w_european_geometric_cost import sha

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_geometric_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_geometric_cost_v1'


def save(fig,name):
    path=PUBLIC/(name+'.svg');fig.savefig(path)
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(PUBLIC/(name+'.png'),dpi=150);plt.close(fig)


def main():
    a=json.loads((PUBLIC/'analysis.json').read_text())
    v=json.loads((PUBLIC/'verification.json').read_text())
    if not v['all_passed'] or v['analysis_sha256']!=sha(PUBLIC/'analysis.json'):
        raise ValueError('Verified complete readout required')
    fig,axes=plt.subplots(1,3,figsize=(15,11),sharey=True,layout='constrained')
    rows=[(f,s,e) for f in range(3) for s in (17,29,43) for e in ('all','easy')]
    colors={'neural':'#136f7c','damping097':'#b44835'}
    for ax,arm,title in zip(axes,('utility_only','risk_only','both'),('Utility head only','Risk head only','Both heads')):
        for i,(fold,seed,event) in enumerate(rows):
            for candidate,offset in [('neural',-.13),('damping097',.13)]:
                key=f'{candidate}_fold{fold}_seed{seed}_{event}_{arm}'
                m=a['replacements'][key]['all'];y=i+offset;color=colors[candidate]
                p,interval=m['equal_scene_gain_percent'],m['scene_bootstrap_ci95']
                if interval is not None:ax.plot(interval,[y,y],color=color,linewidth=.8)
                if p is not None:ax.plot(p,y,'o',markersize=4,markeredgecolor=color,
                    markerfacecolor=color if a['views'][key]['safety_observed_pass'] else 'white')
        ax.set_title(title,fontsize=11);ax.axvline(0,color='#666666',linewidth=.8)
        ax.set_xlabel('ADE gain over old head pair (%)');ax.grid(axis='x',color='#dddddd',linewidth=.5)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(rows)),[f'F{f} / S{s} / {e}' for f,s,e in rows]);axes[0].invert_yaxis()
    handles=[Line2D([],[],color=c,marker='o',linestyle='none',label=n) for n,c in colors.items()]
    handles += [Line2D([],[],color='#444444',marker='o',markerfacecolor=fill,linestyle='none',label=label)
        for fill,label in [('#444444','Observed safety passes'),('white','Observed safety fails')]]
    fig.legend(handles=handles,loc='outside lower center',ncols=4)
    fig.suptitle('Causal-envelope cost heads: all 108 matched replacement contrasts\nEight excluded development localities per fit; 3,000 conditional bootstrap draws',fontsize=12)
    save(fig,'head_ablation')
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for row,candidate in enumerate(('neural','damping097')):
        for col,task in enumerate(('utility','all','easy')):
            ax=axes[row,col]
            for fold in range(3):
                for seed in (17,29,43):
                    path=PRIVATE/'heads'/f'{candidate}_fold{fold}_seed{seed}_{task}'/'complete.json'
                    r=json.loads(path.read_text())
                    parent=ROOT/r['identity']['original_checkpoint']['path']
                    old=json.loads((parent.parent/'complete.json').read_text())
                    for record,color in [(old,'#999999'),(r,colors[candidate])]:
                        trace=record['fit']['trace']
                        ax.plot([x['step'] for x in trace],[max(x['loss'],1e-12) for x in trace],
                            color=color,alpha=.4,linewidth=.7)
            ax.set_yscale('log');ax.set_title(candidate+' / '+task,fontsize=10)
            ax.set_xlabel('Optimizer update');ax.set_ylabel('Native-cost scaled batch MSE')
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('All 54 matched training traces: gray original, color envelope\nBatch losses use different task targets; they are not held-out performance',fontsize=12)
    save(fig,'training_loss')


if __name__=='__main__':
    main()
