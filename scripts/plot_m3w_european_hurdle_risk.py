"""Plot all loss-decomposition contrasts, safety flags and matched moment losses."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.report_m3w_european_hurdle_risk import PUBLIC,verify


def save(fig,name):
    path=PUBLIC/(name+'.svg');fig.savefig(path)
    path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
    fig.savefig(PUBLIC/(name+'.png'),dpi=150);plt.close(fig)


def dot(ax,m,y,color,safe):
    p,interval=m['equal_scene_gain_percent'],m['scene_bootstrap_ci95']
    if interval is not None:ax.plot(interval,[y,y],color=color,linewidth=.8)
    if p is not None:ax.plot(p,y,'o',markersize=4,markeredgecolor=color,markerfacecolor=color if safe else 'white')


def main():
    a,_=verify();rows=[(f,s,e) for f in range(3) for s in (17,29,43) for e in ('all','easy')]
    colors=('#136f7c','#b44835')
    fig,axes=plt.subplots(1,2,figsize=(13,10),sharey=True,layout='constrained')
    for i,(fold,seed,event) in enumerate(rows):
        for candidate,color,offset in zip(('neural','damping097'),colors,(-.13,.13)):
            k=f'{candidate}_fold{fold}_seed{seed}_{event}'
            dot(axes[0],a['objective_pairs'][k]['all'],i+offset,color,a['views'][k+'_hurdle']['safety_observed_pass'])
        for arm,color,offset in zip(('product_mse','hurdle'),colors,(-.13,.13)):
            k=f'fold{fold}_seed{seed}_{event}_{arm}'
            dot(axes[1],a['neural_vs_damping'][k]['all'],i+offset,color,a['views']['neural_'+k]['safety_observed_pass'])
    axes[0].set_title('Hurdle vs matched product-MSE objective',fontsize=11)
    axes[1].set_title('Neural vs matched protected damping',fontsize=11)
    for ax,labels in zip(axes,(('Neural','Damping'),('Product MSE','Hurdle'))):
        ax.axvline(0,color='#666666',linewidth=.8);ax.grid(axis='x',color='#dddddd',linewidth=.5)
        ax.set_xlabel('Paired ADE improvement (%)');ax.spines[['top','right']].set_visible(False)
        ax.legend(handles=[Line2D([],[],color=c,marker='o',linestyle='none',label=l) for c,l in zip(colors,labels)],loc='lower center',bbox_to_anchor=(.5,-.09),ncols=2)
    axes[0].set_yticks(range(len(rows)),[f'F{f} / S{s} / {e}' for f,s,e in rows]);axes[0].invert_yaxis()
    fig.suptitle('Occurrence/severity supervision: all paired contrasts\nFilled: observed safety pass; hollow: fail. Conditional 3,000-locality-bootstrap CIs.',fontsize=12)
    save(fig,'objective_comparison')
    traces={}
    with (PUBLIC/'training_losses.csv').open() as f:
        for r in csv.DictReader(f):traces.setdefault(r['head'],[]).append(r)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,candidate in enumerate(('neural','damping097')):
        for col,event in enumerate(('all','easy')):
            ax=axes[row,col]
            for fold in range(3):
                for seed in (17,29,43):
                    for arm,color in zip(('product_mse','hurdle'),colors):
                        rs=traces[f'{candidate}_fold{fold}_seed{seed}_{event}_{arm}']
                        ax.plot([int(r['step']) for r in rs],[max(float(r['moment_mse']),1e-12) for r in rs],color=color,alpha=.45,linewidth=.8)
            ax.set_yscale('log');ax.set_title(candidate+' / '+event,fontsize=10)
            ax.set_xlabel('Optimizer update');ax.set_ylabel('Scaled native-moment batch MSE')
            ax.spines[['top','right']].set_visible(False)
    fig.legend(handles=[Line2D([],[],color=c,label=l) for c,l in zip(colors,('Product MSE','Hurdle'))],loc='outside lower center',ncols=2)
    fig.suptitle('All 72 training traces: matched moment-loss component\nHurdle additionally optimizes BCE and positive-only severity; training loss is not readout accuracy.',fontsize=12)
    save(fig,'training_moment_loss')
    print(json.dumps(dict(figures=2,all_objective_comparisons=36,all_new_neural_damping_comparisons=36)))


if __name__=='__main__':main()
