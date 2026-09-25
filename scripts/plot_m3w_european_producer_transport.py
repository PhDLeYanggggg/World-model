"""Display all paired producer replacements, not a selected result."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.report_m3w_european_producer_transport import sha

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_producer_transport_v1'


def main():
    a=json.loads((PUBLIC/'analysis.json').read_text())
    v=json.loads((PUBLIC/'verification.json').read_text())
    if not v['all_passed'] or v['analysis_sha256']!=sha(PUBLIC/'analysis.json'):
        raise ValueError('Complete verified readout required')
    fig,axes=plt.subplots(1,3,figsize=(15,8),sharey=True,layout='constrained')
    colors=['#136f7c','#b44835']
    rows=[(f,s) for f in range(3) for s in (17,29,43)]
    for i,(fold,seed) in enumerate(rows):
        for half in (0,1):
            for j,event in enumerate(('all','easy')):
                key=f'fold{fold}_seed{seed}_half{half}_{event}'
                pair=a['small_vs_full'][key]
                y=i+(half*2+j-1.5)*.14
                for ax,subset in zip(axes,('all','hard','easy')):
                    m=pair['policy_small_vs_full'][subset]
                    point,interval=m['equal_scene_gain_percent'],m['scene_bootstrap_ci95']
                    if point is not None:
                        ax.plot(point,y,'o' if event=='all' else 's',color=colors[half],markersize=4)
                    if interval is not None:
                        ax.plot(interval,[y,y],color=colors[half],linewidth=.8)
    axes[0].set_yticks(range(len(rows)),[f'Fitting fold {f} / seed {s}' for f,s in rows])
    axes[0].invert_yaxis()
    for ax,title in zip(axes,('All supported targets','Hard targets','Positive-easy targets')):
        ax.set_title(title,fontsize=11)
        ax.axvline(0,color='#666666',linewidth=.8)
        ax.set_xlabel('ADE gain over full producer (%)')
        ax.grid(axis='x',color='#dddddd',linewidth=.5)
        ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([],[],color=c,marker='o',linestyle='none',label=f'Half {i}') for i,c in enumerate(colors)]
    handles += [Line2D([],[],color='#555555',marker=m,linestyle='none',label=e+'-event gate') for m,e in [('o','All'),('s','Easy')]]
    fig.legend(handles=handles,loc='outside lower center',ncols=4)
    fig.suptitle('Frozen-head producer replacement: all 36 comparisons\nSame excluded sources; 3,000 conditional locality-bootstrap resamples',fontsize=13)
    path=PUBLIC/'producer_comparison.svg'
    fig.savefig(path)
    path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n')
    fig.savefig(PUBLIC/'producer_comparison.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
