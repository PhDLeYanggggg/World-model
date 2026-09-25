"""Plot all registered neural views without selecting a winning policy."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.report_m3w_european_cv_reference import require_verification

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_symmetric_utility_v1'


def main():
    r=json.loads((PUBLIC/'analysis.json').read_text())
    require_verification(PUBLIC,r)
    if not json.loads((PUBLIC/'accounting_audit.json').read_text())['all_passed']:
        raise ValueError('Accounting must pass before figures')
    rows=[(event,arm,guard) for event in ('all','easy') for arm in ('ridge','neural_underharm4')
          for guard in ('no_guard','source_zero_guard')]
    fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True,layout='constrained')
    colors=['#126a7c','#b34f32','#7a4386']
    for i,seed in enumerate((17,29,43)):
        ys=np.arange(len(rows))+(i-1)*.2
        for y,(event,arm,guard) in zip(ys,rows):
            name=f'{seed}_neural_{event}_{arm}_{guard}'
            direct=f'{seed}_{event}_{arm}_{guard}'
            for ax,m in zip(axes[:2],[r['symmetric_vs_asymmetric'][name]['pointwise'],
                                     r['neural_vs_damping'][direct]['pointwise']]):
                v=m['equal_scene_gain_percent']; interval=m['scene_bootstrap_ci95']
                if v is not None:
                    ax.plot(v,y,'o',color=colors[i],markersize=4)
                    if interval is not None:
                        ax.plot(interval,[y,y],color=colors[i],linewidth=1.2)
            m=r['policies'][name]['full']
            degradation=-m['positive_easy_ADE_vs_CV']['worst_scene_gain_percent']
            axes[2].plot(degradation,y,'x' if m['zero_CV']['harmed_rows'] else 'o',
                color=colors[i],markersize=5,label=f'Seed {seed}' if y==ys[0] else None)
    axes[0].set_yticks(np.arange(len(rows)),[
        f"{e} / {'linear risk' if a=='ridge' else 'neural risk'} / {'guarded' if g!='no_guard' else 'unguarded'}"
        for e,a,g in rows])
    axes[0].invert_yaxis()
    for ax,title in zip(axes,['Neural: MSE utility vs old utility','Neural vs protected damping (both MSE)',
                              'Neural: worst-locality easy degradation']):
        ax.set_title(title,fontsize=10,pad=12)
        ax.axvline(0,color='#777777',linewidth=.8)
        ax.grid(axis='x',color='#dddddd',linewidth=.5)
        ax.set_xlabel('Percent',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    axes[2].axvline(2,color='#b34f32',linestyle='--',linewidth=1,label='2% limit')
    axes[2].legend(fontsize=8,loc='lower right')
    fig.suptitle('Symmetric utility, frozen risk: all eight neural configurations and three seeds\n'
        '3,000 conditional source-locality bootstrap resamples; x marker means a zero-CV case was harmed',fontsize=12)
    fig.savefig(PUBLIC/'utility_ablation.svg')
    fig.savefig(PUBLIC/'utility_ablation.png',dpi=160)
    plt.close(fig)
    print('Rendered all registered neural configurations; no selected winner.')


if __name__=='__main__':
    main()
