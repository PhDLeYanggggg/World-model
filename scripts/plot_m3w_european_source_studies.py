"""Scientific figures from frozen source readouts, not scene imagery."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_source_forecast_v1'


def main():
    r = json.loads((PUBLIC/'analysis.json').read_text())
    if not (PUBLIC/'checkpoint_replay.json').exists() or not (PUBLIC/'verification.json').exists():
        raise ValueError('Verified completed experiment required before plotting')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.unicode_minus':False,'svg.hashsalt':'m3w-european-source-v1'})
    figures = PUBLIC/'figures'
    figures.mkdir(exist_ok=True)
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    seeds = list(r['seeds'])
    labels = [f'Seed {seed}' for seed in seeds]+['Mean seed errors']
    metrics = [r['seeds'][seed]['ADE_vs_training_selected_strongest'] for seed in seeds]
    metrics.append(r['mean_seed_ADE_vs_training_selected_strongest'])
    for i,m in enumerate(metrics):
        v = m['equal_scene_gain_percent']; lo,hi = m['scene_bootstrap_ci95']
        axes[0].plot([lo,hi],[i,i],color='#21618c',linewidth=2)
        axes[0].scatter(v,i,color='#21618c' if i<3 else '#a04000',s=45,zorder=3)
    axes[0].set_yticks(range(len(labels)),labels)
    axes[0].invert_yaxis(); axes[0].axvline(0,color='black',linewidth=.8)
    axes[0].set_xlabel('ADE improvement vs training-selected baseline (%)')
    axes[0].set_title('Paired locality bootstrap, 3,000 resamples')
    sites = r['mean_seed_ADE_vs_training_selected_strongest']['by_scene']
    names = list(sites); gains = [sites[k]['gain_percent'] for k in names]
    axes[1].barh(names,gains,color=['#117864' if v>=0 else '#a04000' for v in gains])
    axes[1].invert_yaxis(); axes[1].axvline(0,color='black',linewidth=.8)
    axes[1].set_xlabel('Mean-seed ADE improvement (%)')
    axes[1].set_title('Per-locality heterogeneity')
    fig.suptitle('European Squares: source-only development, not confirmation',fontsize=13)
    for ext in ('png','svg'):
        fig.savefig(figures/('forecast_comparison.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    colors = {17:'#21618c',29:'#a04000',43:'#117864'}
    for trial in r['training']:
        ti,f = trial['identity'],trial['fit']
        ax = axes[0 if ti['kind']=='single' else 1,ti['fold']]
        steps = np.array([x['step'] for x in f['losses']])
        values = np.array([x['loss'] for x in f['losses']])
        smoothed = np.convolve(values,np.ones(5)/5,mode='valid')
        ax.plot(steps[4:],smoothed,color=colors[ti['seed']],label=f"Seed {ti['seed']}",linewidth=1.2)
        ax.set_title(f"{ti['kind']} group {ti['fold']}")
        ax.set_xlabel('Optimizer update'); ax.set_ylabel('Fitting loss, five logged batches')
        ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Actual training losses; no validation-based checkpoint selection',fontsize=13)
    for ext in ('png','svg'):
        fig.savefig(figures/('training_losses.'+ext),dpi=160,metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    for name in ('forecast_comparison.svg','training_losses.svg'):
        path = figures/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print(str(figures))


if __name__=='__main__':
    main()
