"""Render all zero-reference controls without selecting a deployment winner."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/zero_atom_v1'


def figure(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    actions=('damped_velocity_005','transformer','eqmotion')
    labels=('Old strict','Original point','Original population','Original selected',
        'Guarded point','Guarded population','Guarded selected',
        'Matched point','Matched population','Matched selected')
    colors=['#5b6770']+['#267c9b']*3+['#269476']*3+['#ae698f']*3
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for col,action in enumerate(actions):
        rows=[a['summary'][action+'__'+p] for p in a['policies']]
        ade=[r['ADE']['equal_scene_gain_percent'] for r in rows]
        easy=[max(-v['gain_percent'] for s in r['seeds'].values()
            for v in s['subsets']['positive_easy']['by_scene'].values()) for r in rows]
        harms=[sum(s['zero_CV_harmed'] for s in r['seeds'].values()) for r in rows]
        for row,values in enumerate((ade,easy)):
            ax=axes[row,col]; ax.bar(range(10),values,color=colors)
            ax.axhline(0,color='#444444',linewidth=.7)
            ax.spines[['top','right']].set_visible(False)
            ax.set_xticks(range(10),labels,rotation=65,ha='right',fontsize=8)
            ax.set_ylabel(('Equal-site ADE gain over CV (%)','Worst site/seed positive-easy degradation (%)')[row])
            if row==0:
                ax.set_title(action.replace('_',' '))
                for i,n in enumerate(harms):
                    if n: ax.annotate(f'x{n}',(i,values[i]),xytext=(0,4),textcoords='offset points',ha='center',color='#ac2525',fontsize=8)
                ax.margins(y=.2)
            else: ax.axhline(2,color='#ac2525',linestyle='--',linewidth=1)
    fig.suptitle('Explicit zero-reference readout: full control matrix\n'
        'Four exposed SDD sites, three seeds; obs8/pred12. Red x = harmed zero-CV window/seed count.',fontsize=13)
    fig.savefig(PUBLIC/'risk_tradeoff.svg')
    private=ROOT/'data/stage_cvpr2027_experiments/zero_atom_v1'
    fig.savefig(private/'risk_tradeoff_preview.png',dpi=120)
    plt.close(fig)


def main():
    a=json.loads((PUBLIC/'analysis.json').read_text())
    lines=['# Explicit Zero-Reference Atom: All Controls','',
        'Four development-exposed SDD sites, three seeds, obs8/pred12 stride12 annotation pixels.',
        'Equal-physical-site gains over CV; not historical raw t50, strongest-baseline or independent-confirmation claims.','',
        '| Action | Policy | ADE gain % | FDE gain % | Hard gain % | Worst positive-easy degradation % | Zero-CV harms, window/seed | Mean switches |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for action in ('damped_velocity_005','transformer','eqmotion'):
        for p in a['policies']:
            r=a['summary'][action+'__'+p]
            easy=max(-x['gain_percent'] for s in r['seeds'].values() for x in s['subsets']['positive_easy']['by_scene'].values())
            harms=sum(s['zero_CV_harmed'] for s in r['seeds'].values())
            count=np.mean([s['selected'] for s in r['seeds'].values()])
            lines.append(f"| {action} | {p} | {r['ADE']['equal_scene_gain_percent']:.6f} | {r['FDE']['equal_scene_gain_percent']:.6f} | {r['subsets']['hard']['equal_scene_gain_percent']:.6f} | {easy:.6f} | {harms} | {count:.1f} |")
    lines+=['','All three guard/control counts match within every recording/frame/seed, not just globally.',
        'Numerically failed matched proposals retain their feasible incumbent and are not called optimal.',
        'Probability zero means no weighted positive event in any visited source leaf, not calibrated impossibility.',
        'Unknown and incomplete futures remain indexed; full-grid bounds are in analysis.json.','',
        '## All Registered Contrasts','','3000 paired resamples of four physical sites, nominal conditional development intervals.','']
    lines+=['| Contrast | All gain difference pp, 95% CI | Hard difference pp, 95% CI | Positive-easy difference pp, 95% CI |',
        '|---|---:|---:|---:|']
    for k,v in a['contrasts'].items():
        cells=[]
        for subset in ('all','hard','positive_easy'):
            r=v[subset]; lo,hi=r['ci95_pp']
            cells.append(f"{r['mean_gain_difference_pp']:.6f} [{lo:.6f}, {hi:.6f}]")
        lines.append('| '+k+' | '+' | '.join(cells)+' |')
    lines+=['','Positive-easy contrasts are differences in gain, so positive means less degradation.',
        'All 54 intervals are nominal, conditional development comparisons, not multiplicity-adjusted claims.',
        'Per-site values and complete machine-readable intervals remain in analysis.json.','']
    lines+=['## Solver','','```json',json.dumps(a['solver'],indent=2),'```','']
    (PUBLIC/'results.md').write_text('\n'.join(lines))
    lines=['# Fresh Leaf Readout Fits and Support','',
        '36 new source-only leaf-frequency fits on 36 cached_verified forest partitions. No new forest or neural forecast training.',
        'Weighted fitting Brier is an in-source fitting diagnostic, not independent calibration.','',
        '| View | Action | Effective unique rows | Zero-reference rows | Moving zero-reference rows | Zero-reference weighted draws | Fitting Brier | Seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in a['fits']:
        s=r['support']; lines.append(f"| {r['view']} | {r['action']} | {s['effective_unique']} | {s['zero_effective']} | {s['zero_effective_moving']} | {s['zero_draws']} | {s['training_weighted_brier']:.7f} | {r['seconds']:.3f} |")
    lines+=['','## Held-Source Event Diagnostic','','All held sites remain design-exposed. No probability threshold was selected from these numbers.','',
        '| View | Action | Complete labels | Zero-CV rows | Moving zero-CV | Moving zero-CV admitted | Complete-label Brier |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in a['held_event_quality']:
        lines.append(f"| {r['view']} | {r['action']} | {r['known']} | {r['zero']} | {r['moving_zero']} | {r['moving_zero_admitted']} | {r['brier_complete']:.7f} |")
    (PUBLIC/'fit_support.md').write_text('\n'.join(lines)+'\n')
    figure(a)
    print(json.dumps(dict(fits=len(a['fits']),rows=len(a['summary']))))


if __name__=='__main__': main()
