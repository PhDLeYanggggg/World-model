"""Render the completed width control from hash-bound public scalar receipts."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/publication_readiness_2026_09'
PUBLIC=BASE/'european_selection_capacity_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_selection_capacity_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    read=lambda p:json.loads(p.read_text())
    summary=read(PUBLIC/'summary.json');verified=read(PUBLIC/'verification.json')
    if not verified['all54_metric_groups_exact'] or verified['summary_sha256']!=sha(PUBLIC/'summary.json'):
        raise ValueError('Completed exact development readout required')
    train=read(PUBLIC/'training_freeze.json')
    old_path=BASE/'european_selection_cost_forest_v1/decoded_SOURCE_objective.json'
    old={r['trial']:r['score']['total'] for r in read(old_path)['rows'] if r['arm']=='mean_absolute'}
    fresh={}
    for ref in train['fits']:
        path=ROOT/ref['path']
        if sha(path)!=ref['sha256']:raise ValueError('Complete SOURCE fit receipt differs')
        r=read(path);fresh[r['trial']]=r['SOURCE_objective']['total']
    if set(old)!=set(fresh) or len(old)!=9:raise ValueError('The same nine paired producers are required')
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','svg.fonttype':'none',
        'svg.hashsalt':'m3w_selection_capacity_v1','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.2,1]})
    a,b,c=axes
    for trial in sorted(old):a.plot([0,1],[old[trial],fresh[trial]],color='#a9a9a9',linewidth=.8,alpha=.7)
    means=[float(np.mean(list(d.values()))) for d in (old,fresh)]
    a.plot([0,1],means,color='#167d8d',marker='o',linewidth=2,label='Nine-producer mean')
    a.set(xticks=[0,1],xticklabels=['Width 64','Width 128'],ylabel='Complete SOURCE objective',
        title='A  Fitting improvement only',xlim=(-.25,1.25))
    for x,y in enumerate(means):a.annotate(f'{y:.6f}',(x,y),xytext=(0,12),textcoords='offset points',ha='center')
    a.grid(axis='y',alpha=.2)
    effect=summary['effects']['completion_lower_pp'];lo,hi=effect['CI95'];point=effect['point']
    b.axvline(0,color='#777777',linestyle=':',linewidth=1)
    b.errorbar([point],[0],xerr=[[point-lo],[hi-point]],fmt='o',color='#c34949',capsize=4)
    b.set(yticks=[],xlabel='Paired completion-lower effect (pp)',title='B  Registered primary fails',
        xlim=(min(lo-.15,-1.3),.15),ylim=(-.5,.5))
    b.text(.5,.22,f'{point:+.3f} [{lo:+.3f}, {hi:+.3f}]',transform=b.transAxes,ha='center',fontsize=10)
    b.text(.5,.06,'3,000 locality bootstrap resamples',transform=b.transAxes,ha='center',fontsize=8)
    counts=[summary['safety'][k]['supported_views'] for k in ('width64','width128')]
    c.bar([0,1],counts,color=['#676f78','#d79a3f'],width=.55)
    c.set(xticks=[0,1],xticklabels=['Width 64','Width 128'],ylim=(0,252),
        ylabel='Supported / 252 required views',title='C  Complete risk still fails')
    for x,n in enumerate(counts):c.text(x,n+7,f'{n}/252',ha='center')
    c.grid(axis='y',alpha=.2)
    fig.suptitle('A small SOURCE loss reduction does not establish safe routing',fontsize=12)
    folder=PUBLIC/'figures';folder.mkdir(parents=True,exist_ok=True)
    svg=folder/'width_control.svg';png=PRIVATE/'width_control.png'
    fig.savefig(svg,metadata={'Date':None});fig.savefig(png,dpi=150);plt.close(fig)
    receipt=dict(result_source='cached_verified_public_scalars_fresh_figure_not_new_inference',
        summary_sha256=sha(PUBLIC/'summary.json'),verification_sha256=sha(PUBLIC/'verification.json'),
        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),old_SOURCE_objective_sha256=sha(old_path),
        renderer_sha256=sha(Path(__file__)),svg_sha256=sha(svg),private_png_sha256=sha(png),
        original_sources_unchanged=True,visually_checked=False,independent_confirmation=False,
        deployment_changed=False,stage5c=False,smc=False)
    (folder/'sources.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
