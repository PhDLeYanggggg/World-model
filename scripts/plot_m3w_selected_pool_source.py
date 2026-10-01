"""Export the completed source diagnostic without reading transfer outcomes."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_selected_pool_accounting as run


def load_source_rows():
    cfg, _ = run.registration()
    replay = json.loads((run.PUBLIC/'source_replay.json').read_text())
    assert replay['all72_exact_replay'] is True
    manifest = json.loads((run.PUBLIC/'source_manifest.json').read_text())
    assert manifest['source_heads'] == len(manifest['groups']) == 72
    rows = []
    for ref in manifest['groups']:
        path = ROOT/ref['path']
        assert path.parent == run.PRIVATE/'source'
        assert run.base.artifact(path) == ref
        rows.extend(json.loads(path.read_text())['rows'])
    return rows, cfg


def build(rows, cfg):
    selected = [r for r in rows if r['role']=='source_oof' and r['mode']=='joint']
    assert len(selected) == 72 and len({r['view'] for r in selected}) == 72
    points = []
    for r in selected:
        s = r['statistics']; c = s['easy_contrast']
        points.append(dict(view=r['view'], site=r['site'], head_seed=r['head_seed'],
            source_screen=r['source_screen'], raw_risk=s['raw']['easy']['observed_risk'],
            kept_risk=s['kept']['easy']['observed_risk'], predicted_kept_risk=s['kept']['easy']['predicted_risk'],
            harm_retained=c['harm_retained'], reference_retained=c['reference_retained'],
            benefit_retained=c['benefit_retained'], signed_bias_shift=c['signed_bias_shift'],
            new_known_violation=c['new_known_violation'], kept_known=s['kept']['known'],
            kept_unknown=s['kept']['unknown']))
    points.sort(key=lambda r:r['view'])
    interval = run.interval([(r['site'],r['signed_bias_shift']) for r in points],cfg)
    defined = [r for r in points if r['raw_risk'] is not None and r['kept_risk'] is not None]
    cases = [r for r in points if r['new_known_violation']]
    return dict(points=points, bias_interval=interval, risk_pairs_defined=len(defined),
        risk_pairs_undefined=len(points)-len(defined), case_views=[r['view'] for r in cases],
        case_localities=sorted({r['site'] for r in cases}), case_source_screen_pass=sum(r['source_screen'] for r in cases),
        result_source='fresh_figure_and_bootstrap_from_cached_verified_source_aggregates',
        transfer_outcomes_read=False, new_training=False, independent_confirmation=False,
        policy_changed=False, risk_budget=cfg['risk_budget'])


def render(e):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=e['points']; cases=[r for r in points if r['new_known_violation']]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
                         'svg.hashsalt':'m3w_selected_pool_source_v1'}):
        fig,ax=plt.subplots(1,3,figsize=(14,5.3),gridspec_kw={'width_ratios':[1,1,1.15]})
        for r in points:
            if r['raw_risk'] is None or r['kept_risk'] is None: continue
            bad=r['new_known_violation']
            ax[0].plot([0,1],[100*r['raw_risk'],100*r['kept_risk']],marker='o',markersize=3,
                color='#b63e55' if bad else '#287e83',alpha=.9 if bad else .3,lw=1.3 if bad else .8)
        ax[0].axhline(2,color='#454545',linestyle='--',lw=1)
        ax[0].text(.04,2.05,'Unchanged 2% budget',fontsize=8,color='#454545')
        ax[0].set(xticks=[0,1],xticklabels=['Raw eligible','Retained'],xlim=(-.15,1.2),
            ylim=(-.08,3.55),ylabel='Observed easy positive-harm / reference (%)')
        ax[0].set_title('A. All defined source OOF risk pairs',fontsize=10,pad=12)
        ax[0].text(.02,3.39,'30 defined views; 42 undefined',fontsize=8)
        ax[0].annotate('Two heads, one locality;\nboth fail source screening',xy=(1,3.112915),
            xytext=(-.03,2.57),fontsize=8,arrowprops=dict(arrowstyle='-',color='#b63e55'),color='#913348')
        colors=['#b63e55','#287e83','#777777']; fields=['harm_retained','reference_retained','benefit_retained']
        labels=['Positive harm','Reference error','Benefit']
        for j,(field,label,color) in enumerate(zip(fields,labels,colors)):
            x=[i+(j-1)*.24 for i in range(len(cases))]; y=[100*r[field] for r in cases]
            bars=ax[1].bar(x,y,width=.21,label=label,color=color)
            ax[1].bar_label(bars,labels=[f'{v:.2f}' for v in y],padding=3,fontsize=8)
        ax[1].set(xticks=list(range(len(cases))),xticklabels=['Head '+str(r['head_seed']) for r in cases],
            ylim=(0,52),ylabel='Retained fraction of raw-pool mass (%)')
        ax[1].set_title('B. Locality 074 counterexamples',fontsize=10,pad=12)
        ax[1].legend(frameon=False,fontsize=8,loc='upper center',ncol=1)
        d=e['bias_interval']['defined_only_descriptive']; pairs=sorted(d['by_locality'].items())
        labels=[s.rsplit('-',1)[1] for s,_ in pairs]+['Defined-only mean']
        for i,(site,v) in enumerate(pairs):
            ax[2].scatter(v,i,s=26,color='#b63e55' if site.endswith('074') else '#287e83')
        lo,hi=d['CI95']; mean=d['mean']; y=len(pairs)+.25
        ax[2].errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='D',color='#222222',capsize=3,markersize=4)
        ax[2].set_yticks(list(range(len(pairs)))+[y],labels)
        ax[2].invert_yaxis(); ax[2].axvline(0,color='#555555',lw=.8,ls='--')
        ax[2].set_title('C. Locality-level signed bias change',fontsize=10,pad=12)
        ax[2].set_xlabel('Retained minus raw bias / causal envelope')
        ax[2].text(.01,-.22,'3,000 locality draws; descriptive CI crosses zero',transform=ax[2].transAxes,fontsize=8)
        for a in ax:
            a.spines[['top','right']].set_visible(False); a.grid(axis='y' if a is not ax[2] else 'x',alpha=.13)
        fig.suptitle('Conservative selection can concentrate observed risk, but the average mechanism is unresolved',fontsize=12,y=.98)
        fig.subplots_adjust(left=.06,right=.99,bottom=.25,top=.83,wspace=.55)
        fig.text(.06,.075,'Source OOF only. Same two retained rows in the two counterexample heads; neither passes the existing source screen.',fontsize=9)
        fig.text(.06,.03,'42/72 bias contrasts are undefined. The full-view estimate is unavailable; repeated heads are not independent scenes. No deployment claim.',fontsize=9)
        svg, png = io.StringIO(), io.BytesIO()
        fig.savefig(svg,format='svg',metadata={'Date':None})
        fig.savefig(png,format='png',dpi=130)
        plt.close(fig)
        return svg.getvalue(),png.getvalue()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory-only',action='store_true',help='Return in-memory artifacts; write no local image or data files')
    a=p.parse_args()
    rows,cfg=load_source_rows()
    if not a.memory_only: run.guard(cfg)
    e=build(rows,cfg)
    e.update(source_manifest_sha256=run.digest(run.PUBLIC/'source_manifest.json'),
        source_replay_sha256=run.digest(run.PUBLIC/'source_replay.json'),script_sha256=run.digest(__file__))
    svg,png=render(e)
    if a.memory_only:
        print(json.dumps(dict(data=e,svg=svg,png_base64=base64.b64encode(png).decode(),local_artifacts_written=False)))
        return
    run.immutable(run.PUBLIC/'source_figure_data.json',e)
    (run.PUBLIC/'source_mechanism.svg').write_text(svg)
    (run.PRIVATE/'source_mechanism_preview.png').write_bytes(png)
    run.immutable(run.PUBLIC/'source_figure_receipt.json',dict(data_sha256=run.digest(run.PUBLIC/'source_figure_data.json'),
        svg_sha256=run.digest(run.PUBLIC/'source_mechanism.svg'),result_source=e['result_source'],
        transfer_outcomes_read=False,independent_confirmation=False,policy_changed=False))
    print(json.dumps({k:e[k] for k in ('risk_pairs_defined','risk_pairs_undefined','case_localities','case_source_screen_pass')}))


if __name__=='__main__': main()
