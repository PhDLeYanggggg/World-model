"""Synthesize both fixed SOURCE populations only after complete exact replay."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.m3w_european_eqmotion import interval

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_eqmotion_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_eqmotion_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    read=lambda p:json.loads(p.read_text())
    primary=read(PUBLIC/'summary.json');secondary=read(PUBLIC/'strong_controls/summary.json')
    a=read(PUBLIC/'verification.json');b=read(PUBLIC/'strong_controls/verification.json')
    if (not a['all72_SOURCE_groups_exact'] or a['summary_sha256']!=sha(PUBLIC/'summary.json')
            or not b['all144_secondary_groups_exact'] or b['summary_sha256']!=sha(PUBLIC/'strong_controls/summary.json')):
        raise ValueError('All 216 metric groups and separate full forecast replays required')
    infer=read(PUBLIC/'inference_verification.json');extra=read(PUBLIC/'strong_controls/inference_verification.json')
    if not infer['all9_new_and9_control_inferences_exact'] or not extra['all9_history_control_arrays_exact']:
        raise ValueError('Complete primary and stronger-control forecast replay required')
    cfg=read(ROOT/'configs/m3w_european_eqmotion_v1.json')
    groups=[read(p) for p in sorted((PUBLIC/'strong_controls/groups').glob('*.json'))]
    for arm in ('dimensionless','history'):
        rows=[r for r in groups if r['control']==arm]
        for field,value in secondary['controls'][arm].items():
            if interval(rows,field,primary['localities'],cfg['bootstrap_resamples'],cfg['bootstrap_seed'])!=value:
                raise ValueError('Public secondary scalar reconstruction differs')
    comparisons=[('Original PCT','318,969',primary['effects']),
        ('Dimensionless PCT','318,969',secondary['controls']['dimensionless']),
        ('Current history-PCT','149,510',secondary['controls']['history'])]
    fmt=lambda r:f'{r["point"]:+.4f} [{r["CI95"][0]:+.4f}, {r["CI95"][1]:+.4f}]'
    rows=[read(p) for p in sorted((PUBLIC/'groups').glob('*.json'))]
    mean=lambda xs:sum(xs)/len(xs)
    gains={arm:mean([r['scores'][arm]['gain_vs_CV_percent'] for r in rows]) for arm in ('eqmotion','PCT')}
    current=[r for r in groups if r['control']=='history']
    whitelist_gains={arm:mean([r['scores'][arm]['gain_vs_CV_percent'] for r in current]) for arm in ('eqmotion','PCT')}
    lines=['# Complete Stronger-Control SOURCE Readout','',
        'Fresh_run: nine actual full EqMotion fits, first full refit and full primary',
        'forecast/metric replay; current history-PCT was inferred and replayed from',
        'its frozen weights, reproducing the existing array hashes. Cached_verified:',
        'dimensionless-PCT forecast arrays, after exact IDs/schema/source checks.',
        'This report itself aggregates public scalar receipts, not new inference.','',
        '## Fixed Contrasts','',
        '|Comparator|Indexed SOURCE population|Known gain difference pp [95% CI]|Completion-lower pp [95% CI]|',
        '|---|---:|---:|---:|']
    for name,n,s in comparisons:lines.append(f'|{name}|{n}|{fmt(s["known_pp"])}|{fmt(s["completion_lower_pp"])}|')
    lines+=['', 'The original-PCT comparison is the registered primary. Stronger controls',
        'were added before any new EqMotion outcome scoring and remain secondary.',
        'The two fixed populations are not pooled or substituted. Units are percentage',
        'points of gain using the common known CV denominator within each locality,',
        'not percent error reduction relative to each PCT. Intervals use 3,000 SOURCE',
        'locality resamples after averaging excluded folds and actual seeds 17/29/43.',
        'These are exposed SOURCE crossfit results, not independent final tests.','',
        f'Full population gains versus CV: EqMotion {gains["eqmotion"]:+.4f}%, original PCT {gains["PCT"]:+.4f}%.',
        f'Whitelist gains versus CV: EqMotion {whitelist_gains["eqmotion"]:+.4f}%, current PCT {whitelist_gains["PCT"]:+.4f}%.','',
        '![Observed gains and unresolved completions](../figures/strong_controls.svg)','',
        '## What Remains Negative','',
        'All completion-lower intervals cross zero. Wholly unknown-row bounds do not',
        'fill steps within partial futures or identify full-12-step ADE. Current-control',
        'known improvement is negative at locality 082; unknown-row candidate/control',
        'disagreement makes its lower comparison especially wide. That uncertainty',
        'is not an observed catastrophic future error. No locality is excluded.','',
        '|Locality|Known difference vs current PCT pp|Completion-lower pp|',
        '|---|---:|---:|']
    for site in primary['localities']:
        s=secondary['controls']['history']
        lines.append(f'|{site}|{s["known_pp"]["by_locality"][site]:+.4f}|{s["completion_lower_pp"]["by_locality"][site]:+.4f}|')
    lines+=['','The source backbone comparison is positive on observed labels. It does not',
        'prove a new M3W method, isolated equivariance or safe deployment. Current PCT',
        'has different grouped/partial neighbor processing and additional detector-history',
        'fields; EqMotion uses complete aligned neighbors. Observation/request lengths,',
        'SOURCE target identities, labels, seeds and fitting budgets match as registered.',
        'The published core is adapted to fixed deterministic K=1, not best-of-20 results.','',
        '## Next Decisive Experiment','',
        'A stronger proposal is now available for a separate SOURCE-only admission',
        'experiment. Preserve the physical floor and original 2% full-record contract;',
        'test whether new proposal costs/latent features improve safe routing rather',
        'than increase critic width again. Diagnose the 082 slice and unknown-row',
        'disagreement without dropping it or treating unavailable futures as zero error.',
        'Register that experiment before fitting; do not use calibration outcomes for',
        'threshold tuning. It has not been run by this forecast comparison.','',
        'No new policy risk pass, independent confirmation, cross-dataset world-model',
        'success or submission readiness. Historical calibration remains completed and',
        'failed; its private arrays, final confirmation and DroneCrowd are not accessed.',
        'Image-local pixels, raw stride 12 and released detector/tracker silver only.',
        'No metric, seconds, human-gold, physical-safety, true-3D or foundation claim.',
        'Stage5C and SMC remain off.','']
    (PUBLIC/'strong_controls/readout.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','svg.fonttype':'none',
        'svg.hashsalt':'m3w_european_eqmotion_controls','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for ax,field,title in zip(axes,('known_pp','completion_lower_pp'),
            ('Observed-label SOURCE contrasts','Wholly unknown-row completion lower')):
        for i,(_,_,s) in enumerate(comparisons):
            r=s[field];p=r['point'];lo,hi=r['CI95']
            ax.errorbar([p],[2-i],xerr=[[p-lo],[hi-p]],fmt='o',capsize=3,
                color=['#177d8d','#6d767c','#c58a2a'][i])
        ax.axvline(0,color='#777777',linestyle=':',linewidth=1)
        ax.set(yticks=[2,1,0],yticklabels=[n+'\n'+count+' SOURCE rows' for n,count,_ in comparisons],
            xlabel='EqMotion minus control (percentage points)',title=title,ylim=(-.5,2.5))
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Observed forecasting improves; missing outcomes remain unresolved',fontsize=11)
    directory=PUBLIC/'figures';directory.mkdir(parents=True,exist_ok=True)
    svg=directory/'strong_controls.svg';png=PRIVATE/'strong_controls.png'
    fig.savefig(svg,metadata={'Date':None});fig.savefig(png,dpi=150);plt.close(fig)
    receipt=dict(result_source='cached_verified_public_scalars_fresh_report_and_figure_not_new_inference',
        primary_summary_sha256=sha(PUBLIC/'summary.json'),primary_verification_sha256=sha(PUBLIC/'verification.json'),
        secondary_summary_sha256=sha(PUBLIC/'strong_controls/summary.json'),
        secondary_verification_sha256=sha(PUBLIC/'strong_controls/verification.json'),
        reporter_sha256=sha(Path(__file__)),svg_sha256=sha(svg),private_preview_sha256=sha(png),
        preview_visually_checked=False,independent_confirmation=False,deployment_changed=False)
    (directory/'strong_controls_sources.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
