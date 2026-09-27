"""Fixed source-development readout; no model selection or deployment promotion."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_european_dimensionless_refit as run
from src.evaluation.m3w_dimensionless_refit import paired_localities


def fmt(v): return 'undefined' if v is None else f'{v:+.3f}'


def interval(s):
    return 'undefined (fixed roster unsupported)' if s['ci95'] is None else '['+', '.join(map(fmt,s['ci95']))+']'


def main():
    d=json.loads((run.PUBLIC/'evaluation.json').read_text())
    frozen=json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    reg=json.loads((run.PUBLIC/'registration.json').read_text())
    cfg=json.loads((ROOT/run.CONFIG).read_text())
    for p,h in reg['bindings'].items(): assert run.digest(ROOT/p)==h
    assert len(d['rows'])==576 and len(d['causal_slices'])==360
    gates=json.loads((run.PUBLIC/'gates.json').read_text()); s=d['summaries']
    main=s['ADE_all']['gain_vs_grouped_percent']; easy=s['ADE_positive_easy']['gain_vs_grouped_percent']
    hard=s['ADE_hard']['gain_vs_grouped_percent']; ecv=s['ADE_positive_easy']['gain_vs_CV_percent']
    fits=[]
    for r in frozen['training']:
        assert run.artifact(ROOT/r['path'])==r
        fits.append(json.loads((ROOT/r['path']).read_text()))
    lines=['# Dimensionless Correction-Fraction Results','',
        'Fresh source-development inference and scoring; cached_verified controls. Positive means lower error.',
        'No result is independent confirmation or an intervention-policy safety guarantee.','',
        '| Endpoint/subset | vs matched grouped (%) | 95% locality interval | vs prior flat neural (%) | vs train baseline (%) | vs CV (%) |',
        '|---|---:|---|---:|---:|---:|']
    for key,v in s.items():
        lines.append(f"| {key} | {fmt(v['gain_vs_grouped_percent']['point'])} | {interval(v['gain_vs_grouped_percent'])} | "+
            ' | '.join(fmt(v[k]['point']) for k in ('gain_vs_flat_percent','gain_vs_reference_percent','gain_vs_CV_percent'))+' |')
    lines+=['','Means are equal-locality means of percentage gains, not pooled source-coordinate ratios.',
        'Each locality averages two producer contexts and three seeds before 3,000 locality resamples.',
        'Source localities were already exposed for development; fitting sets and windows overlap.',
        'ADE uses supported requested labels; FDE requires the final requested label.',
        'Easy/hard use fixed training CV-ADE thresholds, including for FDE. Zero-CV gains are undefined.',
        'Absolute costs and positive gain/harm remain reported for zero-reference rows.','',
        '## By Locality','','| Locality | ADE gain (%) | Easy gain (%) | Hard gain (%) | FDE gain (%) |',
        '|---|---:|---:|---:|---:|']
    for site in d['expected_localities']:
        lines.append('| '+site+' | '+' | '.join(fmt(s[k]['gain_vs_grouped_percent']['by_site'][site]) for k in
            ('ADE_all','ADE_positive_easy','ADE_hard','FDE_all'))+' |')
    lines+=['','## By Seed','','| Seed | ADE vs grouped (%) | 95% locality interval |','|---|---:|---|']
    for seed,v in d['per_seed'].items(): lines.append(f"| {seed} | {fmt(v['point'])} | {interval(v)} |")
    lines+=['','## By Producer','','| Fold | Seed | ADE vs grouped (%) |','|---|---|---:|']
    for fold in cfg['producer_folds']:
        for seed in cfg['seeds']:
            rows=[r for r in d['rows'] if r['endpoint']=='ADE' and r['subset']=='all' and r['fold']==fold and r['seed']==seed]
            v=paired_localities(rows,sorted({r['site'] for r in rows}),'gain_vs_grouped_percent')
            lines.append(f"| {fold} | {seed} | {fmt(v['point'])} |")
    (run.PUBLIC/'results.md').write_text('\n'.join(lines)+'\n')

    lines=['# Absolute Error and Tails','',
        'Image-local source units, not meters. Each row is a dependent producer/seed/locality view.',
        'Positive harm means a larger error; negative mean harm means lower mean error.','',
        '| Trial | Locality | Endpoint/subset | Rows | Dimensionless mean | Grouped mean | Reference mean | CV mean | Dimensionless p95 | Grouped p95 | Dimensionless p99 | Grouped p99 | Positive harm | Positive gain | Error increased fraction |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['rows']:
        m=r['metric']; values=[m.get(k) for k in ('new_mean','old_mean','reference_mean','cv_mean',
            'new_p95','old_p95','new_p99','old_p99','mean_positive_harm_vs_grouped','mean_positive_gain_vs_grouped','increased_error_fraction')]
        lines.append(f"| {r['trial']} | {r['site']} | {r['endpoint']}/{r['subset']} | {m['rows']} | "+' | '.join(map(fmt,values))+' |')
    (run.PUBLIC/'absolute_costs.md').write_text('\n'.join(lines)+'\n')

    lines=['# Prespecified Causal Slices','',
        'Descriptive source readout, not interaction ablation or a new primary endpoint.',
        'Slices overlap; they are not additive components of the primary gain.',
        'Incomplete fixed-roster intervals are undefined, not replaced by a favorable subset.','',
        '| Slice | ADE gain vs grouped (%) | 95% locality interval |','|---|---:|---|']
    for name,v in d['causal_slice_summaries'].items(): lines.append(f"| {name} | {fmt(v['point'])} | {interval(v)} |")
    lines+=['','| Trial | Locality | Slice | Labeled rows | ADE gain vs grouped (%) |','|---|---|---|---:|---:|']
    for r in d['causal_slices']:
        lines.append(f"| {r['trial']} | {r['site']} | {r['subset']} | {r['metric']['rows']} | {fmt(r['metric'].get('gain_vs_grouped_percent'))} |")
    (run.PUBLIC/'causal_slices.md').write_text('\n'.join(lines)+'\n')

    lines=['# Numerical Motion Proxies','',
        'Raw-step second differences do not establish physical smoothness or collision safety.',
        'This per-target deterministic forecaster does not impose scene-joint trajectory consistency.','',
        '| Trial | Locality | Queries | Recordings | Finite | Dimensionless second difference | Grouped second difference |',
        '|---|---|---:|---:|---|---:|---:|']
    for r in d['motion_proxies']:
        lines.append(f"| {r['trial']} | {r['site']} | {r['rows']} | {r['recordings']} | {r['finite_output']} | "+
            f"{r['raw_step_acceleration_dimensionless']:.6f} | {r['raw_step_acceleration_grouped']:.6f} |")
    (run.PUBLIC/'motion_proxies.md').write_text('\n'.join(lines)+'\n')

    lines=['# Fixed-Budget Training','','| Trial | Parameters | Updates | Draws | Distinct fitting queries | Held draws | Fit seconds | Last logged loss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in fits:
        i,f=e['identity']['parent_trial'],e['fit']
        lines.append(f"| single{i['fold']}_seed{i['seed']} | {f['parameters']} | {f['step']} | {f['total_draws']} | "+
            f"{f['unique_training_rows']} | {f['held_rows_sampled']} | {f['seconds']:.3f} | {f['losses'][-1]['loss']:.6f} |")
    lines+=['','The first 100 updates resume into the first 4,000-update endpoint, not a second budget.',
        'All paired initial parameters and sampling states/counts match the grouped controls.',
        'The last training-batch loss is not a validation score. All nine endpoints are retained.',
        f"Cumulative fresh training seconds: {sum(e['fit']['seconds'] for e in fits):.3f}.",
        'No new control fitting; cached controls have hash and full fresh inference verification.']
    (run.PUBLIC/'training.md').write_text('\n'.join(lines)+'\n')

    conclusion=f'''# Conclusion

Nine real native-Torch neural models were trained for 4,000 updates each,
three seeds per producer fold. Partial-neighbor inputs, training samples,
loss, initial weights and parameter budget were matched to grouped controls.
Only coordinate-scale restoration before the bounded radial squash was removed.

Primary ADE gain vs matched grouped: {fmt(main['point'])}%, with exploratory
95% locality interval {interval(main)}%. Easy gain vs grouped: {fmt(easy['point'])}%;
hard gain: {fmt(hard['point'])}%. The registered forecaster screen
**{'passes' if gates['exploratory_forecaster_benefit'] else 'fails'}**.
Easy gain vs causal CV is {fmt(ecv['point'])}% ({interval(ecv)}%).
An improvement over an unsafe neural control is not safe deployment.

No risk-head fitting, threshold choice, deployment change, independent
selection/calibration/confirmation access, Stage5C execution or SMC occurred.
This is source-development evidence, not a new historical Stage37/t+50 score.
The task is obs8/pred12, raw-frame stride12, detector-derived image-local
trajectories. It is not metric, calibrated seconds, true3D, human gold or
foundation-model evidence. Tables disclose negative seeds, easy and tail costs.

Unit-consistent correction fractions are an implementation repair, not our
paper's novelty claim. Gain/harm prediction and calibrated scene-joint
intervention remain the intended method question and are not tested here.
'''
    (run.PUBLIC/'conclusions.md').write_text(conclusion)
    plot(d,fits)
    print(json.dumps(dict(primary=main['point'],ci95=main['ci95'],easy_vs_CV=ecv['point'],gates=gates)))


def plot(d,fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    s=d['summaries']; sites=d['expected_localities']
    for k,label,color in [('ADE_all','All','#17788a'),('ADE_positive_easy','Easy','#af4054'),('ADE_hard','Hard','#528746')]:
        ax[0].plot([s[k]['gain_vs_grouped_percent']['by_site'][v] for v in sites],range(len(sites)),'.',label=label,color=color)
    ax[0].set_yticks(range(len(sites)),sites,fontsize=8); ax[0].axvline(0,color='black',lw=.6)
    ax[0].set_xlabel('ADE gain vs matched grouped (%)'); ax[0].legend(fontsize=8); ax[0].set_title('Each held locality; all seeds')
    for i,(seed,v) in enumerate(d['per_seed'].items()):
        lo,hi=v['ci95']; point=v['point']; ax[1].plot([lo,hi],[i,i],color='#17788a'); ax[1].plot(point,i,'o',color='#17788a')
    ax[1].set_yticks(range(3),d['per_seed'].keys()); ax[1].axvline(0,color='black',lw=.6)
    ax[1].set_xlabel('ADE gain vs matched grouped (%)'); ax[1].set_title('Exploratory locality intervals')
    for e in fits:
        rows=e['fit']['losses']; loss=[r['loss'] for r in rows]
        ax[2].plot([r['step'] for r in rows][4:],np.convolve(loss,np.ones(5)/5,mode='valid'),alpha=.6,lw=1)
    ax[2].set_xlabel('Optimizer updates'); ax[2].set_ylabel('Training loss (5-log running mean)')
    ax[2].set_title('Nine fresh fits; not validation')
    fig.savefig(run.PUBLIC/'matched_fraction.png',dpi=160,metadata={'Software':'M3W fixed contrast report'}); plt.close(fig)


if __name__=='__main__': main()
