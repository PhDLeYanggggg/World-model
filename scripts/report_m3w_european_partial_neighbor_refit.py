"""Report the fixed neural contrast without selecting seeds or changing deployment."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import run_m3w_european_partial_neighbor_refit as run
from src.evaluation.m3w_partial_neighbor_refit import paired_localities


def fmt(x): return 'undefined' if x is None else f'{x:+.3f}'


def interval(s):
    return 'undefined (fixed roster unsupported)' if s['ci95'] is None else f"[{fmt(s['ci95'][0])}, {fmt(s['ci95'][1])}]"


def report():
    doc = json.loads((run.PUBLIC/'evaluation.json').read_text())
    freeze = json.loads((run.PUBLIC/'prediction_freeze.json').read_text())
    reg = json.loads((run.PUBLIC/'registration.json').read_text())
    for path,h in reg['bindings'].items(): assert run.digest(ROOT/path) == h
    assert len(doc['rows']) == 576 and len(doc['expected_localities']) == 12
    cfg = json.loads((ROOT/run.CONFIG).read_text())
    summaries = doc['summaries']
    main = summaries['ADE_all']['gain_vs_legacy_percent']
    easy = summaries['ADE_positive_easy']['gain_vs_legacy_percent']
    hard = summaries['ADE_hard']['gain_vs_legacy_percent']
    gates = dict(primary_positive_lower_CI=main['ci95'] is not None and main['ci95'][0]>0,
        easy_vs_legacy_within_two_percent=easy['point'] is not None and easy['point']>=-cfg['easy_degradation_guard_percent'],
        hard_vs_legacy_nonnegative=hard['point'] is not None and hard['point']>=0)
    gates['exploratory_forecaster_benefit'] = all(gates.values())
    gates.update(independent_confirmation=False, policy_safety_established=False,
                 deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    run.immutable_json(run.PUBLIC/'gates.json', gates)

    text = ['# Matched Partial-Neighbor Neural Results', '',
        'Fresh source-development neural refit, not independent confirmation. Positive is lower error.', '',
        '| Endpoint/subset | vs legacy neural (%) | 95% locality interval | vs training-selected baseline (%) | vs CV (%) |',
        '|---|---:|---|---:|---:|']
    for key,v in summaries.items():
        a=v['gain_vs_legacy_percent']
        text.append(f"| {key} | {fmt(a['point'])} | {interval(a)} | {fmt(v['gain_vs_reference_percent']['point'])} | {fmt(v['gain_vs_CV_percent']['point'])} |")
    text += ['', 'These are averages of locality-specific percentage gains, not a pooled-pixel ratio.',
        'Each locality averages two producer contexts and three seeds before 3,000 resamples.',
        'The 12 source localities were already development-exposed; producer fits overlap.',
        'ADE uses available requested future labels; FDE requires the final requested label.',
        'Easy/hard membership uses training-derived CV-ADE cuts, including for FDE reporting.',
        'FDE gains compare FDE against FDE. Zero-reference percentages remain undefined.', '',
        '## By Locality', '', '| Locality | ADE vs legacy (%) | easy gain (%) | hard gain (%) | FDE gain (%) |',
        '|---|---:|---:|---:|---:|']
    for site in doc['expected_localities']:
        vals=[summaries[k]['gain_vs_legacy_percent']['by_site'][site]
              for k in ('ADE_all','ADE_positive_easy','ADE_hard','FDE_all')]
        text.append('| '+site+' | '+' | '.join(map(fmt,vals))+' |')
    text += ['', '## By Training Seed', '', '| Seed | ADE gain (%) | Locality interval |', '|---|---:|---|']
    for seed,v in doc['per_seed'].items(): text.append(f"| {seed} | {fmt(v['point'])} | {interval(v)} |")
    text += ['', '## By Producer', '', '| Fold | Seed | Held localities | ADE gain (%) |', '|---|---|---:|---:|']
    for fold in cfg['producer_folds']:
        for seed in cfg['seeds']:
            rr=[r for r in doc['rows'] if r['endpoint']=='ADE' and r['subset']=='all' and r['fold']==fold and r['seed']==seed]
            v=paired_localities(rr, sorted({r['site'] for r in rr}), 'gain_vs_legacy_percent')
            text.append(f"| {fold} | {seed} | {len(rr)} | {fmt(v['point'])} |")
    (run.PUBLIC/'results.md').write_text('\n'.join(text)+'\n')

    text=['# Absolute Costs and Error Tails', '',
        'All costs are image-local source coordinates. Do not pool these as a metric distance.',
        'Each row is a dependent held-locality view of one fixed producer and seed.', '',
        '| Trial | Locality | Endpoint/subset | Labeled rows | New mean | Legacy mean | Reference mean | New p95 | Legacy p95 | Mean positive harm | Mean positive gain |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in doc['rows']:
        m=r['metric']
        cols=[m.get(k) for k in ('new_mean','old_mean','reference_mean','new_p95','old_p95',
              'mean_positive_harm_vs_legacy','mean_positive_gain_vs_legacy')]
        text.append(f"| {r['trial']} | {r['site']} | {r['endpoint']}/{r['subset']} | {m['rows']} | "+' | '.join(map(fmt,cols))+' |')
    (run.PUBLIC/'absolute_costs.md').write_text('\n'.join(text)+'\n')
    motion=['# Motion Proxies', '',
        'Finite coordinates and raw-step second differences are numerical diagnostics, not physical safety.', '',
        '| Trial | Locality | Indexed queries | Recordings | Finite | New second difference | Legacy second difference |',
        '|---|---|---:|---:|---|---:|---:|']
    for r in doc['motion_proxies']:
        motion.append(f"| {r['trial']} | {r['site']} | {r['rows']} | {r['recordings']} | {r['finite_output']} | {r['raw_step_acceleration_new']:.6f} | {r['raw_step_acceleration_legacy']:.6f} |")
    (run.PUBLIC/'motion_proxies.md').write_text('\n'.join(motion)+'\n')

    direction=['# Metric Direction Note', '',
        'The registered generic aggregator stores the minimum locality statistic in its',
        '`worst_locality` field. This names the worst gain correctly, but for absolute',
        'harm it is only the minimum, not the worst harm. No harm-minimum field is used',
        'in a gate or a safety conclusion. The correct maxima are shown below without',
        'altering the frozen evaluation file, primary gains, intervals or training.', '',
        '| Endpoint/subset | Maximum locality mean harm vs legacy |', '|---|---:|']
    for key,v in summaries.items():
        values=list(v['absolute_harm_vs_legacy']['by_site'].values())
        direction.append(f"| {key} | {fmt(max(values) if all(x is not None for x in values) else None)} |")
    (run.PUBLIC/'metric_direction_note.md').write_text('\n'.join(direction)+'\n')

    endpoints=[]
    for ref in freeze['training']:
        assert run.artifact(ROOT/ref['path']) == ref
        endpoints.append(json.loads((ROOT/ref['path']).read_text()))
    fit_seconds=sum(e['fit']['seconds'] for e in endpoints)
    conclusion=f'''# Conclusion

Nine partial-neighbor neural forecasters completed 4,000 updates each, with
three fixed seeds and matched locality folds. A fresh 4,000-update legacy
control reproduces every model parameter exactly, including after pilot
checkpoint resume. Every new/old pair has identical sampling counts and final
sampler state. All weights and predictions were frozen before comparison.

The primary ADE gain against legacy neural prediction is {fmt(main['point'])}%
with a 95% exploratory locality interval {interval(main)}%. Positive-easy gain
is {fmt(easy['point'])}%; hard gain is {fmt(hard['point'])}%.
The preregistered exploratory benefit screen is **{'passed' if gates['exploratory_forecaster_benefit'] else 'failed'}**.
See results.md for causal-baseline comparisons; improving a weak neural
control alone does not establish useful intervention or safe deployment.

Training is fresh native Torch CPU work, not a NumPy fallback. The nine new
fits used {fit_seconds:.2f} training seconds in total, excluding preflight and
inference. Cached legacy assets are hash-verified and replayed with fresh
inference. Raw source labels are detector-derived, not human gold.

The task is eight observations and twelve requested future steps at raw-frame
stride12. It is not raw t+50, calibrated seconds, metric 3D or foundation-model
evidence. Independent selection/calibration/confirmation remain closed. No
new gain/harm policy is trained, no deployment is changed, and Stage5C/SMC
remain disabled. A supported input fix is not a calibrated safety guarantee.
'''
    (run.PUBLIC/'conclusions.md').write_text(conclusion)

    training=['# Training Endpoints', '', '| Trial | Parameters | Updates | Draws | Unique train queries | Held draws | Fit seconds | Last logged loss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for e in endpoints:
        i,f=e['identity']['parent_trial'],e['fit']
        training.append(f"| single{i['fold']}_seed{i['seed']} | {f['parameters']} | {f['step']} | {f['total_draws']} | {f['unique_training_rows']} | {f['held_rows_sampled']} | {f['seconds']:.3f} | {f['losses'][-1]['loss']:.6f} |")
    training += ['', 'All fixed-budget endpoints are retained; the last mini-batch loss is not a validation score.',
        'Losses, gradients, learning rate and heartbeat are retained locally in hash-bound fit receipts.',
        'The legacy control adds 4,000 updates; total new execution is 40,000 updates, not 36,000 plus a separate pilot budget.']
    (run.PUBLIC/'training.md').write_text('\n'.join(training)+'\n')
    plot(doc,endpoints)
    print(json.dumps(dict(primary=main['point'], ci95=main['ci95'], gates=gates)))


def plot(doc, endpoints):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    sites=doc['expected_localities']; s=doc['summaries']
    for k,label,color in [('ADE_all','All','#17788a'),('ADE_positive_easy','Easy','#af4054'),('ADE_hard','Hard','#528746')]:
        ax[0].plot([s[k]['gain_vs_legacy_percent']['by_site'][v] for v in sites],range(len(sites)),'.',label=label,color=color)
    ax[0].set_yticks(range(len(sites)),sites,fontsize=8); ax[0].axvline(0,color='black',lw=.6)
    ax[0].set_xlabel('ADE gain vs legacy (%)'); ax[0].legend(fontsize=8); ax[0].set_title('Each held locality; all seeds')
    for i,(seed,v) in enumerate(doc['per_seed'].items()):
        lo,hi=v['ci95']; point=v['point']
        ax[1].plot([lo,hi],[i,i],color='#17788a'); ax[1].plot(point,i,'o',color='#17788a')
    ax[1].set_yticks(range(3),doc['per_seed'].keys()); ax[1].axvline(0,color='black',lw=.6)
    ax[1].set_xlabel('ADE gain vs legacy (%)'); ax[1].set_title('Exploratory locality intervals')
    for e in endpoints:
        rows=e['fit']['losses']; steps=[r['step'] for r in rows]; loss=[r['loss'] for r in rows]
        smoothed=np.convolve(loss,np.ones(5)/5,mode='valid')
        ax[2].plot(steps[4:],smoothed,alpha=.6,lw=1)
    ax[2].set_xlabel('Optimizer updates'); ax[2].set_ylabel('Training loss (5-log running mean)')
    ax[2].set_title('Nine fresh neural fits; not validation')
    fig.savefig(run.PUBLIC/'matched_refit.png',dpi=160,metadata={'Software':'M3W fixed contrast report'})
    plt.close(fig)


if __name__=='__main__': report()
