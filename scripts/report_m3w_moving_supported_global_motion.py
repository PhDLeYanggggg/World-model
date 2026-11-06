"""Report only frozen public scalar metrics, including all negative cells."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_supported_global_motion as run


def number(value):return 'NA' if value is None else f'{value:+.4f}'


def primary_positive(a):
    ci=a['contrasts']['primary']['completion_lower_pp']['interval95']
    return ci is not None and ci[0]>0


def worst_signed(a,arm,name):
    values=[v['known_easy_net_degradation'] for r in a['rows'] if r['arm']==arm
        for v in r['policies'][name]['recordings'].values() if v['known_easy_net_degradation'] is not None]
    return max(values) if values else None


def plot_fit_losses():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    freeze=run.read(run.PUBLIC/'training_freeze.json')
    fits=[run.read(run.checked(ref)) for ref in freeze['fits']]
    if len(fits)!=96:raise ValueError('All96actual fitting traces required')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,support in zip(axes,('own','augmented')):
        for suffix,color in (('local','#2563eb'),('global','#0f766e')):
            traces=[r['trace'] for r in fits if r['arm']==support+'_'+suffix]
            if len(traces)!=24:raise ValueError('All24source/risk-seed fits per cell required')
            steps=[r['step'] for r in traces[0]]
            if any([r['step'] for r in t]!=steps for t in traces):raise ValueError('Fixed trace budget changed')
            v=run.np.array([[r['total'] for r in t] for t in traces])
            median=run.np.median(v,axis=0);q=run.np.quantile(v,[.25,.75],axis=0)
            ax.plot(steps,median,color=color,label=suffix+' (24 fits)')
            ax.fill_between(steps,q[0],q[1],alpha=.12,color=color)
        ax.set(title=support+' support: median and descriptive IQR',xlabel='Optimizer updates',
            ylabel='Matched training-minibatch objective (not validation)');ax.grid(alpha=.2);ax.legend()
    path=run.PRIVATE/'figures';path.mkdir(exist_ok=True)
    fig.savefig(path/'actual_fitting_losses.png',dpi=170);fig.savefig(path/'actual_fitting_losses.pdf');plt.close(fig)


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm,color in zip(run.method.ARMS,('#2563eb','#0f766e','#7c3aed','#b45309')):
        p=a['policies'][arm]['joint_free'];u=p['utility']['known'];ci=u['interval95']
        i=list(run.method.ARMS).index(arm)
        axes[0].errorbar(i,u['point'],yerr=[[u['point']-ci[0]],[ci[1]-u['point']]],fmt='o',color=color,capsize=4)
        axes[1].bar(i,p['safety']['supported_record_views'],color=color)
    labels=['Own/local','Own/global','Augmented/local','Augmented/global']
    for ax in axes:
        ax.set_xticks(range(4),labels,rotation=18);ax.grid(axis='y',alpha=.2)
    axes[0].axhline(0,color='#525252',linewidth=.6)
    axes[0].set(ylabel='Available-label ADE gain versus floor (%)',title='Joint-free, all exposed own-B rows')
    axes[1].axhline(72,color='#dc2626',linestyle='--',label='All 72 repeated views required')
    axes[1].set(ylabel='Original recording contract passed (of 72)',ylim=(0,76),title='24 recordings, three risk seeds');axes[1].legend()
    path=run.PRIVATE/'figures';path.mkdir(exist_ok=True)
    fig.savefig(path/'support_context_matrix.png',dpi=170);fig.savefig(path/'support_context_matrix.pdf');plt.close(fig)


def main():
    a=run.read(run.PUBLIC/'analysis.json');freeze=run.read(run.PUBLIC/'training_freeze.json')
    verification=run.read(run.PUBLIC/'verification.json')
    if freeze['models']!=96 or not verification['all96_full_refits_exact']:raise ValueError('Complete actual fits required')
    lines=['# Data Support By Global Past Motion','',
        'Fresh_run:96Torch cost/support heads and96complete exact refits. Cached_verified',
        'inputs:fixed forecast bank and admitted past-only feature caches. Not_run:new',
        'forecaster training,new independent confirmation or upstream label-causality',
        'certification. No risk-rule or deployment change.','',
        '## Full Four-Cell Comparison','',
        '| Cell | Decision | Known gain % | Separate lower % | Hard gain % | Easy gain % | Switch % | Original pass | Easy harm violations | Easy NA | Worst signed easy loss % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in run.method.ARMS:
        for name in run.POLICIES:
            p=a['policies'][arm][name];s=p['safety'];tail=worst_signed(a,arm,name)
            values=[p['utility']['known']['point'],p['utility']['lower']['point'],p['hard_gain_percent'],
                p['easy_gain_percent'],100*p['switch_rate']]
            lines.append(f'| {arm} | {name} | '+' | '.join(number(v) for v in values)+
                f' | {s["supported_record_views"]}/72 | {s["known_easy_violations"]} | {s["undefined_easy_views"]} | {number(None if tail is None else 100*tail)} |')
    lines+=['','These72views repeat24recordings over3risk-head seeds,not independent',
        'forecaster seeds. Mean easy gain does not replace per-record positive harm,',
        'signed easy-net,completion and support checks. Undefined views remain NA.','',
        '## Registered And Secondary Contrasts','',
        '| Contrast | Component | Point pp | Exploratory source interval |','|---|---|---:|---|']
    for name,parts in a['contrasts'].items():
        for key,u in parts.items():
            ci=u['interval95'];interval='NA' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'
            lines.append(f'| {name} | {key} | {number(u["point"])} | {interval} |')
    lines+=['','Primary is augmented-global minus augmented-local joint-free paired lower.',
        'Own-context and support-by-context interaction are secondary. Interaction',
        'lower subtracts own paired upper,not own lower. Fixed-control contrast uses',
        'the preregistered guarded1/32control,not an outcome-selected amplitude.','',
        'Intervals:3,000source-component bootstrap draws after averaging risk seeds',
        'within each of8components. They are exploratory and not adjusted for this',
        'project\'s adaptive search. No independent confirmation or novelty proof.','',
        '## Primary By Source And Risk Seed','',
        '| Source | Known pp | Paired completion lower pp |','|---|---:|---:|']
    primary=a['contrast_rows']['primary']
    for source in sorted(set(r['source'] for r in primary)):
        rr=[r for r in primary if r['source']==source]
        lines.append(f'| {source} | {number(float(run.np.mean([r["known_pp"] for r in rr])))} | '
            f'{number(float(run.np.mean([r["completion_lower_pp"] for r in rr])))} |')
    lines+=['','| Risk-head seed | Known pp | Paired completion lower pp |','|---|---:|---:|']
    for seed in (17,29,43):
        rr=[r for r in primary if r['seed']==seed]
        lines.append(f'| {seed} | {number(float(run.np.mean([r["known_pp"] for r in rr])))} | '
            f'{number(float(run.np.mean([r["completion_lower_pp"] for r in rr])))} |')
    lines+=['','## Interpretation Boundaries','',
        'The equal-parameter-count local/global cells test an already unsuccessful feature',
        'under a new support/loss regime. Donor augmentation changes producer/candidate',
        'population as well as source support; it is not a pure source-count effect.',
        'The48variables describe past crowd motion,not images or calibrated camera motion.',
        'No new JEPA,Transformer or trajectory dynamics model is trained here.','',
        f'Registered conservative utility supported: {primary_positive(a)}.',
        'Original augmented-global full risk supported: '+str(a['policies']['augmented_global']['joint_free']['safety']['original_record_contract_pass'])+'.',
        'Neither scalar alone authorizes deployment; the whole original contract and',
        'independent evidence remain required. Floor empty references remain NA.','',
        'Native8/12image-coordinate available-label ADE,not complete12ADE,SDD rawt50,',
        'metric/seconds,true3D,foundation or human-gold evidence. Stage5C/SMC off.','',
        '[Protocol](protocol.md) | [Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)']
    (run.PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    gates=['# Support By Global Motion Gates','', '| Gate | Status | Evidence |','|---|---|---|',
        '| Registered definition | PASS | Fixed before fresh preparation/training |',
        '| Real Torch training | PASS | 96final models,393216updates |',
        '| Complete exact refits | PASS | 96models,393216repeat updates |',
        '| Matched parameter count/draws | PASS | Feature pairs; masked parameters inactive; support factor differs |',
        '| Causal role/source controls | PASS within pipeline | Known fitting normalization,no query-source fitting,no future input |',
        '| Registered utility | '+('PASS, exploratory' if primary_positive(a) else 'FAIL / NA')+' | Augmented global-local paired lower |',
        '| Original full risk | '+('PASS, development only' if a['policies']['augmented_global']['joint_free']['safety']['original_record_contract_pass'] else 'FAIL')+' | All72views and original2%rules |',
        '| Independent confirmation | NOT RUN | Already exposed own-B only |',
        '| Original24group main | NA | No new C/H |',
        '| Identity exception/deployment/submission | FALSE | No approval or evidence change |',
        '| Stage5C / SMC | FALSE | Not executed |']
    (run.PUBLIC/'world_model_gate_supported_global_motion.md').write_text('\n'.join(gates)+'\n')
    plot(a);plot_fit_losses();print('Complete four-cell matrix reported; no outcome-selected arm or deployment.')


if __name__=='__main__':main()
