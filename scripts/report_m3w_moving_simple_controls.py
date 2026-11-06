"""Public-scalar-only report of fixed controls, without choosing a coefficient."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_simple_controls as run


def text_number(x):return 'NA' if x is None else f'{x:+.4f}'


def primary_supported(a):
    ci=a['primary']['completion_lower_pp']['interval95']
    return bool(ci is not None and ci[0]>0)


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=run.np.array(run.method.FRACTIONS)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for guard,color in zip(run.method.GUARDS,('#2563eb','#15803d')):
        p=[a['static_policies'][run.method.name(i,guard)] for i in range(1,7)]
        point=[r['utility']['known']['point'] for r in p]
        ci=[r['utility']['known']['interval95'] for r in p]
        axes[0].plot(x,point,'o-',color=color,label=guard)
        axes[0].fill_between(x,[c[0] for c in ci],[c[1] for c in ci],alpha=.12,color=color)
        axes[1].plot(x,[r['safety']['supported_record_views'] for r in p],'o-',color=color,label=guard)
    nn=a['learned_policies']['expert_context']['joint_free']['utility']['known']['point']
    axes[0].axhline(nn,color='#6b7280',linestyle='--',label='Learned joint (risk-seed mean)')
    axes[0].axhline(0,color='#525252',linewidth=.6)
    axes[0].set(ylabel='Available-label ADE gain versus CV (%)',title='Exact exposed own-B cohort, native 8/12')
    axes[1].axhline(24,color='#dc2626',linestyle='--',label='All 24 static views required')
    axes[1].set(ylabel='Original recording contract passed (of 24)',title='One forecaster seed; no identity exception',ylim=(0,26))
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(x,['1/32','1/16','1/8','1/4','1/2','1'])
        ax.set_xlabel('Fixed forecast amplitude');ax.grid(alpha=.2);ax.legend(fontsize=8)
    p=run.PRIVATE/'figures';p.mkdir(exist_ok=True)
    fig.savefig(p/'fixed_control_matrix.png',dpi=170);fig.savefig(p/'fixed_control_matrix.pdf');plt.close(fig)


def main():
    a=run.read(run.PUBLIC/'analysis.json')
    if not a['frozen_learned_metrics_exact'] or len(a['static_policies'])!=13:raise ValueError('Complete exact controls required')
    lines=['# Exact-Cohort Fixed-Amplitude Controls','',
        'Fresh causal static choices and complete scoring on62,466exposed own-B rows.',
        'All six original coefficients are reported, not selected from this readout.',
        'No new model is trained; frozen learned-policy metrics reproduce exactly.','',
        '## Complete Static Matrix','',
        '| Policy | Known gain % | Separate lower % | Hard gain % | Easy gain % | Original risk pass | Known easy violations | Easy NA | Worst signed easy degradation % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,p in a['static_policies'].items():
        s=p['safety'];u=p['utility'];t=p['tails']['known_easy_net_degradation']['maximum_defined']
        lines.append(f'| {name} | '+ ' | '.join(text_number(u[k]['point']) for k in ('known','lower','hard','easy'))+
            f' | {s["supported_record_views"]}/{s["required_record_views"]} | {s["known_easy_violations"]} | '
            f'{s["undefined_easy_views"]} | {text_number(None if t is None else 100*t)} |')
    lines+=['','Static policies have24recording views and one frozen forecaster seed17.',
        'They have no cost-head seeds. The floor has zero added motion error but',
        'empty selected-reference risk remains NA, not an identity-certified pass.','',
        '## Frozen Learned Controls','',
        '| Family | Decision | Known gain % | Separate lower % | Original risk pass | Known easy violations | Easy NA |',
        '|---|---|---:|---:|---:|---:|---:|']
    for family,policies in a['learned_policies'].items():
        for name in ('independent','joint_free'):
            p=policies[name];s=p['safety']
            lines.append(f'| {family} | {name} | {text_number(p["utility"]["known"]["point"])} | '
                f'{text_number(p["utility"]["lower"]["point"])} | {s["supported_record_views"]}/72 | '
                f'{s["known_easy_violations"]} | {s["undefined_easy_views"]} |')
    lines+=['','72learned views repeat24recordings over3risk-head seeds. They are not',
        '72independent recordings or three forecaster runs. All five decisions',
        'per learned family are retained in analysis.json and the per-context metrics.','',
        '## Registered Primary','',
        'Expert-context joint-free minus guarded alpha1/32; all required sources:',
        '| Contrast | Point pp | Exploratory source interval |','|---|---:|---|']
    for key,v in a['primary'].items():
        ci=v['interval95'];interval='NA' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'
        lines.append(f'| {key} | {text_number(v["point"])} | {interval} |')
    lines+=['','Eight source components,3,000resamples,with learned risk seeds averaged',
        'within source. The static baseline is repeated only for pairing, not as',
        'new independent model runs. Separate completion lower gains cannot simply',
        'be subtracted to get the paired contrast. A negative lower contrast is a',
        'failed conservative criterion, not proof of the unknown true effect sign.','',
        '## Verdict','',
        'Learned primary positive: '+str(primary_supported(a))+'.',
        'Any fixed nonfloor policy passes all24original views: '+str(any(p['safety']['original_record_contract_pass']
            for n,p in a['static_policies'].items() if n!='floor'))+'.',
        'No coefficient or deployment winner is selected. Average accuracy does',
        'not override original per-record2%positive all/easy harm,signed easy-net,',
        'reference support or unknown-label completion. The original24-source-group',
        'main stays NA; no new independent C/H is opened.','',
        'Native8/12available-label image-coordinate ADE is not complete12ADE,',
        'SDD raw t50,metric/seconds,true3D,foundation or human-gold evidence.',
        'Annotation-generation causality remains unverified. Stage5C/SMC off.','',
        '[Protocol](protocol.md) | [Interpretation](failure_analysis.md) | [Reproduction](operation_zh.md)']
    (run.PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    gates=['# Exact-Cohort Strong-Control Gates','', '| Gate | Status | Evidence |','|---|---|---|',
        '| Fixed full cohort | PASS | 62466rows,8sourcecomponents,24recordings |',
        '| All static choices frozen before scoring | PASS | 13policies,3context receipts |',
        '| Learned control parity | PASS | Every cached metric exact |',
        '| Registered learned utility over guarded minimum | '+('PASS, exploratory' if primary_supported(a) else 'FAIL / NA')+' | No secondary coefficient replaces it |',
        '| Full static risk | '+('PASS for some diagnostic controls' if any(p['safety']['original_record_contract_pass'] for n,p in a['static_policies'].items() if n!='floor') else 'FAIL for every coefficient')+' | All24views required |',
        '| Identity exception | FALSE | Empty selected reference remains NA |',
        '| Independent confirmation/deployment | NOT RUN / FALSE | Exposed own-B only |',
        '| Original24-source-group main | NA, not passed | Unchanged |',
        '| Stage5C / SMC | FALSE | Not executed |']
    (run.PUBLIC/'world_model_gate_simple_controls.md').write_text('\n'.join(gates)+'\n');plot(a)
    print('Complete fixed control matrix reported without selecting a coefficient.')


if __name__=='__main__':main()
