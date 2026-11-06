"""Report full-policy failure without erasing positive trajectory information."""
import csv
import io
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_complete_donor_policy as run


def category(view):
    observed=any(view[k] is not None and view[k]>.02 for k in
                 ('known_all_risk','known_easy_risk','known_easy_net_degradation'))
    completion=any(view[k] is not None and view[k]>.02 for k in
                   ('completion_all_risk_upper','completion_easy_risk_upper'))
    if observed:return 'observed_harm_violation'
    if completion:return 'unknown_completion_bound_violation'
    if not view['supported']:return 'missing_reference_support'
    return 'passes_original_record_contract'


def validate(a):
    if a['complete_own_B_rows']!=62466 or len(a['rows'])!=8:
        raise ValueError('Original complete exposed population required')
    if any(a[k] for k in ('identity_certification','new_outer_C_H_readout','independent_confirmation',
                          'deployment_changed','submission_ready','stage5c_executed','smc_enabled')):
        raise ValueError('Do not turn development readout into deployment')


def number(v,precision=6):
    return 'NA' if v is None else f'{v:.{precision}f}'


def main():
    p=run.PUBLIC;a=run.read(p/'analysis.json');validate(a)
    lines=['# Complete Policy Readout: Trajectory Gains, Original Risk Contract Fails','',
        'I reconstructed the entire exposed own-B population and evaluated the',
        'frozen policies with the original per-record risk/support evaluator.',
        'This is fresh complete-policy development evidence, not a new held test.','',
        '**Neither mean nor unanimity passes the original complete risk contract.**',
        'No deployment upgrade. Empty support stays NA; no identity exemption.','',
        '## Complete Population','',
        '62,466 past-eligible rows, 8 source components, 24 recordings. There are 60,701',
        'rows with some labels, 40,043 complete 12-label rows, 20,658 partial rows and',
        '1,765 wholly unknown rows. The frozen proposer selects 49,335 rows:',
        '48,061 known and 1,274 unknown. No future mask enters any choice.','',
        '## Full Policy Results','',
        '| Policy | Available-label ADE gain % | Completion lower gain % | Selected | Unknown selected | Supported record views | Easy reference NA |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in a['policies'].items():
        s=m['safety']
        lines.append(f'| {name} | {m["source_macro_known_ADE_gain"]:+.6f} | '
            f'{m["source_macro_completion_lower_gain"]:+.6f} | {m["selected"]} | {m["unknown_selected"]} | '
            f'{s["supported_record_views"]}/{s["required_record_views"]} | {s["undefined_easy_views"]} |')
    lines+=['','Gains are equal-source macro versus each parent training-defined floor.',
        'ADE uses available labels, not an official complete12-step ADE estimate.',
        'The lower gain subtracts causal path-disagreement bounds for unknown labels.',
        'Unknown samples are not treated as safe. The baseline has undefined selected',
        'reference mass; its 0/24 support is NA, not proof that baseline motion is unsafe.','',
        '## Actual Violations Versus Missing Support','',
        '| Policy | Known all >2% | Known easy >2% | Completion all >2% | Completion easy >2% | Easy-net >2% |',
        '|---|---:|---:|---:|---:|---:|']
    categories={}
    records=[]
    for name,m in a['policies'].items():
        s=m['safety']
        lines.append(f'| {name} | {s["known_all_violations"]} | {s["known_easy_violations"]} | '
            f'{s["completion_all_violations"]} | {s["completion_easy_violations"]} | {s["easy_net_violations"]} |')
        counts={}
        for row in a['rows']:
            for recording,v in row['policies'][name]['recordings'].items():
                reason=category(v);counts[reason]=counts.get(reason,0)+1
                records.append(dict(policy=name,job=row['job'],source=row['source'],recording=recording,reason=reason,**v))
        categories[name]=counts
    lines+=['','Violation columns can overlap; they are not additive categories. A missing',
        'reference is separately NA. Benefit cannot cancel positive harm in the',
        'all/easy checks. Zero easy-net violations therefore do not imply safety.','',
        '## Paired Trajectory Contrast Versus Forest','',
        '| Policy | Known gain difference pp | Conservative completion lower pp | Exploratory source interval for lower pp |',
        '|---|---:|---:|---:|']
    for name,c in a['paired_vs_forest'].items():
        lower=c['completion_lower_pp'];ci=lower['interval95']
        lines.append(f'| {name} | {number(c["known_pp"]["point"])} | {number(lower["point"])} | '
                     f'[{number(ci[0])}, {number(ci[1])}] |' if ci else f'| {name} | NA | NA | NA |')
    lines+=['','3,000 paired source-component bootstrap replicates on 8 exposed sources, not',
        'overlapping windows. These intervals describe full-population trajectory',
        'contrasts, not the earlier signed-decision-cost intervals or a safety guarantee.',
        'No independent confirmation or complete three-forecaster-seed claim follows.','',
        '## Causal Matching And Verification','',
        'Same-query/action-count controls are utility-only. Their eligibility uses',
        'frozen choices and causal utility margins; empty controls remain NA.',
        'All old known features and decisions reproduce. One original scalar float',
        'check failed at a batch tail; the retained v2 amendment checks training-scaled',
        'float32 roundoff plus exact per-channel and mean/unanimity decisions. No model,',
        'inference math, threshold or risk rule changed to pass that engineering check.','',
        '## What Remains','',
        'The risk heads were trained mainly on known outcomes and can admit censored',
        'tracks. Consensus also discards selected easy reference mass. Next test',
        'complete-risk donor supervision using the existing conservative unknown',
        'envelope targets, and model reference support explicitly. Keep the same',
        'full-policy evaluator; do not repair these failures by ignoring records,',
        'changing 2% or selecting thresholds on test outcomes.','',
        'fresh_run: scoped reconstruction, full causal inference and risk/utility metrics.',
        'cached_verified: weights, forecasts, original splits and known-row results.',
        'not_run: new training, new outer C/H, independent test or deployment.',
        'Original 24-group main NA, old failures and Stage5C/SMC prohibitions remain.',
        'Native 8/12 image-coordinate silver labels, not verified meters/seconds,',
        'true 3D or foundation. Source annotation-generation causality is unverified.','',
        '[Record diagnostics](record_risk.csv) | [Failure analysis](failure_analysis.md)',
        '| [Reproduction](operation_zh.md)']
    (p/'README.md').write_text('\n'.join(lines)+'\n')
    out=io.StringIO();keys=['policy','job','source','recording','reason','rows','selected','unknown_selected',
        'known_all_risk','known_easy_risk','completion_all_risk_upper','completion_easy_risk_upper',
        'known_easy_net_degradation','supported']
    writer=csv.DictWriter(out,fieldnames=keys,lineterminator='\n');writer.writeheader()
    for row in records:writer.writerow({k:row[k] for k in keys})
    (p/'record_risk.csv').write_text(out.getvalue())
    structural=sum(v['known_easy_net_degradation'] is None for row in a['rows']
                   for v in row['policies']['physical_floor']['recordings'].values())
    run.once(p/'failure_taxonomy.json',dict(categories=categories,
        recordings_without_any_observed_easy_reference=structural,
        categories_mutually_exclusive_by_observed_then_completion_then_support=True))
    gates=['# Complete-Policy Gates','',
        '| Gate | Status | Evidence |','|---|---|---|',
        '| Complete exposed population | PASS | 62,466 rows, 8 sources, 24 recordings |',
        '| Past-only frozen inference | PASS by interface/parity | No future masks in choices; annotation-generation causality unverified |',
        '| Known feature/decision parity | PASS | All features and decisions reproduce; float amendment retained |']
    for name in ('mean','unanimity'):
        s=a['policies'][name]['safety']
        gates.append(f'| {name} original risk/support | {"PASS" if s["original_record_contract_pass"] else "FAIL / NA"} | '
                     f'{s["supported_record_views"]}/{s["required_record_views"]} record views |')
    gates+=['| Independent confirmation | NOT RUN | Exposed own-B only |',
        '| Deployment / submission | FALSE | Complete safety not established |',
        '| Stage5C / SMC | FALSE | Not executed |']
    (p/'world_model_gate_complete_donor_policy.md').write_text('\n'.join(gates)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['forest64','seed17','mean','unanimity'];x=run.np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for key,offset,color,label in [('source_macro_known_ADE_gain',-.18,'#2563eb','Available labels'),
        ('source_macro_completion_lower_gain',.18,'#15803d','Completion lower')]:
        axes[0].bar(x+offset,[a['policies'][n][key] for n in names],width=.36,color=color,label=label)
    axes[0].set(xticks=x,xticklabels=names,ylabel='Gain versus floor (%)',title='Complete exposed population')
    axes[0].legend(fontsize=8)
    axes[1].bar(x,[a['policies'][n]['safety']['supported_record_views'] for n in names],color='#ca8a04')
    axes[1].axhline(24,color='#dc2626',linestyle='--',label='All 24 required')
    axes[1].set(xticks=x,xticklabels=names,ylim=(0,26),ylabel='Supported record views',title='Original 2% and support contract')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    directory=run.PRIVATE/'figures';directory.mkdir(exist_ok=True)
    fig.savefig(directory/'gain_and_original_risk.png',dpi=170);fig.savefig(directory/'gain_and_original_risk.pdf');plt.close(fig)
    print('Complete policy metrics reported; original risk/support gate remains unpassed.')


if __name__=='__main__':main()
