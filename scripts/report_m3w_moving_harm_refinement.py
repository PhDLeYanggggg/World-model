"""Complete readout for the registered three-arm harm refinement experiment."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_moving_harm_refinement as run
from scripts.report_m3w_moving_amplitude_cost import interval,secondary_intervals
from scripts.report_m3w_moving_convex_amplitude import cost_ratio_diagnostic

source=run.source
PUBLIC=run.PUBLIC


def text_once(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text()!=text:raise ValueError('Changed immutable text artifact')
    else:
        with path.open('x')as stream:stream.write(text)


def factorization(actual,predicted):
    a,p=np.asarray(actual,float),np.asarray(predicted,float)
    if a.shape!=(5,)or p.shape!=(5,)or not np.isfinite([a,p]).all()or min(a.min(),p.min())<0:
        raise ValueError('Nonnegative aligned B H R ER EH masses required')
    moments={k:cost_ratio_diagnostic(a[i],p[i])for i,k in enumerate(('B','H','R','ER','EH'))}
    risk={}
    for key,h,r in (('all',1,2),('easy',4,3)):
        known=float(a[h]/a[r])if a[r]>0 else None
        estimated=float(p[h]/p[r])if p[r]>0 else None
        ratio=float(known/estimated)if known is not None and estimated is not None and estimated>0 else None
        hratio=moments[('B','H','R','ER','EH')[h]]['ratio']
        rratio=moments[('B','H','R','ER','EH')[r]]['ratio']
        decomposition=float(hratio/rratio)if hratio is not None and rratio is not None and rratio>0 else None
        if ratio is not None and decomposition is not None and not np.isclose(ratio,decomposition,rtol=1e-12):
            raise ValueError('Risk ratio factorization failed')
        risk[key]=dict(known_pooled_risk=known,predicted_on_same_known_rows_risk=estimated,
            known_over_predicted_risk=ratio,harm_factor_over_reference_factor=decomposition)
    return dict(moments=moments,risk=risk,pooled_fitting_diagnostic_not_macro_or_independent=True)


def usage(records):
    sites=sorted({r['site']for r in records});result=[]
    for arm in run.ARMS:
        rate=np.mean([np.mean([r['policies'][arm]['joint_free']['switch_rate']for r in records if r['site']==s])for s in sites])
        alpha=np.mean([np.mean([r['usage'][arm]['mean_amplitude_all_rows']for r in records if r['site']==s])for s in sites])
        empty=sum(r['matched_utility'][arm]['contrast']['known_pp']is None for r in records)
        result.append(dict(arm=arm,source_macro_switch_percent=100*float(rate),
            source_macro_amplitude_all_rows=float(alpha),empty_same_count_group_views=empty,
            required_same_count_group_views=len(records),full_scope_same_count_utility_estimable=empty==0))
    return result


def solvers(jobs):
    rows=[]
    if len(set(jobs))!=18:raise ValueError('All18held jobs required')
    for job in sorted(set(jobs)):
        for arm in run.ARMS:
            base=run.parent.PRIVATE if arm=='frozen'else run.PRIVATE
            tag='amplitude_bank'if arm=='frozen'else arm
            path=base/'allocation'/(job+'_'+tag+'.npz')
            with np.load(path,allow_pickle=False)as cache:meta=json.loads(str(cache['metadata_json']))
            if meta['bindings']['seconds']!=.25 or any(sum(meta['reason_counts'][p].values())!=meta['completed_queries']
                    for p in ('joint_free','joint_matched')):raise ValueError('Complete original-budget solver receipt required')
            rows.append(dict(arm=arm,job=job,queries=meta['completed_queries'],reasons=meta['reason_counts'],
                nonoptimal=meta['nonoptimal_queries'],sha256=source.sha(path)))
    return rows


def figures(summary,fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    public=PUBLIC/'figures';private=run.PRIVATE/'figure_previews'
    public.mkdir(exist_ok=True);private.mkdir(exist_ok=True)
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    fig,ax=plt.subplots(figsize=(8,3.5))
    names=list(summary['paired'])
    for i,name in enumerate(names):
        value=summary['paired'][name]['joint_free']['completion_lower_pp']
        if value['point']is None:ax.text(0,i,'not estimable')
        else:ax.errorbar(value['point'],i,xerr=[[value['point']-value['CI95'][0]],
            [value['CI95'][1]-value['point']]],fmt='o',capsize=3)
    ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.axvline(0,color='.5',lw=1)
    ax.set_xlabel('Direct paired completion-lower effect (pp), grouped development CI')
    ax.set_title('All registered contrasts, not independent confirmation')
    fig.tight_layout();fig.savefig(public/'paired_effects.svg');fig.savefig(private/'paired_effects.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,3.5))
    for arm in run.api.ARMS:
        traces=[run.api.base.read_checkpoint(source.checked(r['checkpoint']))['trace']for r in fits if r['arm']==arm]
        matrix=np.asarray([[r['weighted_normalized_harm_MSE']for r in t]for t in traces])
        ax.plot(range(1,21),np.median(matrix,0),label=arm)
        ax.fill_between(range(1,21),matrix.min(0),matrix.max(0),alpha=.1)
    ax.set_yscale('log');ax.set_xlabel('Tree fitting rounds');ax.set_ylabel('Own-weight normalized H/EH MSE')
    ax.set_title('Median and min-max over18fits per arm\nDifferent region weights: loss levels are not matched evaluation metrics')
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(public/'fitting_losses.svg');fig.savefig(private/'fitting_losses.png',dpi=150);plt.close(fig)


def main():
    summary=source.read(PUBLIC/'summary.json');verification=source.read(PUBLIC/'verification.json')
    freeze=source.read(PUBLIC/'training_freeze.json')
    fitting_check=source.read(PUBLIC/'fitting_check/analysis.json')
    held_check=source.read(PUBLIC/'held_cost_check/analysis.json')
    if (not verification['all72cost_outputs_and_policy_scalars_exact']or verification['summary_sha256']!=source.sha(PUBLIC/'summary.json')
            or freeze['complete_refiners']!=54 or not fitting_check['all72fitting_prediction_hashes_exact']
            or not held_check['all72causal_decisions_and_cost_predictions_unchanged']):
        raise ValueError('Complete all fits, held evaluation, common-region diagnosis and exact observed replay first')
    fits=[source.read(source.checked(ref))for ref in freeze['fits']]
    for row in fits:source.checked(row['checkpoint'])
    for arm in run.api.ARMS:
        if not source.read(PUBLIC/('full_refit_'+arm+'.json'))['exact']:raise ValueError('All three full refits required')
    records=[];jobs=[]
    for ref in summary['group_receipts']:
        row=source.read(source.checked(ref));records.extend(row['rows']);jobs.append(row['job']['key'])
    if len(records)!=138 or len({r['site']for r in records})!=23:raise ValueError('Complete supported source inventory required')
    fitting={}
    for arm in run.api.ARMS:
        rows=[r['own_selected_costs']for r in fits if r['arm']==arm]
        if len(rows)!=18:raise ValueError('All18fitting receipts per arm required')
        actual=np.sum([r['realized_selected_known_moments']for r in rows],0)
        predicted=np.sum([r['predicted_on_selected_known_moments']for r in rows],0)
        fitting[arm]=dict(factorization(actual,predicted),selected=sum(r['selected']for r in rows),
            selected_known=sum(r['selected_known']for r in rows),
            selected_unknown=sum(r['selected_unknown']for r in rows))
    primary=summary['paired']['adaptive_minus_fixed']['joint_free']['completion_lower_pp']
    primary_pass=primary['CI95']is not None and primary['CI95'][0]>0
    risk_pass=summary['safety']['adaptive_region']['joint_free']['full_original_risk_pass']
    decision=dict(primary_lower_positive=primary_pass,complete_original_risk_pass=risk_pass,
        original24group_primary='not_estimable_not_passed',independent_confirmation=False,
        new_deployment=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)
    solver=solvers(jobs);use=usage(records)
    for name,value in (('fitting_summary.json',fitting),('policy_usage.json',use),('solver_summary.json',solver),
            ('secondary_metrics.json',secondary_intervals(records,source.read(run.parent.CONFIG))),('decision.json',decision)):
        source.once(PUBLIC/name,source.stable(value))
    table=['|Contrast, joint-free|Known effect pp [95%CI]|Completion lower pp [95%CI]|',
        '|---|---:|---:|']
    for name,value in summary['paired'].items():
        table.append(f"|{name}|{interval(value['joint_free']['known_pp'])}|{interval(value['joint_free']['completion_lower_pp'])}|")
    safety=['|Arm|Supported views /348|All risk violations|Easy risk violations|Undefined selected-easy views|Worst known easy net %|',
        '|---|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:
        r=summary['safety'][arm]['joint_free'];worst=r['worst_known_easy_net_degradation_percent']
        safety.append(f"|{arm}|{r['supported_record_views']}|{r['known_all_risk_violations']}|{r['known_easy_risk_violations']}|{r['undefined_easy_views']}|{worst if worst is not None else 'not estimable'}|")
    executions={phase:source.read(PUBLIC/(phase+'_execution.json'))for phase in ('pilot','train','evaluate','verify')}
    executions['common_fitting_region_diagnosis']=source.read(PUBLIC/'fitting_check/execution.json')
    interruption=PUBLIC/'training_interruption.json'
    if interruption.exists():
        attempt=source.read(interruption)
        executions['interrupted_training_attempt']=attempt
        executions['total_training_phase_seconds_lower_bound']=attempt['last_heartbeat_elapsed_seconds_lower_bound']+executions['train']['seconds']
        executions['total_primary_tree_fit_calls_including_one_unsaved_recomputed_round']=1081
        executions['retained_primary_trees']=1080
        executions['additional_pilot_and_full_refit_tree_fits']=132
    source.once(PUBLIC/'runtime_summary.json',executions)
    text='# Matched upward harm refinement: complete readout\n\n'
    text+='Fresh54tree refiners /1080primary tree fits on18cached-verified neural heads. No new neural training. All18original fitting outputs and held baseline outputs replay exactly; all72held cost outputs and policy scalars replay, with72saved solver ledgers. Three complete first-job refits are exact.\n\n'
    text+='\n'.join(table)+'\n\n'+'\n'.join(safety)+'\n\n'
    text+=f'Registered supported-group primary positive: {primary_pass}. Complete original risk pass: {risk_pass}. Neither a positive pooled/known score nor a floor-only decision can replace the complete contract.\n\n'
    text+='All6policies,138source/producer/seed views and348repeated recording views are retained. Intervals use3000source-group bootstrap resamples after producer/seed averaging, not IID windows. The original24group primary remains not estimable; this exposed23group study cannot independently recertify deployment. Empty same-count contrasts stay not estimable. Exact replay checks saved solver bindings and feasibility, not bitwise MILP reoptimization.\n\n'
    text+='The registered paired effects use a common observed-label floor-error denominator. Completion bounds add worst-case path disagreement for wholly unknown rows; partially observed rows still use available-label ADE. These are not full12-step complete-trajectory ADE reductions or bounds on a ratio with an unobserved complete-data denominator. An unchanged floor has zero path change analytically, while unsupported selected-event risk remains not estimable.\n\n'
    text+='[All policies and intervals](summary.json), [secondary metrics](secondary_metrics.json), [fitting moment factorization](fitting_summary.json), [switches and empty controls](policy_usage.json), [solver ledger](solver_summary.json), [actual runtime](runtime_summary.json), [decision](decision.json).\n\n'
    text+='[Fixed choice-region fitting comparison](fitting_check/readout.md) separates estimation changes from abstention. [Post-decision held-cost diagnostics](held_cost_check/readout.md) retain every causal policy and recording without changing any model prediction or cached decision. These are diagnostics, not independent calibration guarantees.\n\n'
    text+='No new deployment or submission readiness. Native8/12steps and dataset-local pixel coordinates only; no verified seconds, metric scale, true3D or foundation claim. Stage5C and SMC remain off.\n'
    text_once(PUBLIC/'readout.md',text)
    per=['# Every supported source group\n','|Source|Adaptive minus fixed known pp|Adaptive minus fixed completion lower pp|','|---|---:|---:|']
    for site in sorted({r['site']for r in records}):
        rows=[r for r in records if r['site']==site]
        vals=[np.mean([r['paired']['adaptive_minus_fixed']['joint_free'][m]for r in rows])for m in ('known_pp','completion_lower_pp')]
        per.append(f'|{site}|{vals[0]:+.6f}|{vals[1]:+.6f}|')
    text_once(PUBLIC/'per_source.md','\n'.join(per)+'\n')
    figures(summary,fits)
    print(json.dumps(dict(primary=primary,decision=decision),indent=2))


if __name__=='__main__':main()
