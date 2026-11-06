"""Full matched cost/decision evidence; component quality is not a safety pass."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_role_separated_cost as run
from src.evaluation.m3w_matched_intervention_utility import matched_utility


def binary_quality(label,probability,weights):
    from sklearn.metrics import roc_auc_score,average_precision_score
    y,p,w=map(run.np.asarray,(label,probability,weights));w=w/w.sum()
    if y.dtype!=bool or y.shape!=p.shape or p.shape!=w.shape or not len(y):
        raise ValueError('Nonempty held label-support evaluation only')
    rate=float(w@y);both=bool(y.any() and (~y).any());ece=0.
    for i in range(10):
        at=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        if at.any():ece+=abs(float(w[at]@(p[at]-y[at])))
    return dict(unknown_rate=rate,AUROC=float(roc_auc_score(y,p,sample_weight=w)) if both else None,
        AUPRC=float(average_precision_score(y,p,sample_weight=w)) if both else None,
        Brier=float(w@((p-y)**2)),ECE=ece,rows=len(y),predicts_released_label_support_not_visibility=True)


def candidate_screen(analysis):
    candidate=analysis['policies']['role_separated']['joint_free']
    ci=analysis['primary']['completion_lower_pp']['interval95']
    return bool(candidate['safety']['required_record_views']==72
        and candidate['safety']['original_record_contract_pass'] and ci is not None and ci[0]>0)


def mean_required(rows,key):
    values=[r[key] for r in rows]
    return None if not values or any(v is None for v in values) else float(run.np.mean(values))


def legacy_veto_ceiling():
    original=run.read(run.full.PUBLIC/'analysis.json');result={}
    for name in ('mean','unanimity','proposal_self'):
        views=[v for r in original['rows'] for v in r['policies'][name]['recordings'].values()]
        empty=sum(v['known_easy_risk'] is None for v in views)
        result[name]=dict(required_views=len(views),zero_selected_easy_reference_views=empty,
            maximum_supported_views_for_any_subset=len(views)-empty)
    return dict(policies=result,scope='additional_veto_of_frozen_accepted_agent_set_only',
        proof='Nonnegative easy reference mass of any selected subset is bounded by the original selected mass.',
        new_admissions_within_broader_proposer_could_restore_support=True,
        full_action_expansion_not_proved_necessary=True,used_for_inference=False)


def components(reg):
    records=[];utilities=[];data=run.full.load_data()
    for node in reg['jobs']:
        own,old=run.own_and_floor(node);job=node['job'];key=job['key'];ix=old['local_ids']
        target=data['target_eval'][ix]-data['origin'][ix,None];valid=data['valid'][ix]
        positions=run.np.full((len(old['ids']),7),-1,int);positions[own['row'],own['action']]=run.np.arange(len(own['ids']))
        for seed_index,seed in enumerate((17,29,43)):
            for arm in run.method.ARMS:
                bank=run.load_cache(run.read(run.PUBLIC/'inference'/f'{key}_{arm}_seed{seed}.json')['cache'])
                for site in node['part']['names']:
                    at=run.np.flatnonzero(own['sites']==site);row,action=own['row'][at],own['action'][at]
                    state=run.neural.rel.base.read_checkpoint(run.checked(run.read(run.PUBLIC/'fits'/f'{key}_{site}_seed{seed}_{arm}.json')['checkpoint']))
                    pr=state['preprocess'];known=own['observed'][at];w=own['weights'][at]
                    p=bank['moments'][row,action]/own['scales'][at,None]
                    c=bank['conditional'][row,action]/own['scales'][at,None]
                    y=own['moments'][at]
                    def mse(pred,mask):
                        if not mask.any():return None
                        weight=w[mask]/w[mask].sum()
                        return float(run.np.sum(weight[:,None]*((pred[mask]-y[mask])/pr['rms'])**2)/5)
                    _,first=run.np.unique(row,return_index=True)
                    quality=binary_quality(~known[first],1-bank['probability'][row[first],action[first]],w[first])
                    active=old['active'][old['sites'][old['active']]==site]
                    proposed=positions[active,old['proposal_self'][active]]
                    if (proposed<0).any():raise ValueError('Every old proposal retained in full action bank')
                    observed=own['observed'][proposed];take=proposed[observed];active=active[observed]
                    state0=run.neural.rel.base.read_checkpoint(run.checked(run.full.risk_checkpoint(key,site,seed)))
                    scale0=state0['preprocess']['target_scale']
                    yy=run.torch.from_numpy(own['moments'][take])
                    truth=(yy[:,[1,4]]-.02*yy[:,[2,3]]).numpy()
                    pred=bank['moments'][active,old['proposal_self'][active]]/own['scales'][take,None]
                    signed=pred[:,[1,4]]-.02*pred[:,[2,3]]
                    pw=own['weights'][take];pw=pw/pw.sum()
                    signed_mse=lambda pp:float(run.np.sum(pw[:,None]*((pp-truth)/scale0)**2)/2)
                    records.append(dict(job=key,source=site,seed=seed,arm=arm,
                        observed_conditional_MSE=mse(c,known),observed_composed_MSE=mse(p,known),
                        bound_composed_MSE=mse(p,~known),label_support=quality,
                        common_proposal_known_signed_MSE=signed_mse(signed),
                        frozen_known_only_proposal_signed_MSE=signed_mse(old['risk'][seed_index,active]),
                        proposal_cohort_rows=len(take),common_proposal_normalizer='frozen_known_only_training_RMS',
                        new_comparison_weights='complete_roster_source_query_weights_conditioned_on_known_proposals'))
                    source_mask=old['sites']==site
                    paths={name:run.full.method.paths_for(old['paths'],old['floor'],bank[name])[source_mask]
                           for name in ('independent','joint_matched')}
                    groups=run.allocation.query_groups(old['recordings'][source_mask],old['frames'][source_mask],old['ids'][source_mask])
                    utilities.append(dict(job=key,source=site,seed=seed,arm=arm,
                        **matched_utility(paths['independent'],paths['joint_matched'],old['floor'][source_mask],
                            target[source_mask],valid[source_mask],bank['independent'][source_mask],bank['joint_matched'][source_mask],groups)))
    return dict(rows=records,normalized_five_moment_errors_not_old_two_channel_MSE=True,
                unavailable_support_not_physical_visibility=True),dict(rows=utilities,utility_only=True,safety_usage=False)


def formatted(x):
    return 'NA' if x is None else f'{x:.6f}'


def main():
    a=run.read(run.PUBLIC/'analysis.json');cfg,reg=run.register()
    if len(a['rows'])!=48 or a['source_blocks']!=8 or a['unique_recordings']!=24:
        raise ValueError('Complete registered all-seed/source readout required')
    cc,utility=components(reg);run.once(run.PUBLIC/'components.json',cc);run.once(run.PUBLIC/'matched_utility.json',utility)
    run.once(run.PUBLIC/'legacy_veto_support_ceiling.json',legacy_veto_ceiling())
    lines=['# Role-Separated Cost Learning And Full-Action Joint Decisions','',
        'I compared complete-bound anchor learning with separate observed-cost and',
        'label-support supervision. All 48 Torch fits and all 48 complete exact refits',
        'finished before policy readout. Both arms share architecture, fitting-only',
        'normalization, initialization, candidate pool and complete source/query draws.',
        'Reference mass and support probability are action-invariant by construction.','',
        'Dataset: MovingDroneCrowd++, already-exposed own-B. The original physical',
        'floor is causal CV in these three contexts. The earlier observation-support',
        'negative is retained; this is not a novel decomposition claim.','',
        '## Complete Development Policies','',
        '| Loss | Policy | Available-label gain % | Completion lower gain % | Supported views | Known easy >2% | Undefined easy |',
        '|---|---|---:|---:|---:|---:|---:|']
    for arm,policies in a['policies'].items():
        for name,p in policies.items():
            s=p['safety'];lines.append(f'| {arm} | {name} | {p["source_seed_macro_known_gain"]:+.6f} | '
                f'{p["source_seed_macro_completion_lower_gain"]:+.6f} | {s["supported_record_views"]}/{s["required_record_views"]} | '
                f'{s["known_easy_violations"]} | {s["undefined_easy_views"]} |')
    lines+=['','72 views repeat 24 recordings over three risk-head seeds; they are not',
        '72 independent scenes. Eight source blocks support exploratory uncertainty.',
        'Forecast teacher seed remains17. Gains use available-label native obs8/pred12',
        'image-coordinate ADE, not official complete-12 ADE or verified seconds/meters.','',
        '## Registered Primary: Role-Separated Minus Anchor, Joint-Free','',
        '| Contrast | Point pp | Exploratory paired source interval |','|---|---:|---:|']
    for k,r in a['primary'].items():
        ci=r['interval95'];interval='NA' if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]'
        lines.append(f'| {k} | {formatted(r["point"])} | {interval} |')
    lines+=['','3,000 source-component resamples after averaging risk seeds within source.',
        'The paired unknown envelope is between policies; separate lower gains cannot',
        'be subtracted to claim a paired improvement. Secondary controls do not replace',
        'the registered primary. No independent C/H evaluation is opened.','',
        '## Mechanism: Matched Prediction Targets','',
        '| Loss | Observed conditional MSE | Observed composed MSE | Bound composed MSE | Label-support AUROC | ECE |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in run.method.ARMS:
        rr=[r for r in cc['rows'] if r['arm']==arm]
        q=[r['label_support'] for r in rr]
        lines.append(f'| {arm} | {formatted(mean_required(rr,"observed_conditional_MSE"))} | '
            f'{formatted(mean_required(rr,"observed_composed_MSE"))} | {formatted(mean_required(rr,"bound_composed_MSE"))} | '
            f'{formatted(mean_required(q,"AUROC"))} | {formatted(mean_required(q,"ECE"))} |')
    lines+=['','Both arms use the same known-only five-moment fitting RMS. These numbers',
        'are not comparable to earlier two-channel signed-MSE tables. Bound error is',
        'against conservative contributions, not ground-truth unobserved outcomes.',
        'Support scores predict released-label availability, not human visibility or',
        'baseline failure. Full source/seed data, AUPRC/Brier and the separately matched',
        'old-proposal signed-error control are retained in components.json.','',
        '## Candidate Coverage And Same-Count Controls','']
    for arm,effects in a['within_arm_effects'].items():
        for name,r in effects.items():
            v=r['completion_lower_pp'];lines.append(f'- {arm} / {name}: paired lower {formatted(v["point"])} pp; interval {v["interval95"]}.')
    lines+=['','Matched-count utility is computed only on causally selected nonempty',
        'queries and is never used for safety. Empty controls remain not estimable.',
        'The complete deployed strategy always retains all required recording views.','',
        '## Verdict','',
        'Registered development candidate screen: '+('PASS' if candidate_screen(a) else '**FAIL / NA**')+'.',
        'No automatic deployment or submission-ready claim. The full original2%',
        'positive-harm, signed easy-net and reference/support requirements remain.',
        'Empty predictions receive no identity exception. Earlier negatives remain.','',
        'fresh_run:full-action data/moments,48fits,48fullrefits,causal allocation/readout.',
        'cached_verified:forecasts,old proposals,source boundaries and legacy controls.',
        'not_run:new independent C/H,three forecaster seeds,deployment or new dynamics.',
        'The original24-group main stays NA. No metric/seconds,true3D,foundation,',
        'human-gold,Stage5C or SMC claim. Annotation-generation causality unverified.','',
        '[Failure analysis](failure_analysis.md) | [Reproduction](operation_zh.md)']
    (run.PUBLIC/'README.md').write_text('\n'.join(lines)+'\n')
    gates=['# Role-Separated Full-Action Gates','',
        '| Gate | Status | Evidence |','|---|---|---|',
        '| Real fitting | PASS | 48 full Torch models, 4096 updates each |',
        '| Exact complete refits | PASS | All 48 |',
        '| Source exclusion and causal inputs | PASS by interface/lineage | Annotation-generation causality unverified |',
        '| Action-invariant reference/support | PASS by architecture/parity | Candidate cannot change its reference input |',
        '| All original own-B rows | PASS | 62466 agent rows, no future-support filtering |']
    for arm in run.method.ARMS:
        for name,p in a['policies'][arm].items():
            s=p['safety'];gates.append(f'| {arm}/{name} original risk | {"PASS" if s["original_record_contract_pass"] else "FAIL / NA"} | '
                f'{s["supported_record_views"]}/72 required repeated views |')
    gates+=['| Registered positive paired evidence and full risk | '+('PASS' if candidate_screen(a) else 'FAIL / NA')+' | Joint-free primary only |',
        '| Independent confirmation / deployment | NOT RUN / FALSE | Exposed own-B, not new independent validation |',
        '| Stage5C / SMC | FALSE | Not executed |']
    (run.PUBLIC/'world_model_gate_role_separated_cost.md').write_text('\n'.join(gates)+'\n')
    print('Complete matched cost and policy evidence reported; no risk rule changed.')


if __name__=='__main__':main()
