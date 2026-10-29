"""Complete matched detector-input readout; fitting evidence is not deployment."""
from src.evaluation import m3w_completion_cost_readout as parent
from src.evaluation import m3w_multiaction_cost_readout as base

REPRESENTATIONS=('disabled','current','noise','history')
BANKS=('physical_only','physical_neural')


def expected_roster(contexts,seeds):
    expected=set()
    for context,items in contexts.items():
        names=[item['packet']['identity']['source'] for item in items]
        if len(names)!=4 or len(set(names))!=4:raise ValueError('Four fitting sources required')
        expected.update((context,source,seed,arm) for source in names for seed in seeds for arm in REPRESENTATIONS)
    return expected


def summarize(rows,contexts,cfg):
    keys=[(r['context'],r['source'],r['seed'],r['representation']) for r in rows]
    if len(keys)!=288 or len(set(keys))!=288 or set(keys)!=expected_roster(contexts,cfg['seeds']):
        raise ValueError('Complete288 original representation/source/seed groups required')
    output={}
    for arm in REPRESENTATIONS:
        group=[r for r in rows if r['representation']==arm]
        names=list(group[0]['result']['policies'])
        contrasts={}
        for control in REPRESENTATIONS:
            contrasts[control]={bank:{mode+'_'+bound:base.interval(group,
                lambda r,c=control,b=bank,m=mode,k=bound:r['representation_contrasts'][c][b][m][k],cfg)
                for mode in ('full','matched') for bound in ('known_pp','lower_pp','upper_pp')}
                for bank in BANKS}
        output[arm]=dict(groups=72,
            unique_fitting_recordings=len({(r['source'],key) for r in group for key in r['result']['recordings']}),
            safety={name:parent.safety(group,name) for name in names},
            accuracy={name:base.interval(group,lambda r,n=name:r['policies'][n]['known_ADE_gain_floor_percent'],cfg) for name in names},
            FDE={name:base.interval(group,lambda r,n=name:r['policies'][n]['known_FDE_gain_floor_percent'],cfg) for name in BANKS},
            representation_contrasts=contrasts,
            neural_bank_increment={mode+'_'+bound:base.interval(group,
                lambda r,m=mode,k=bound:r['comparisons']['physical_neural_minus_physical_only'][m][k],cfg)
                for mode in ('full','matched') for bound in ('known_pp','lower_pp','upper_pp')},
            subsets=parent.pooled_subsets(group))
    primary=output['history'];risk=primary['safety']['physical_neural']
    positive=lambda value:value['CI95'] is not None and value['CI95'][0]>0
    gates=dict(full288groups=True,complete_original_recording_risk=
        risk['supported_views']==risk['required_views'] and risk['supported_record_views']==risk['required_record_views'])
    for control in ('disabled','current','noise'):
        for mode in ('full','matched'):
            gates['history_minus_'+control+'_'+mode+'_lower_CI_positive']=positive(
                primary['representation_contrasts'][control]['physical_neural'][mode+'_lower_pp'])
    gates['mixed_minus_physical_lower_CI_positive']=positive(primary['neural_bank_increment']['full_lower_pp'])
    return dict(result_source='fresh_run_frozen_detector_cost_fitting_readout',representations=output,
        primary='history_physical_neural',fitting_gates=gates,fitting_advancement_gate=all(gates.values()),
        heads=144,primary_optimizer_updates=288000,scope='fitting_resubstitution_only',
        independent_confirmation=False,whole_pipeline_OOF=False,held_scored=False,independent_roles_opened=False,
        count_matching_utility_only=True,empty_risk_not_estimable=True,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False)
