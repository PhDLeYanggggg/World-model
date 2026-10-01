"""Verify sealed numeric evidence and its research-state/narrative delivery."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_source_forest as run


def main():
    pub=run.PUBLIC;cfg,reg=run.registration()
    seal=json.loads((pub/'verification.json').read_text())
    assert seal['source_bindings']==reg['bindings']
    for name,h in seal['artifacts'].items():
        assert run.digest(pub/name)==h
    amendment=json.loads((pub/'resource_amendment.json').read_text())
    assert run.digest(ROOT/'scripts/resume_m3w_source_forest_resource_checked.py')==amendment['wrapper_sha256']
    assert run.digest(pub/'resource_amendment.md')==amendment['amendment_sha256']
    s=json.loads((pub/'summary.json').read_text())
    state=json.loads((ROOT/'research_state.json').read_text())
    r=state['cvpr2027_research_track'][run.NAME]
    check={
        'diagnostic_primary_signed_MSE_difference':s['diagnostic_primary']['mean'],
        'diagnostic_primary_nominal_CI95':s['diagnostic_primary']['CI95'],
        'matched_ADE_improvement_percent':s['matched_ADE_improvement']['mean'],
        'matched_ADE_nominal_CI95':s['matched_ADE_improvement']['CI95'],
        'transferred_signed_MSE_difference':s['transferred_signed_MSE_difference']['mean'],
        'source_validation_forest_better_than_neural':s['source_forest_better_than_neural'],
        'source_screen_pass':s['source_forest_screen_pass'],
        'unique_fits_completed':s['unique_fits'],
        'cumulative_fit_seconds':s['cumulative_fit_seconds'],
        'checkpoint_bytes':s['checkpoint_bytes'],
        'independent_metric_checks':s['independent_metric_checks'],
        'independent_quality_checks':s['independent_quality_checks'],
        'query_count_checks':s['query_count_checks'],
        'verification_sha256':run.digest(pub/'verification.json')}
    p=s['policies']['screened']
    for key,metric in [('screened_ADE_gain_floor_percent','all_gain_floor'),
                       ('screened_hard_gain_floor_percent','hard_gain_floor'),
                       ('screened_intervention_rate','intervention_rate')]:
        check[key]=p[metric]['mean']
    check.update(worst_whole_easy_degradation_percent=p['worst_whole_easy_degradation_percent'],
        screened_defined_risk_views=p['easy_risk']['defined'],
        screened_undefined_risk_views=p['easy_risk']['undefined'],
        screened_easy_risk_failures=p['easy_risk']['violating'],
        screened_all_risk_failures=p['all_risk']['violating'],
        screened_worst_easy_risk=p['easy_risk']['worst'],
        screened_unknown_intervention_occurrences=p['easy_risk']['unknown_interventions'])
    for k,v in check.items():
        assert r[k]==v,k
    rows=json.loads((pub/'readout.json').read_text())['rows']
    failures=[x for x in rows if x['policy']=='screened'
              and (x['metric']['selected_easy_positive_harm_ratio'] or 0)>.02+1e-10]
    assert len(failures)==6 and {x['seed'] for x in failures}=={43}
    frozen={x['view']:x for x in json.loads((pub/'decision_freeze.json').read_text())['rows']}
    for row in rows:
        for k,v in frozen[row['view']].items():
            assert row[k]==v
    docs=run.forest_docs()
    assert len(docs)==72 and all(x['trees']==128 for x in docs.values())
    assert sum(x['checkpoint_bytes'] for x in docs.values())==s['checkpoint_bytes']
    assert all(not set(x['partition']['train_recordings'])&set(x['partition']['validation_recordings'])
               for x in docs.values())
    assert not r['independent_roles_read'] and not r['deployment_changed']
    assert not r['stage5c_executed'] and not r['smc_enabled']
    paths=[ROOT/'README.md',ROOT/'README_RESULTS.md',ROOT/'research_state.json',Path(__file__),
        ROOT/'scripts/resume_m3w_source_forest_resource_checked.py',pub/'results.md',pub/'failure_analysis.md',
        pub/'verification.json',pub/'operation.md']
    run.immutable(pub/'delivery_verification.json',dict(status='verified',
        numeric_seal_sha256=run.digest(pub/'verification.json'),state_value_checks=len(check),
        frozen_views=216,metric_rows=len(rows),checkpoint_hashes_verified=len(docs),
        screened_easy_failures_all_shared_seed43=True,source_validation_recording_disjoint=True,
        bindings={str(p.relative_to(ROOT)):run.digest(p) for p in paths},
        independent_confirmation=False,deployment_changed=False,full_legacy_suite='not_run'))
    print('Verified numeric seal,72 checkpoints,216 frozen views, research-state values and delivery hashes')


if __name__=='__main__':main()
