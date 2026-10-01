"""Independent order-statistic, recording isolation and delivery verification."""
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_component_calibration as run


def verify_margin(scores, fitted):
    assert fitted['recordings']==[r['recording'] for r in scores]
    counts=[]
    for j in range(4):
        values=sorted(r['scores'][j] for r in scores if r['scores'][j] is not None)
        counts.append(len(values))
        expected=min(1.,max(0.,values[math.ceil(.9*(len(values)-1))])) if values else 1.
        assert math.isclose(expected,fitted['margins'][j],rel_tol=1e-12,abs_tol=1e-12)
    assert fitted['component_recording_counts']==counts
    assert fitted['supported']==all(n>=1 for n in counts)
    assert not fitted['conformal_guarantee'] and fitted['quantile']==.9
    return 4


def main():
    cfg,reg=run.registration();pub=run.PUBLIC
    seal=json.loads((pub/'verification.json').read_text())
    assert seal['source_bindings']==reg['bindings']
    for k,h in seal['artifacts'].items():assert run.digest(pub/k)==h
    docs=run.docs();checks=folds=0
    for record in docs.values():
        part=record['identity']['partition']
        assert set(part['train_recordings']).isdisjoint(part['validation_recordings'])
        scores=record['record_scores'];names=[x['recording'] for x in scores]
        assert set(names)==set(part['validation_recordings']) and len(names)==len(set(names))
        checks+=verify_margin(scores,record['final'])
        for fold in record['folds']:
            held=fold['held_recording']
            assert held not in fold['calibration']['recordings']
            checks+=verify_margin([r for r in scores if r['recording']!=held],fold['calibration'])
            folds+=1
    s=json.loads((pub/'summary.json').read_text())
    d=json.loads((pub/'readout.json').read_text())
    frozen={r['view']:r for r in json.loads((pub/'decision_freeze.json').read_text())['rows']}
    assert len(frozen)==216 and len(d['rows'])==2376
    for row in d['rows']:
        assert all(row[k]==frozen[row['view']][k] for k in ('site','source','head_seed'))
    state=json.loads((ROOT/'research_state.json').read_text())['cvpr2027_research_track'][run.NAME]
    expected=dict(calibration_models_completed=72,recording_oof_folds=folds,
        full_calibrators_unsupported=72-s['full_calibrators_supported'],
        source_oof_screen_pass={m:v['oof_screen_pass'] for m,v in s['source_screens'].items()},
        matched_ADE_improvement_percent=s['matched_ADE_improvement']['mean'],
        matched_ADE_nominal_CI95=s['matched_ADE_improvement']['CI95'],
        joint_screen_easy_risk=s['policies']['joint_screen']['easy_risk'],
        joint_screen_ADE_gain_percent=s['policies']['joint_screen']['all_gain_floor']['mean'],
        joint_screen_intervention_rate=s['policies']['joint_screen']['intervention_rate']['mean'],
        independent_metric_checks=d['independent_metric_checks'],query_count_checks=d['query_count_checks'],
        parent_metric_views_exact=d['parent_metric_views_exact'],verification_sha256=run.digest(pub/'verification.json'))
    for k,v in expected.items():assert state[k]==v,k
    assert cfg['risk_budget']==.02 and not cfg['threshold_search']
    assert not state['independent_roles_read'] and not state['deployment_changed']
    assert not state['stage5c_executed'] and not state['smc_enabled']
    paths=[ROOT/'README.md',ROOT/'README_RESULTS.md',ROOT/'research_state.json',Path(__file__),
        pub/'results.md',pub/'failure_analysis.md',pub/'operation.md',pub/'verification.json']
    run.immutable(pub/'delivery_verification.json',dict(status='verified',
        independent_margin_scalar_checks=checks,source_held_recording_isolation_checks=folds,
        calibration_models=72,state_value_checks=len(expected),metric_rows=2376,
        bindings={str(p.relative_to(ROOT)):run.digest(p) for p in paths},
        independent_confirmation=False,deployment_changed=False,full_legacy_suite='not_run'))
    print(json.dumps(dict(verified=True,margin_checks=checks,isolated_folds=folds)))


if __name__=='__main__':main()
