"""Bind the exact timing bank's learned forecasts to the cached image screen."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_timing_capacity as study

run = study.run
HOME = ROOT/'outputs/publication_readiness_2026_09'


def main():
    combined = HOME/'dronecrowd_combined_forecaster_coverage_v1/analysis.json'
    cover = run.read(combined)
    for path,digest in cover['source_sha256'].items():
        if run.sha(ROOT/path)!=digest:
            raise ValueError('The combined coverage receipt changed')
    if cover['covered_predictors']!=18 or any(t['still_uncovered'] for t in cover['trials']):
        raise ValueError('Require verified training recording screen for all original predictors')
    screen = HOME/'movingdronecrowd_background_v2/registration.json'
    screened = {v['node'] for v in run.read(screen)['view_selection']}
    p = HOME/'dronecrowd_three_seed_bank_v1/analysis.json'
    probes = {t['key']:t for t in run.read(p)['trials']}
    old = []
    for f in run.read(run.parent.REG)['forecasts']:
        i = f['identity']
        name = f'{i["family"]}_fold{i["fold"]}_seed{i["seed"]}'
        t = probes[name]
        if (f['checkpoint_sha256']!=t['checkpoint']['sha256'] or f['sha256']!=t['prediction_sha256']
                or set(t['lineage']['recordings'])-screened):
            raise ValueError('Exact PCT/EqMotion bank not covered')
        old.append(dict(family=i['family'],seed=i['seed'],recordings=len(t['lineage']['recordings'])))
    kin = run.kin.PUBLIC
    train, predict = kin/'training_freeze.json',kin/'forecast_freeze.json'
    jobs = {j['seed']:j for j in run.read(run.kin.prior.REG)['jobs']}
    fits = {f['seed']:f for f in run.read(train)['fits'] if f['arm']=='self'}
    new = []
    for f in run.read(predict)['forecasts']:
        if f['arm']!='self':
            continue
        t,j = fits[f['seed']],jobs[f['seed']]
        ids = j['identity']
        roster = set(j['lineage']['recordings'])
        if (t['checkpoint']!=f['checkpoint'] or not t['complete'] or t['held_rows_sampled']
                or t['identity']['global_fit_rows_sha256']!=ids['train_ids_sha256']
                or t['identity']['registration_sha256']!=run.sha(run.kin.REG)
                or set(j['lineage']['groups'])!=set(ids['fit_groups']) or roster-screened):
            raise ValueError('Self-history predictor changed fitting rows or lacks image coverage')
        run.checked(f['checkpoint'])
        new.append(dict(family='self',seed=f['seed'],recordings=len(roster)))
    rows = old+new
    if {(r['family'],r['seed']) for r in rows}!={(f,k) for f in ('pct','eqmotion','self') for k in (17,29,43)} or len(rows)!=9:
        raise ValueError('Require the exact nine learned bank members')
    inputs = (combined,screen,p,run.parent.REG,run.kin.prior.REG,run.kin.REG,train,predict)
    result = run.stable(dict(members=rows,learned_members_covered=len(rows),fixed_causal_methods=5,
        inputs={str(x.relative_to(ROOT)):run.sha(x) for x in inputs},
        physical_independence_certified=False,external_inference_run=False,external_outcomes_opened=False,
        external_policy_frozen=False,scientific_roles_assigned=False))
    run.once(study.PUBLIC/'bank_lineage.json',result)
    print(rows)


if __name__=='__main__':
    main()
