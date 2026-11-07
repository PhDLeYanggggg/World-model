"""Replay complete recording risks and fitting-selected baseline comparisons on A."""
import argparse
from pathlib import Path
import os
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_support_anchored_cost as anchor
from src.evaluation import m3w_recording_forecast_comparison as readout
from src.world_model.m3w_european_source_forecast import BASELINES, baseline_numpy

old, np, read, sha, artifact, checked, once, stable = (anchor.old, anchor.np, anchor.read, anchor.sha,
    anchor.artifact, anchor.checked, anchor.once, anchor.stable)
PUBLIC = old.HOME/'moving_recording_forecast_replay_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(readout.__file__), PUBLIC/'protocol.md',
    ROOT/'tests/test_m3w_recording_forecast_comparison.py')
CV, OLS = BASELINES[1], BASELINES[4]
PAIRS = {**{key+'_vs_cv':(CV,key,CV) for key in (*BASELINES,'pct_raw','eqmotion_raw',*anchor.model.ARMS) if key != CV},
    'pct_vs_ols4':(OLS,'pct_raw',OLS), 'eqmotion_vs_ols4':(OLS,'eqmotion_raw',OLS),
    'reference_vs_ols4':(OLS,'reference',OLS), 'learned_anchor_vs_ols4':(OLS,'learned_anchor',OLS),
    'eqmotion_vs_pct':('pct_raw','eqmotion_raw',CV),
    'learned_anchor_vs_reference':('reference','learned_anchor',CV)}


def register():
    parent = [anchor.PUBLIC/f for f in ('registration.json','freeze.json','analysis.json','replay_execution.json')]
    proof = read(parent[-1])
    if proof['artifact']['sha256'] != sha(parent[-2]) or not proof['probabilities_choices_and_parent_exact']:
        raise ValueError('Require exact prior action replay')
    forecasts = [b for b in read(old.PARENTS[-2])['prediction_receipts'] if b['identity']['fold'] == 2]
    if len(forecasts) != 6 or any(b['identity']['baseline_index'] != 4 for b in forecasts):
        raise ValueError('Require the same fitting-selected baseline in all matched producer cells')
    inputs = [*parent, old.PUBLIC/'freeze.json', *old.PARENTS, Path(readout.policy.__code__.co_filename)]
    once(REG, stable(dict(files=[artifact(p) for p in FILES], parents=[artifact(p) for p in inputs],
        forecasts=forecasts, comparisons=PAIRS, expected_rows=125540, expected_sources=14, expected_recordings=34,
        actual_seeds=old.SEEDS, source_bootstrap_draws=3000, source_bootstrap_seed=73107,
        original_risk_reference=CV, fitting_selected_accuracy_comparator=OLS, original_risk_budget=.02,
        new_inference=False, new_training=False, external_readout=False, deployment_changed=False)))
    return forecasts


def read_banks(refs, seed, ids, sites):
    forecasts = {}
    for r in refs:
        ident = r['identity']
        if ident['seed'] != seed: continue
        if set(sites)&set(ident['fit_groups']) or not set(sites) <= set(ident['excluded_groups']):
            raise ValueError('Producer training exposed evaluation source')
        with np.load(checked(r), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            forecasts[ident['family']] = z['relative_prediction'].astype(float)
    if set(forecasts) != {'pct','eqmotion'}:
        raise ValueError('Missing matched forecast family')
    return forecasts


def score(banks):
    data = old.prior.arrays(('global_ids','history','origin','sites','queries','recordings'))
    labels = old.prior.arrays(('target_eval','valid'))
    s, r, q, ids = (data[k] for k in ('sites','recordings','queries','global_ids'))
    if (len(ids), len(np.unique(s)), len(np.unique(r)), len(np.unique(q))) != (125540,14,34,1613):
        raise ValueError('Keep the entire fixed A roster')
    names = np.unique(s); ref = read(anchor.PUBLIC/'analysis.json')
    original = {(x['seed'],x['source']):x for x in ref['sources']}
    actions = {(x['seed'],x['held_source']):x for x in read(anchor.PUBLIC/'freeze.json')['fits']}
    cut_bank = read(old.PARENTS[4])['cuts']
    cuts = {site:cut_bank[str(x['job']['producer_fold'])] for x in read(old.PARENTS[3])['forecasts'] for site in x['job']['donor_sites']}
    records, sources = [], []; max_delta = 0.
    for seed in old.SEEDS:
        candidates = read_banks(banks, seed, ids, s)
        for site in names:
            ix = np.flatnonzero(s == site)
            saved = old.joblib.load(checked(actions[seed,site]['checkpoint']))
            np.testing.assert_array_equal(saved['ids'], ids[ix])
            masks = saved['choices']
            if masks.shape != (len(anchor.model.ARMS),len(ix)) or masks.dtype != bool:
                raise ValueError('Missing frozen complete actions')
            for g in old.prior.groups(q[ix]):
                if not (masks[:,g].sum(1) == min(2,len(g)-1)).all() or len(np.unique(r[ix[g]])) != 1:
                    raise ValueError('Cannot alter count or split a query over recordings')
            per_source = []
            for record in np.unique(r[ix]):
                loc = r[ix] == record; take = ix[loc]
                h = data['history'][take].astype(float); origin = data['origin'][take]
                np.testing.assert_array_equal(h[:,-1],origin)
                paths = {name:baseline_numpy(h,k)-origin[:,None] for k,name in enumerate(BASELINES)}
                floor = np.arange(1,13)[None,:,None]*(h[:,-1]-h[:,-2])[:,None]
                paths.update({CV:floor, 'pct_raw':candidates['pct'][take], 'eqmotion_raw':candidates['eqmotion'][take]})
                for arm, chosen in zip(anchor.model.ARMS,masks[:,loc]):
                    paths[arm] = np.where(chosen[:,None,None],paths['eqmotion_raw'],floor)
                cut = cuts[site]
                out = readout.score_recording(paths, PAIRS, labels['target_eval'][take]-origin[:,None],
                    labels['valid'][take], q[take], str(record), easy_cut=cut['easy'], hard_cut=cut['hard'])
                out.update(source=str(site),seed=seed)
                records.append(out); per_source.append(out)
            merged = {key:readout.merge_comparisons([row['comparisons'][key] for row in per_source]) for key in PAIRS}
            truth = original[seed,site]['arms']['learned_anchor']
            check = merged['learned_anchor_vs_reference']
            for key in ('known_pp','completion_lower_pp','completion_upper_pp'):
                delta = abs(check[key]-truth['paired_'+key]); max_delta = max(max_delta,delta)
                if delta > 1e-10: raise ValueError('Recording accumulation differs from original full-source evaluation')
            sources.append(dict(source=str(site), seed=seed, comparisons=merged))
        print(dict(seed=seed,record_views=len(records),sources=len(sources)),flush=True)
    if len(records)!=102 or len(sources)!=42:
        raise ValueError('No recording view or seed may be dropped')
    draws = np.random.default_rng(73107).integers(len(names),size=(3000,len(names)))
    def bootstrap(pair,field):
        values = [[v['comparisons'][pair][field] for v in sources if v['source']==site] for site in names]
        if any(x is None for v in values for x in v): return None
        x = np.asarray(values).mean(1)
        return dict(point=float(x.mean()),interval95=np.quantile(x[draws].mean(1),[.025,.975]).tolist())
    paired = {name:{k:bootstrap(name,k) for k in ('known_pp','completion_lower_pp','completion_upper_pp')} for name in PAIRS}
    risk = {}
    for key in records[0]['metrics']:
        views = [v['metrics'][key] for v in records]; scores = [v['risk'] for v in views]
        counts = {}
        for field in ('known_all_risk','known_easy_risk','completion_all_risk_upper',
                      'completion_easy_risk_upper','known_easy_net_degradation'):
            x = [v[field] for v in scores]; valid = [y for y in x if y is not None]
            counts[field] = dict(undefined=len(x)-len(valid),over_2=sum(y>.02 for y in valid),worst=max(valid,default=None))
        risk[key] = dict(views=len(scores),supported=sum(v['supported'] for v in scores),
            worst_recording_p95_ADE=max(v['p95_known_ADE_pixel'] for v in views if v['p95_known_ADE_pixel'] is not None),
            worst_recording_ADE=max(v['known_ADE_pixel'] for v in views if v['known_ADE_pixel'] is not None),
            switched=sum(v['selected'] for v in scores),diagnostics=counts)
    once(PUBLIC/'analysis.json',stable(dict(result_source='fresh_recording_readout_cached_verified_paths',
        source_views=sources, recording_views=records, rows_per_seed=len(ids), paired=paired, risk=risk,
        streamed_original_anchored_source_max_abs_difference_pp=max_delta,
        safety_rule_unchanged=True, scientific_roles_assigned=False, new_inference=False,
        new_training=False, external_label_access=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('phase', choices=('register','run','replay'))
    args = p.parse_args(); start = time.monotonic(); banks = register()
    if args.phase == 'register': return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit frozen readout before scoring')
    if args.phase=='replay' and not (PUBLIC/'run_execution.json').exists():
        raise ValueError('Original completed run required')
    score(banks)
    once(PUBLIC/(args.phase+'_execution.json'),dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        artifact=artifact(PUBLIC/'analysis.json'),original_rules_unchanged=True))


if __name__ == '__main__':
    main()
