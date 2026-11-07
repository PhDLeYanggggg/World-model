"""Exact free-count and fixed-count label-aware diagnostics on the frozen A bank."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_recording_forecast_replay as parent
from src.evaluation import m3w_recording_switch_ceiling as api

old,np,read,sha,artifact,once,stable = (parent.old,parent.np,parent.read,parent.sha,parent.artifact,parent.once,parent.stable)
PUBLIC = old.HOME/'moving_recording_switch_ceiling_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_recording_switch_ceiling_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__),Path(api.__file__),PUBLIC/'protocol.md',ROOT/'tests/test_m3w_recording_switch_ceiling.py')


def registration():
    prior = parent.PUBLIC/'analysis.json'; proof = parent.PUBLIC/'replay_execution.json'
    if read(proof)['artifact']['sha256'] != sha(prior) or not read(proof)['original_rules_unchanged']:
        raise ValueError('Original recording metrics must have exact replay')
    value = stable(dict(files=[artifact(p) for p in FILES], parents=[artifact(p) for p in
        (parent.REG,Path(parent.__file__),prior,proof,*old.PARENTS)], budget=.02,
        matched_solver_seconds=15., actual_seeds=old.SEEDS, source_bootstrap_draws=3000,
        expected_record_views=102, expected_recordings=34, expected_sources=14,
        label_aware_posthoc_only=True, deployable=False, new_training=False, new_inference=False,
        external_label_access=False, stage5c_executed=False, smc_enabled=False))
    once(REG,value)
    return read(parent.REG)['forecasts']


def run(banks,replay):
    data = old.prior.arrays(('global_ids','history','origin','sites','queries','recordings'))
    labels = old.prior.arrays(('target_eval','valid'))
    ids,sites,recs,q = (data[k] for k in ('global_ids','sites','recordings','queries'))
    if (len(ids),len(np.unique(recs)),len(np.unique(sites)),len(np.unique(q))) != (125540,34,14,1613):
        raise ValueError('The complete A population is fixed')
    cuts = read(old.PARENTS[4])['cuts']
    cut = {s:cuts[str(ref['job']['producer_fold'])] for ref in read(old.PARENTS[3])['forecasts'] for s in ref['job']['donor_sites']}
    prior_risk = {(v['seed'],v['recording']):v for v in read(parent.PUBLIC/'analysis.json')['recording_views']}
    rows = []
    for seed in old.SEEDS:
        pred = parent.read_banks(banks,seed,ids,sites)['eqmotion']
        for rec in np.unique(recs):
            ix = np.flatnonzero(recs==rec); names = np.unique(sites[ix])
            if len(names)!=1:raise ValueError('One source per recording required')
            site = str(names[0]); path = PRIVATE/'records'/f'{seed}_{int(rec):03d}.json'
            identity = dict(registration_sha256=sha(REG),seed=seed,recording=str(rec),source=site)
            if replay or not path.exists():
                h = data['history'][ix].astype(float); origin = data['origin'][ix]
                np.testing.assert_array_equal(h[:,-1],origin)
                cv = np.arange(1,13)[None,:,None]*(h[:,-1]-h[:,-2])[:,None]
                truth = labels['target_eval'][ix]-origin[:,None]
                outcome = api.inspect(cv,pred[ix],truth,labels['valid'][ix],q[ix],str(rec),
                    cut[site]['easy'],cut[site]['hard'],seconds=15.)
                prior = prior_risk[seed,str(rec)]
                if (outcome['rows'] != prior['rows'] or outcome['queries'] != prior['queries']
                        or outcome['matched_required_switches'] != prior['metrics']['learned_anchor']['risk']['selected']):
                    raise ValueError('Fixed-count oracle must match original recorded switches and rows')
                once(path,stable(dict(identity=identity,**outcome)))
            v = read(path)
            if v['identity']!=identity:raise ValueError('Cannot resume mismatched result')
            rows.append(v)
            old.prior.base.source.atomic(PRIVATE/'heartbeat.json',dict(pid=os.getpid(),complete=len(rows),total=102,
                seed=seed,recording=str(rec),phase='replay' if replay else 'run'))
            if len(rows)%8==0:print(dict(complete=len(rows),total=102,seed=seed,recording=str(rec)),flush=True)
    if len(rows)!=102:raise ValueError('All views required')
    sites = sorted({v['identity']['source'] for v in rows})
    draws = np.random.default_rng(73107).integers(len(sites),size=(3000,len(sites)))
    def view_state(r,arm):
        val = r[arm]
        if arm=='free_count': return 'optimal' if val else 'infeasible'
        s = r['solver']
        if s['infeasible']:return 'infeasible'
        if val and val['supported']:return 'optimal' if s['optimal'] else 'feasible_unproven'
        return 'unresolved'
    def summary(arm):
        status = [view_state(v,arm) for v in rows]
        source = {site:[] for site in sites}; gain = None
        if all(x=='optimal' for x in status):
            for seed in old.SEEDS:
                for site in sites:
                    take = [v for v in rows if v['identity']['source']==site and v['identity']['seed']==seed]
                    source[site].append(100*sum(v[arm]['completion_gain_mass'] for v in take)/sum(v['reference_mass'] for v in take))
            a = np.array([np.mean(source[s]) for s in sites]); gain = dict(point=float(a.mean()),
                interval95=np.quantile(a[draws].mean(1),[.025,.975]).tolist())
        return dict(status_counts=dict(Counter(status)),supported_views=sum(x in ('optimal','feasible_unproven') for x in status),
            positive_supported_views=sum(bool(v[arm] and v[arm]['supported'] and v[arm]['completion_gain_mass']>0) for v in rows),
            whole_roster_optimal_completion_lower_gain_pp=gain,
            switches=sum(v[arm]['switched'] for v in rows if v[arm] and v[arm]['supported']),
            infeasible_or_unresolved_views=[v['identity'] for v,t in zip(rows,status) if t in ('infeasible','unresolved')])
    once(PUBLIC/'analysis.json',stable(dict(result_source='fresh_label_aware_bound_from_cached_verified_A_paths',
        recording_views=rows, free_count=summary('free_count'), matched_count=summary('matched_count'),
        original_rules_unchanged=True, oracle_row_choices_exported=False, independent_confirmation=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','run','replay'))
    a=p.parse_args(); banks=registration()
    if a.phase=='register':return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
            raise ValueError('Commit diagnostic before reading outcomes')
    if a.phase=='replay' and not (PUBLIC/'run_execution.json').exists():raise ValueError('Original run required')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();run(banks,a.phase=='replay')
        once(PUBLIC/(a.phase+'_execution.json'),dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,artifact=artifact(PUBLIC/'analysis.json'),
            fresh_label_aware_solves=True, original_rules_unchanged=True, exact_replay=a.phase=='replay'))


if __name__=='__main__':main()
