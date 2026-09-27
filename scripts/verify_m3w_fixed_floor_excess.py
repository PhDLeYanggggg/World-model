"""Independent query/metric reductions and matched-control provenance checks."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_excess as run
from scripts.verify_m3w_fixed_floor_tail import independently_match,reduce_check,TESTS as PARENT_TESTS
TESTS=['tests/test_m3w_fixed_floor_excess.py','tests/test_m3w_fixed_floor_excess_protocol.py',*PARENT_TESTS]


def main():
    cfg,data,_,_,_,_,bound=run.load()
    freeze=json.loads((run.PUBLIC/'decision_freeze.json').read_text()); assert freeze['identity']==bound
    refs={}
    for ref in freeze['heads']:
        assert run.artifact(ROOT/ref['path'])==ref
        r=json.loads((ROOT/ref['path']).read_text()); ident=r['identity']; h=ident['roles_and_targets']
        run.floor_api.api.assert_roles(h['producer_sites'],h['controller_sites'],h['training_sites'],h['held_sites'])
        assert ident['experiment']==bound and not r['held_labels_read']
        assert r['fit']['complete'] and r['fit']['step']==2000 and r['fit']['unknown_rows_sampled']==0
        assert run.artifact(ROOT/ident['control']['path'])==ident['control']
        control=json.loads((ROOT/ident['control']['path']).read_text())
        for record in (r,control):
            for artifact in record['artifacts'].values(): assert run.artifact(ROOT/artifact['path'])==artifact
        a=run.torch.load(ROOT/r['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
        b=run.torch.load(ROOT/control['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=False)
        run.api.assert_matched(a,b); refs[h['group']]=(r,control)
    queries=0
    for ref in freeze['decisions']:
        assert run.artifact(ROOT/ref['path'])==ref
        r=json.loads((ROOT/ref['path']).read_text()); assert r['identity']['experiment']==bound
        assert run.artifact(ROOT/r['arrays']['path'])==r['arrays']
        name=r['identity']['roles_and_targets']['group']; head,control=refs[name]
        with np.load(ROOT/r['arrays']['path'],allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
        with np.load(ROOT/control['artifacts']['scores']['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],a['ids']); p=z['scores'].astype(float)
        ids=a['ids']; q=np.maximum(p[:,1]-.02*p[:,0],p[:,3]-.02*p[:,2])
        keys=[str(s)+'|'+str(rec) for s,rec in zip(data['sites'][ids],data['recordings'][ids])]
        m,n=independently_match(a['excess'],a['eligible'],q,keys,data['frames'][ids],ids)
        np.testing.assert_array_equal(m,a['mse_matched_count']); queries+=n
        with np.load(ROOT/head['artifacts']['scores']['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'],ids); p=z['scores'].copy()
        np.testing.assert_array_equal(a['excess'],a['eligible']&(p[:,1]<=.02*p[:,0])&(p[:,3]<=.02*p[:,2]))
    pred=json.loads((run.PUBLIC/'prediction_replay.json').read_text()); assert pred['exact'] and pred['new_heads']==108
    causal=json.loads((run.PUBLIC/'causal_replay.json').read_text()); assert causal['exact'] and causal['future_fields_removed']
    fitting=json.loads((run.PUBLIC/'training_replay.json').read_text()); assert fitting['exact'] and fitting['optimizer_rng_draws_loss_trace_exact']
    for k in ('replay','reference'): assert run.artifact(ROOT/fitting[k]['path'])==fitting[k]
    ev=json.loads((run.PUBLIC/'evaluation_replay.json').read_text()); assert ev['exact']
    for k in ('summary','details'): assert run.artifact(ROOT/ev[k]['path'])==ev[k]
    d=json.loads((run.PUBLIC/'summary.json').read_text()); details=json.loads((run.PRIVATE/'details.json').read_text()); count=0
    for section,rowskey in [('summary','rows'),('quality','quality'),('paired','contrasts')]:
        for policy,metrics in d[section].items():
            rows=[r for r in details[rowskey] if r['policy']==policy]
            for key,reported in metrics.items(): count+=reduce_check(reported,rows,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    for seed,policies in d['by_seed'].items():
        for policy,metrics in policies.items():
            rows=[r for r in details['rows'] if r['policy']==policy and r['seed']==int(seed)]
            for key,reported in metrics.items(): count+=reduce_check(reported,rows,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    diagnosis=json.loads((run.PUBLIC/'failure_localization.json').read_text())
    qidx={(r['group'],r['site'],r['policy']):r['metric'] for r in details['quality']}
    for key,reported in diagnosis['paired_signed_MSE_differences'].items():
        sourcekey=key.removesuffix('_difference'); rr=[]
        for r in details['quality']:
            if r['policy']!='excess': continue
            delta=r['metric'][sourcekey]-qidx[(r['group'],r['site'],'mse')][sourcekey]
            rr.append(dict(site=r['site'],metric={key:delta}))
        count+=reduce_check(reported,rr,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    log=run.PRIVATE/'scoped_pytest.txt'
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log.write_text(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    artifacts={p.name:run.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.json','.md','.png') and p.name!='verification.json'}
    for script in ('report_m3w_fixed_floor_excess.py','diagnose_m3w_fixed_floor_excess.py'):
        proc=subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,capture_output=True,text=True)
        if proc.returncode: raise RuntimeError(proc.stderr)
    assert artifacts=={p:run.digest(run.PUBLIC/p) for p in artifacts}
    parent=json.loads((run.parent.PUBLIC/'verification.json').read_text()); bindings=dict(parent['source_bindings']); bindings.update(bound['bindings'])
    extras=['scripts/report_m3w_fixed_floor_excess.py','scripts/replay_m3w_fixed_floor_excess_training.py',
            'scripts/diagnose_m3w_fixed_floor_excess.py',
            'scripts/verify_m3w_fixed_floor_excess.py',*TESTS]
    bindings.update({p:run.digest(ROOT/p) for p in extras})
    run.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        tests=tests,test_files=len(TESTS),test_log=run.artifact(log),matched_cached_controls=108,
        source_excluded_groups=108,exact_new_prediction_replays=108,exact_decision_replays=108,
        exact_full_fit_replays=1,future_fields_removed_real_heads=1,evaluation_replay_exact=True,
        independently_reconstructed_query_views=queries,independently_reduced_locality_metrics=count,
        reports_figure_byte_reproducible=True,all_scoped_technical_checks_passed=True,
        full_legacy_suite='not_run',cold_raw_rebuild=False,independent_confirmation=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(verified=True,tests=tests,query_views=queries,metric_reductions=count,artifacts=len(artifacts),source_files=len(bindings))))


if __name__=='__main__': main()
