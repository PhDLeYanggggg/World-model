"""Verify fitting, causal decisions, and independent fixed-roster reductions."""
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_tail as run
TESTS=['tests/test_m3w_fixed_floor_tail.py','tests/test_m3w_fixed_floor_tail_protocol.py',
       'tests/test_m3w_fixed_floor_tail_reporting.py','tests/test_m3w_fixed_floor_tail_verification.py',
       'tests/test_m3w_fixed_floor_probe.py','tests/test_m3w_fixed_floor_protocol.py',
       'tests/test_m3w_risk_excess.py','tests/test_m3w_fixed_producer_roles.py']


def independently_match(anchor,eligible,score,recordings,frames,ids):
    groups=defaultdict(list)
    for i,(rec,frame) in enumerate(zip(recordings,frames)): groups[(str(rec),int(frame))].append(i)
    result=np.zeros(len(ids),bool)
    for positions in groups.values():
        count=sum(bool(anchor[i]) for i in positions)
        ranked=sorted((i for i in positions if eligible[i]),key=lambda i:(float(score[i]),int(ids[i])))
        if count>len(ranked): raise ValueError('Anchor exceeds query eligibility')
        result[ranked[:count]]=True
    return result,len(groups)


def reduce_check(reported,rows,key,seed,draws):
    values=[]; checks=0
    for site,expected in reported['by_site'].items():
        samples=[r['metric'][key] for r in rows if r['site']==site]
        if not samples or any(v is None for v in samples):
            assert expected is None
            continue
        value=float(np.mean(samples)); np.testing.assert_allclose(value,expected,atol=1e-12)
        values.append(value); checks+=1
    if len(values)==len(reported['by_site']):
        a=np.asarray(values); np.testing.assert_allclose(a.mean(),reported['point'],atol=1e-12)
        rng=np.random.default_rng(seed)
        boot=a[rng.integers(0,len(a),size=(draws,len(a)))].mean(1)
        np.testing.assert_allclose(np.quantile(boot,[.025,.975]),reported['ci95'],atol=1e-12)
    else:
        assert reported['point'] is None and reported['ci95'] is None
    return checks


def main():
    cfg,data,_,_,_,bound=run.load()
    freeze=json.loads((run.PUBLIC/'decision_freeze.json').read_text())
    assert freeze['identity']==bound and freeze['heads_count']==216
    head_refs={}
    for ref in freeze['heads']:
        assert run.artifact(ROOT/ref['path'])==ref
        d=json.loads((ROOT/ref['path']).read_text()); ident=d['identity']
        assert ident['experiment']==bound and not d['held_labels_read']
        run.parent.api.assert_roles(ident['producer_sites'],ident['controller_sites'],
                                   ident['training_sites'],ident['held_sites'])
        assert d['fit']['complete'] and d['fit']['step']==2000 and d['fit']['unknown_rows_sampled']==0
        for r in d['artifacts'].values(): assert run.artifact(ROOT/r['path'])==r
        head_refs[(ident['group'],d['arm'])]=d
    queries=0; unknown_selected=0
    for ref in freeze['decisions']:
        assert run.artifact(ROOT/ref['path'])==ref
        rec=json.loads((ROOT/ref['path']).read_text()); assert rec['identity']['experiment']==bound
        assert run.artifact(ROOT/rec['arrays']['path'])==rec['arrays']
        name=rec['identity']['group']
        with np.load(ROOT/rec['arrays']['path'],allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
        score_ref=head_refs[(name,'mse')]['artifacts']['scores']
        with np.load(ROOT/score_ref['path'],allow_pickle=False) as z:
            np.testing.assert_array_equal(a['ids'],z['ids']); p=z['scores'].astype(float)
        ids=a['ids']; score=np.maximum(p[:,1]-.02*p[:,0],p[:,3]-.02*p[:,2])
        query=[str(s)+'|'+str(r) for s,r in zip(data['sites'][ids],data['recordings'][ids])]
        fresh,n=independently_match(a['tail4'],a['eligible'],score,query,data['frames'][ids],ids)
        np.testing.assert_array_equal(fresh,a['mse_matched_count']); queries+=n
        unknown_selected+=int((a['tail4']&~np.isfinite(data['baseline_ade'][ids,1])).sum())
        for arm in ('mse','tail4'):
            with np.load(ROOT/head_refs[(name,arm)]['artifacts']['scores']['path'],allow_pickle=False) as z: p=z['scores'].copy()
            assert np.isfinite(p).all() and (p>=0).all()
            np.testing.assert_array_equal(a[arm],a['eligible']&(p[:,1]<=.02*p[:,0])&(p[:,3]<=.02*p[:,2]))
    prediction=json.loads((run.PUBLIC/'prediction_replay.json').read_text())
    assert prediction['all_exact'] and prediction['heads']==216 and prediction['decisions']==108
    causal=json.loads((run.PUBLIC/'causal_replay.json').read_text())
    assert causal['exact'] and causal['future_fields_removed'] and causal['heads']==2
    fitting=json.loads((run.PUBLIC/'training_replay.json').read_text())
    assert fitting['exact'] and fitting['heads']==2 and fitting['paired_initialization_optimizer_rng_draws_trace_exact']
    for item in fitting['checkpoints']:
        for key in ('replay','reference'): assert run.artifact(ROOT/item[key]['path'])==item[key]
    evaluation=json.loads((run.PUBLIC/'evaluation_replay.json').read_text()); assert evaluation['exact']
    for key in ('summary','details'): assert run.artifact(ROOT/evaluation[key]['path'])==evaluation[key]
    d=json.loads((run.PUBLIC/'summary.json').read_text())
    details=json.loads((run.PRIVATE/'details.json').read_text()); count=0
    for section,rowskey in [('summary','rows'),('paired','contrasts'),('quality','quality')]:
        for policy,metrics in d[section].items():
            rows=[r for r in details[rowskey] if r['policy']==policy]
            for key,reported in metrics.items():
                count+=reduce_check(reported,rows,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    for seed,policies in d['by_seed'].items():
        for policy,metrics in policies.items():
            rows=[r for r in details['rows'] if r['policy']==policy and r['seed']==int(seed)]
            for key,reported in metrics.items():
                count+=reduce_check(reported,rows,key,cfg['bootstrap_seed'],cfg['bootstrap_resamples'])
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log=run.PRIVATE/'scoped_pytest.txt'; log.write_text(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    artifacts={p.name:run.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.md','.json','.png') and p.name!='verification.json'}
    for script in ('report_m3w_fixed_floor_tail.py','diagnose_m3w_fixed_floor_tail.py'):
        proc=subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,capture_output=True,text=True)
        if proc.returncode: raise RuntimeError(proc.stderr)
    assert artifacts=={p:run.digest(run.PUBLIC/p) for p in artifacts}
    parent=json.loads((run.parent.PUBLIC/'verification.json').read_text())
    bindings=dict(parent['source_bindings']); bindings.update(bound['bindings'])
    extras=['scripts/report_m3w_fixed_floor_tail.py','scripts/replay_m3w_fixed_floor_tail_training.py',
            'scripts/diagnose_m3w_fixed_floor_tail.py',
            'scripts/verify_m3w_fixed_floor_tail.py',*TESTS]
    bindings.update({p:run.digest(ROOT/p) for p in extras})
    run.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        tests=tests,test_files=len(TESTS),test_log=run.artifact(log),source_excluded_groups=108,
        checkpoint_scores_bound_heads=216,exact_training_replays=2,exact_prediction_replays=216,
        matched_sampler_pairs=108,real_inference_future_fields_removed_heads=2,
        independently_reconstructed_query_views=queries,unknown_label_tail_interventions_dependent_views=unknown_selected,
        evaluation_replay_exact=True,independently_reduced_locality_metrics=count,
        reports_figure_byte_reproducible=True,all_scoped_technical_checks_passed=True,
        full_legacy_suite='not_run',cold_raw_rebuild=False,independent_confirmation=False,
        safety_certificate=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(verified=True,tests=tests,locality_reductions=count,query_views=queries,
                         artifacts=len(artifacts),source_files=len(bindings))))


if __name__=='__main__': main()
