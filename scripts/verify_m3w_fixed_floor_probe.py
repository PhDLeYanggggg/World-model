"""Verify producer bindings, replay and independent public-metric reductions."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_fixed_floor_probe as run
TESTS=['tests/test_m3w_fixed_floor_probe.py','tests/test_m3w_fixed_floor_protocol.py',
    'tests/test_m3w_score_support_calibration.py','tests/test_m3w_floor_relative.py',
    'tests/test_m3w_fixed_producer_roles.py','tests/test_m3w_dimensionless_intervention.py',
    'tests/test_m3w_score_support_reporting.py']


def main():
    cfg,_,_,_,identity=run.load()
    freeze=json.loads((run.PUBLIC/'decision_freeze.json').read_text())
    for r in freeze['groups']:
        assert run.artifact(ROOT/r['path'])==r
        doc=json.loads((ROOT/r['path']).read_text()); assert doc['identity']==identity
        run.api.assert_roles(doc['producer_sites'],doc['controller_sites'],doc['fitting_sites'],doc['held_sites'])
        for ref in doc['artifacts'].values(): assert run.artifact(ROOT/ref['path'])==ref
        assert not doc['held_labels_read']
    for name,n in [('fit_replay.json',1),('prediction_replay.json',108)]:
        d=json.loads((run.PUBLIC/name).read_text()); assert d['exact'] and d['groups']==n
    ev=json.loads((run.PUBLIC/'evaluation_replay.json').read_text()); assert ev['exact']
    for key in ('summary','details'): assert run.artifact(ROOT/ev[key]['path'])==ev[key]
    result=json.loads((run.PUBLIC/'summary.json').read_text())
    rows=json.loads((run.PRIVATE/'details.json').read_text())['rows']; count=0
    # Independent plain reduction, not the reporter's aggregation helper.
    for policy,metrics in result['summary'].items():
        for key,summary in metrics.items():
            values=[]
            for site,reported in summary['by_site'].items():
                samples=[r['metric'][key] for r in rows if r['policy']==policy and r['site']==site]
                if not samples or any(v is None for v in samples):
                    assert reported is None
                    continue
                mean=float(np.mean(samples)); np.testing.assert_allclose(mean,reported,atol=1e-12)
                values.append(mean); count+=1
            if len(values)==12:
                a=np.array(values); np.testing.assert_allclose(summary['point'],a.mean(),atol=1e-12)
                rng=np.random.default_rng(cfg['bootstrap_seed'])
                boot=a[rng.integers(0,len(a),size=(cfg['bootstrap_resamples'],len(a)))].mean(1)
                np.testing.assert_allclose(summary['ci95'],np.quantile(boot,[.025,.975]),atol=1e-12)
    log=run.PRIVATE/'scoped_pytest.txt'
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log.write_text(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    tests=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    artifacts={p.name:run.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.md','.json','.png') and p.name!='verification.json'}
    for script in ('report_m3w_fixed_floor_probe.py','diagnose_m3w_fixed_floor_probe.py'):
        proc=subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,capture_output=True,text=True)
        if proc.returncode: raise RuntimeError(proc.stderr)
    assert artifacts=={p:run.digest(run.PUBLIC/p) for p in artifacts}
    parent=json.loads((run.parent.PUBLIC/'verification.json').read_text())
    bindings=dict(parent['source_bindings']); bindings.update(identity['bindings'])
    extras=['scripts/report_m3w_fixed_floor_probe.py','scripts/diagnose_m3w_fixed_floor_probe.py',
            'scripts/verify_m3w_fixed_floor_probe.py',*TESTS]
    bindings.update({p:run.digest(ROOT/p) for p in extras})
    run.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        tests=tests,test_files=len(TESTS),test_log=run.artifact(log),producer_chain_exclusion_checked_groups=108,
        checkpoint_scores_bound_groups=108,exact_fitting_replays=1,exact_prediction_replays=108,
        real_inference_future_fields_removed=True,evaluation_replay_exact=True,
        independently_reduced_locality_metrics=count,reports_figure_byte_reproducible=True,
        linear_heads=216,new_neural_updates=0,all_scoped_technical_checks_passed=True,
        full_legacy_suite='not_run',cold_raw_rebuild=False,independent_confirmation=False,
        safety_certificate=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(verified=True,tests=tests,locality_reductions=count,artifacts=len(artifacts),source_files=len(bindings))))


if __name__=='__main__': main()
