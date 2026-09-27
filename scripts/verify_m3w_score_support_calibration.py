"""Seal the exact study without converting technical checks into a safety claim."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_score_support_calibration as run
TESTS = ['tests/test_m3w_score_support_calibration.py','tests/test_m3w_score_support_reporting.py',
    'tests/test_m3w_nested_calibration.py','tests/test_m3w_risk_excess.py',
    'tests/test_m3w_fixed_producer_roles.py','tests/test_m3w_floor_relative.py',
    'tests/test_m3w_dimensionless_intervention.py']


def main():
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    _,data,jobs,parent_identity,old_identity,identity = run.load()
    # Future labels/masks/costs are unavailable to this real first-group replay.
    causal = {k:data[k] for k in ('sites','geometry','history','origin')}
    c = next(run.contexts(causal,jobs,identity['rosters']))
    arrays,_ = run.score_group(c,causal,parent_identity,old_identity)
    frozen = run.read_scores(c,identity)
    for k,v in arrays.items(): np.testing.assert_array_equal(frozen[k],v)
    for name, field, n in [('inference_replay.json','groups',36),('calibration_replay.json','groups',216)]:
        doc = json.loads((run.PUBLIC/name).read_text()); assert doc['all_exact'] and doc[field] == n
    ev = json.loads((run.PUBLIC/'evaluation_replay.json').read_text()); assert ev['exact']
    assert run.artifact(ROOT/ev['summary']['path']) == ev['summary']
    assert run.artifact(ROOT/ev['detail']['path']) == ev['detail']
    artifacts = {p.name:run.digest(p) for p in run.PUBLIC.iterdir()
                 if p.suffix in ('.json','.md','.png') and p.name != 'verification.json'}
    for script in ('report_m3w_score_support_calibration.py','plot_m3w_score_support_calibration.py',
                   'diagnose_m3w_score_support_calibration.py'):
        proc = subprocess.run([sys.executable,'scripts/'+script],cwd=ROOT,capture_output=True,text=True)
        if proc.returncode: raise RuntimeError(proc.stderr)
    assert artifacts == {p:run.digest(run.PUBLIC/p) for p in artifacts}
    proc = subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log = run.PRIVATE/'scoped_pytest.txt'; log.write_text(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    count = int(re.search(r'(\d+) passed',proc.stdout).group(1))
    parent = json.loads((run.parent.PUBLIC/'verification.json').read_text())
    bindings = dict(parent['source_bindings']); bindings.update(identity['bindings'])
    extras = ['scripts/report_m3w_score_support_calibration.py','scripts/plot_m3w_score_support_calibration.py',
              'scripts/verify_m3w_score_support_calibration.py','scripts/diagnose_m3w_score_support_calibration.py',*TESTS]
    restored = ['configs/m3w_european_nested_calibration_v1.json','scripts/run_m3w_european_nested_calibration.py',
                'tests/test_m3w_nested_calibration.py']
    for p in restored:
        prior = subprocess.run(['git','show','15768a5d:'+p],cwd=ROOT,capture_output=True,check=True).stdout
        assert prior == (ROOT/p).read_bytes()
    extras += restored
    bindings.update({p:run.digest(ROOT/p) for p in extras})
    for p,h in bindings.items(): assert run.digest(ROOT/p) == h
    run.immutable_json(run.PUBLIC/'verification.json',dict(source_bindings=bindings,artifacts=artifacts,
        tests=count,test_files=len(TESTS),test_log=run.artifact(log),scores_exact=36,calibration_decisions_exact=216,
        scoring_exact=True,reports_figure_byte_reproducible=True,legacy_namespace_restored_exact=True,
        first_group_inference_with_future_fields_removed_exact=True,
        all_scoped_technical_checks_passed=True,full_legacy_suite='not_run',cold_raw_rebuild=False,
        new_training_updates=0,independent_confirmation=False,calibration_certificate=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(tests=count,source_files=len(bindings),artifacts=len(artifacts),verified=True)))


if __name__ == '__main__': main()
