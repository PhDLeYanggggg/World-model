"""Scoped technical verification, distinct from empirical gates or confirmation."""
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_european_dimensionless_intervention as run

TESTS=[
    'tests/test_m3w_dimensionless_intervention.py',
    'tests/test_m3w_dimensionless_intervention_reporting.py',
    'tests/test_m3w_dimensionless_refit.py',
    'tests/test_m3w_dimensionless_correction.py',
    'tests/test_m3w_fixed_producer_roles.py',
    'tests/test_m3w_geometric_cost_head.py',
    'tests/test_m3w_scaled_risk_controls.py',
    'tests/test_m3w_joint_intervention.py',
    'tests/test_m3w_interaction_controls.py',
    'tests/test_m3w_european_source_intervention.py',
    'tests/test_m3w_native_gain_harm.py',
    'tests/test_m3w_floor_relative.py',
    'tests/test_m3w_agent_track_context.py']


def main():
    run.torch.set_num_threads(4);run.torch.set_num_interop_threads(1)
    cfg,data,jobs,identity,_=run.load()
    before={p.name:run.digest(p) for p in run.PUBLIC.iterdir() if p.suffix in ('.md','.json','.png')
            and p.name not in ('verification.json','evaluation.json')}
    for file,key in [('head_replay.json','all_exact'),('decision_replay.json','all_exact'),('evaluation_replay.json','exact'),
                     ('training_replay.json','parameters_optimizer_rng_draws_losses_exact')]:
        assert json.loads((run.PUBLIC/file).read_text())[key]
    assert len(json.loads((run.PUBLIC/'head_replay.json').read_text())['checks'])==108
    replay=json.loads((run.PUBLIC/'evaluation_replay.json').read_text())
    assert replay['evaluation_sha256']==run.digest(run.PUBLIC/'evaluation.json')
    for script in ('report_m3w_european_dimensionless_intervention.py','diagnose_m3w_dimensionless_intervention.py',
                   'plot_m3w_dimensionless_intervention.py'):
        p=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,capture_output=True,text=True)
        if p.returncode:raise RuntimeError(p.stderr)
    assert before=={k:run.digest(run.PUBLIC/k) for k in before}
    proc=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],cwd=ROOT,capture_output=True,text=True)
    log=run.PRIVATE/'scoped_pytest.txt';log.write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    passed=int(re.search(r'(\d+) passed',proc.stdout).group(1))
    parent=json.loads((run.parent.PUBLIC/'verification.json').read_text())
    bindings=dict(parent['source_bindings']);bindings.update(identity['bindings'])
    extra=['scripts/report_m3w_european_dimensionless_intervention.py','scripts/diagnose_m3w_dimensionless_intervention.py',
           'scripts/replay_m3w_dimensionless_intervention_training.py','scripts/plot_m3w_dimensionless_intervention.py',
           'scripts/verify_m3w_dimensionless_intervention.py',*TESTS]
    bindings.update({p:run.digest(ROOT/p) for p in extra})
    for p,h in bindings.items():assert run.digest(ROOT/p)==h,p
    record=dict(all_scoped_technical_checks_passed=True,source_bindings=bindings,artifacts=before,
        scoped_tests=passed,scoped_test_files=len(TESTS),test_log=run.artifact(log),
        full_legacy_suite='not_run',independent_confirmation=False,cold_raw_rebuild=False,
        completed_head_replays=108,completed_decision_groups=36,reports_figures_byte_reproducible=True,
        full_readout_exact=True,fixed_first_training_replay_exact=True,
        detailed_evaluation_local_only=run.artifact(run.PUBLIC/'evaluation.json'),
        empirical_gates=json.loads((run.PUBLIC/'gates.json').read_text()),
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    run.immutable_json(run.PUBLIC/'verification.json',record)
    print(json.dumps(dict(tests=passed,test_files=len(TESTS),artifacts=len(before),source_bindings=len(bindings),
                          technical_verification_pass=True,empirical_safe_benefit=record['empirical_gates']['exploratory_useful_safe_screen'])))


if __name__=='__main__':main()
