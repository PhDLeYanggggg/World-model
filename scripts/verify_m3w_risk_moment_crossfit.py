"""Technical reproduction checks, separate from model benefit and deployment."""
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_risk_moment_crossfit as run

TESTS = ['tests/test_m3w_risk_moment_crossfit.py', 'tests/test_m3w_geometric_cost_head.py',
    'tests/test_m3w_risk_moment_reporting.py',
    'tests/test_m3w_native_gain_harm.py', 'tests/test_m3w_fixed_producer_roles.py',
    'tests/test_m3w_floor_relative.py', 'tests/test_m3w_dimensionless_intervention.py']


def main():
    run.torch.set_num_threads(4); run.torch.set_num_interop_threads(1)
    _, _, _, identity = run.load()
    replay = json.loads((run.PUBLIC/'prediction_replay.json').read_text())
    assert replay['exact'] and len(replay['checks']) == 144
    training = json.loads((run.PUBLIC/'training_replay.json').read_text())
    assert training['parameters_optimizer_rng_draws_losses_exact'] and training['steps'] == 2000
    ev = json.loads((run.PUBLIC/'evaluation_replay.json').read_text())
    assert ev['exact']
    for key in ('summary','detail'): assert run.artifact(ROOT/ev[key]['path']) == ev[key]
    before = {p.name: run.digest(p) for p in run.PUBLIC.iterdir()
              if p.suffix in ('.json','.md','.png') and p.name != 'verification.json'}
    for script in ('report_m3w_risk_moment_crossfit.py', 'plot_m3w_risk_moment_crossfit.py'):
        report = subprocess.run([sys.executable, 'scripts/'+script], cwd=ROOT, capture_output=True, text=True)
        if report.returncode: raise RuntimeError(report.stderr)
    assert before == {name: run.digest(run.PUBLIC/name) for name in before}
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', *TESTS],
                          cwd=ROOT, capture_output=True, text=True)
    log = run.PRIVATE/'scoped_pytest.txt'; log.write_text(proc.stdout+proc.stderr)
    if proc.returncode: raise RuntimeError(proc.stdout+proc.stderr)
    count = int(re.search(r'(\d+) passed', proc.stdout).group(1))
    parent = json.loads((run.parent.PUBLIC/'verification.json').read_text())
    bindings = dict(parent['source_bindings']); bindings.update(identity['bindings'])
    extra = ['scripts/replay_m3w_risk_moment_training.py', 'scripts/report_m3w_risk_moment_crossfit.py',
             'scripts/plot_m3w_risk_moment_crossfit.py',
             'scripts/verify_m3w_risk_moment_crossfit.py', *TESTS]
    bindings.update({p: run.digest(ROOT/p) for p in extra})
    for p, h in bindings.items(): assert run.digest(ROOT/p) == h
    seal = dict(source_bindings=bindings, artifacts=before, scoped_tests=count, scoped_test_files=len(TESTS),
        test_log=run.artifact(log), head_replays_exact=144, full_training_replay_exact=True,
        diagnostic_readout_replay_exact=True, reports_byte_reproducible=True,
        all_scoped_technical_checks_passed=True, full_legacy_suite='not_run', cold_raw_rebuild=False,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    run.immutable_json(run.PUBLIC/'verification.json', seal)
    print(json.dumps(dict(tests=count, source_files=len(bindings), artifacts=len(before), verified=True)))


if __name__ == '__main__': main()
