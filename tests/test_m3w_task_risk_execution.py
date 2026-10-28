import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import manage_m3w_task_risk as api
from scripts import train_m3w_task_risk as worker


def record(**changes):
    f = dict(JobId='123', JobName='m3w_task_risk_pilot', JobState='COMPLETED',
             ExitCode='0:0', WorkDir=api.REMOTE)
    f.update(changes)
    return dict(returncode=0, stdout=' '.join(k+'='+v for k,v in f.items()))


def test_controller_requires_correct_job_and_successful_exit():
    assert api.controller_success(record(), '123', 'pilot')
    for changed in [dict(JobId='124'), dict(JobName='other'), dict(JobState='RUNNING'),
                    dict(ExitCode='1:0'), dict(WorkDir='/unrelated')]:
        assert not api.controller_success(record(**changed), '123', 'pilot')
    assert not api.controller_success(dict(returncode=1, stdout=''), '123', 'pilot')


def test_execution_requires_allocation_before_numerical_import():
    import os
    env = {k:v for k,v in os.environ.items() if k != 'SLURM_JOB_ID'}
    p = subprocess.run([sys.executable, '-m', 'scripts.train_m3w_task_risk', 'pilot',
                        '--root', api.REMOTE], cwd=api.base.ROOT, env=env, capture_output=True, text=True)
    assert p.returncode != 0 and 'Allocated compute required' in p.stderr


def test_refuses_unowned_remote_root(tmp_path):
    with pytest.raises(ValueError, match='Owned experiment only'):
        worker.verify(tmp_path)


def test_control_plane_payloads_parse_and_do_not_use_failed_accounting_dependency():
    for code in [api.INSTALL, api.SUBMIT, api.OBSERVE]:
        ast.parse(code)
        assert 'sacct' not in code
    assert "'--hold'" in api.SUBMIT and 'submission_intent' not in api.INSTALL
    assert 'unknown_timeout_inspect_do_not_resubmit' in api.SUBMIT
    assert "'TIMEOUT'" in api.SUBMIT and "'--no-requeue'" not in api.SUBMIT  # header, not CLI argument
    assert '#SBATCH --no-requeue' in api.SUBMIT


def test_preregistered_dimensions_and_closed_roles():
    cfg = json.loads(api.CONFIG.read_text())
    assert cfg['source_heads']*len(cfg['arms']) == cfg['neural_fits'] == 216
    assert cfg['head_training']['steps'] == 2000
    assert cfg['num_workers'] == 0 and cfg['cpu_threads'] == 4
    for key in ['independent_roles_read', 'threshold_search', 'new_forecasters',
                'deployment_changed', 'stage5c_executed', 'smc_enabled']:
        assert cfg[key] is False
    assert cfg['risk_budget'] == .02 and cfg['bootstrap_resamples'] >= 2000


def test_transport_closure_contains_control_plane_dependencies():
    from scripts.export_m3w_easy_hurdle_create import closure
    paths = closure(api.base.ROOT, ['scripts.train_m3w_task_risk', 'scripts.manage_m3w_task_risk'])
    for name in ['scripts/manage_m3w_temporal_auxiliary_create.py',
                 'src/world_model/m3w_task_risk_objective.py',
                 'src/world_model/m3w_task_risk_training.py']:
        assert api.base.ROOT/name in paths
