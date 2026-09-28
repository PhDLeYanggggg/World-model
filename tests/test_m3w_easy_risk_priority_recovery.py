import pytest
from scripts.recover_m3w_easy_risk_priority_replay import validate_job


def fields():
    return dict(JobId='37577264', JobName='m3w_risk_priority_replay', UserId='k24101830(123)',
        Account='kcl', WorkDir='/tmp/m3w', Command='/tmp/m3w/run_replay.sh', Partition='cpu',
        NumCPUs='4', MinMemoryNode='16G', Restarts='0', JobState='PENDING')


def test_recover_only_exact_owned_registered_job():
    validate_job(fields(), '37577264', '/tmp/m3w')


@pytest.mark.parametrize('key,value', [('JobId','1'), ('JobName','unrelated'),
    ('WorkDir','/tmp/simulation'), ('Command','/tmp/simulation/run.sh'),
    ('UserId','other(123)'), ('NumCPUs','8'), ('MinMemoryNode','8G'), ('Restarts','1'),
    ('JobState','FAILED')])
def test_other_work_or_terminal_failure_cannot_be_recovered_as_live(key, value):
    f = fields(); f[key] = value
    with pytest.raises(AssertionError):
        validate_job(f, '37577264', '/tmp/m3w')
