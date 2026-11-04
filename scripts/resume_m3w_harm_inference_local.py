"""Read-only model inference continuation with a phase-specific disk reserve."""
import hashlib
from pathlib import Path
import platform
import subprocess
import sys

if platform.system()=='Darwin'and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before importing training libraries')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_harm_refinement as run

AMENDMENT=run.PUBLIC/'inference_resource_amendment.md'
RESERVE=5368709120


def main():
    if len(sys.argv)<2 or sys.argv[1]not in ('evaluate','verify'):
        raise ValueError('Inference only; cannot register, train, change a model or change its budget')
    for path in (Path(__file__),AMENDMENT):
        old=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,
            capture_output=True,check=True).stdout
        if hashlib.sha256(old).hexdigest()!=run.source.sha(path):raise ValueError('Commit phase resource amendment first')
    original=run.source.runtime
    def runtime(config,phase):
        if config['disk_reserve_bytes']!=10200547328:raise ValueError('Original resource configuration changed')
        start,guard,beat=original(dict(config,disk_reserve_bytes=RESERVE),phase)
        beat(state='inference_specific_disk_reserve',disk_reserve_bytes=RESERVE,
            unchanged_private_artifact_cap_bytes=config['private_artifact_cap_bytes'],
            amendment_sha256=run.source.sha(AMENDMENT),scientific_registration_sha256=run.source.sha(run.REG),
            model_training_allowed=False,scientific_rule_changed=False)
        return start,guard,beat
    run.source.runtime=runtime
    run.main()


if __name__=='__main__':main()
