"""Native local continuation with a separately recorded operational disk reserve."""
import hashlib
from pathlib import Path
import platform
import subprocess
import sys

if platform.system()=='Darwin'and platform.machine()!='arm64':
    raise RuntimeError('Use the native arm64 .venv-pytorch environment, not Rosetta/Intel Conda')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_harm_refinement as run

AMENDMENT=run.PUBLIC/'operational_disk_amendment.md'
RESERVE=9663676416


def main():
    if len(sys.argv)<2 or sys.argv[1]not in ('train','evaluate','verify'):
        raise ValueError('Only continuation of the unchanged complete experiment is permitted')
    for path in (Path(__file__),AMENDMENT):
        old=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,
            capture_output=True,check=True).stdout
        if hashlib.sha256(old).hexdigest()!=run.source.sha(path):
            raise ValueError('Commit the operational amendment before using it')
    original=run.source.runtime
    def runtime(config,phase):
        if config['disk_reserve_bytes']!=10200547328:raise ValueError('Original registered operational setting changed')
        start,guard,beat=original(dict(config,disk_reserve_bytes=RESERVE),phase)
        beat(state='operational_reserve_amended',disk_reserve_bytes=RESERVE,
            original_disk_reserve_bytes=config['disk_reserve_bytes'],amendment_sha256=run.source.sha(AMENDMENT),
            registered_scientific_definition_sha256=run.source.sha(run.REG),
            risk_budget_changed=False,training_budget_changed=False)
        return start,guard,beat
    run.source.runtime=runtime
    if sys.argv[1]=='train'and '--resume'not in sys.argv:sys.argv.append('--resume')
    run.main()


if __name__=='__main__':main()
