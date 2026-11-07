"""Run the registered local calibration as a resumable, logged background job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT/'data/stage_cvpr2027_experiments/dronecrowd_calibration_readout_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/dronecrowd_calibration_readout_v1'
RUNNER = ROOT/'scripts/run_m3w_dronecrowd_calibration.py'
STATE = DIR/'job_state.json'


def atomic(value):
    tmp = STATE.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(STATE)


def worker(fd):
    os.fstat(fd)
    start = time.time()
    state = dict(supervisor_pid=os.getpid(), started_unix=start, status='running')
    if sys.platform == 'darwin' and shutil.which('caffeinate'):
        subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for phase in ('infer', 'score', 'replay'):
        receipt = PUBLIC/f'{phase}_execution.json'
        if receipt.is_file():
            continue
        child = subprocess.Popen([sys.executable, str(RUNNER), phase], cwd=ROOT,
                                 stdin=subprocess.DEVNULL)
        atomic(dict(state, phase=phase, child_pid=child.pid))
        status = child.wait()
        if status:
            atomic(dict(state, status='failed', phase=phase, child_pid=child.pid, exit_code=status))
            return status
    atomic(dict(state, status='complete', elapsed_seconds=time.time()-start))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('start', 'worker'))
    p.add_argument('--lock-fd', type=int)
    args = p.parse_args()
    if args.mode == 'worker':
        return worker(args.lock_fd)
    if not (PUBLIC/'smoke_execution.json').is_file():
        raise ValueError('Require a successful registered real-model smoke run')
    if (PUBLIC/'replay_execution.json').is_file():
        raise ValueError('Calibration already replayed; inspect its report')
    DIR.mkdir(parents=True, exist_ok=True)
    with (DIR/'launcher.lock').open('a') as lock, (DIR/'calibration.log').open('a') as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fd = lock.fileno()
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--lock-fd', str(fd)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, pass_fds=(fd,))
    print(dict(supervisor_pid=child.pid, state=str(STATE), log=str(DIR/'calibration.log')), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
