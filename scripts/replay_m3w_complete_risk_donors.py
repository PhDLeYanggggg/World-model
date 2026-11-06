"""Repeatable no-fit replay with a unique private execution receipt per run."""
import fcntl
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_complete_risk_donors as run


def main():
    run.PRIVATE.mkdir(parents=True,exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,reg=run.register();run.committed()
        start,_,beat=run.runtime(cfg,'repeatable_replay')
        before={str(p):run.sha(p) for p in run.PUBLIC.rglob('*') if p.is_file()}
        original=run.seeds.method.fit
        def forbidden(*_args,**_kwargs):
            raise RuntimeError('Fitting forbidden during replay')
        run.seeds.method.fit=forbidden
        try:
            run.infer(reg,beat,replay=True);run.evaluate(reg,cfg,beat,replay=True)
            run.neural.rel.base.exact(before,{str(p):run.sha(p) for p in run.PUBLIC.rglob('*') if p.is_file()})
        finally:
            run.seeds.method.fit=original
        receipt=run.PRIVATE/'replays'/f'{time.time_ns()}_{os.getpid()}.json'
        run.once(receipt,dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            all_arrays_and_metrics_exact=True,public_files_unchanged=True,fitting_forbidden=True,
            wrapper_sha256=run.sha(Path(__file__))))
        beat(state='replay_complete',receipt=str(receipt.relative_to(ROOT)) if receipt.is_relative_to(ROOT) else str(receipt))


if __name__=='__main__':main()
