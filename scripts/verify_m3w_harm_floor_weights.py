"""Recompute fitting-only loss weights independently of saved model receipts."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_moving_harm_floor as run


def main():
    source, api = run.source, run.api
    with (run.PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock, source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        run.old.previous.teacher.lineage.PRIVATE = run.PRIVATE
        started, guard, beat = run.old.previous.teacher.lineage.runtime({}, source.read(run.CONFIG), 'verify_weights')
        cfg, values = run.register(beat); checked = []
        for job in values[3]:
            data, origin = run.inputs(values, job)
            api.parent.validate(data, origin)
            expected = api.weighting(data, origin, cfg['training'])
            for arm in api.ARMS:
                row = source.read(run.PUBLIC/'fits'/(job['key']+'_'+arm+'.json'))
                state = api.base.read_checkpoint(source.checked(row['checkpoint']))
                api.base.exact(expected, state['fitting_harm_weighting'])
                api.base.exact(source.stable(expected), row['fitting_harm_weighting'])
                api.base.exact(expected['planned_query_counts'], state['sampler']['counts'])
                if not source.np.isclose(expected['floor_positive_mass'],
                        expected['original_positive_mass']*expected['uniform_multiplier'], rtol=1e-12, atol=1e-15):
                    raise AssertionError('Exact planned positive coefficient mass control required')
                checked.append(dict(job=job['key'], arm=arm, exact=True))
            del data, origin, state; source.gc.collect(); guard()
            beat(state='weights_recomputed', completed=len(checked))
        source.once(run.PUBLIC/'weighting_verification.json', dict(rows=checked, all9_exact=len(checked) == 9,
            own_B_only=True, no_new_V_or_H=True, pid=source.os.getpid(), seconds=time.monotonic()-started))
        beat(state='complete')


if __name__ == '__main__':
    main()
