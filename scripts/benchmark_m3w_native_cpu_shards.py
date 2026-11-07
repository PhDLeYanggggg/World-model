"""Benchmark process-isolated recordings with exact dev prediction replay."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dronecrowd_calibration as cal

run, np, torch = cal.run, cal.np, cal.torch
PUBLIC = cal.roles.HOME/'native_cpu_shard_benchmark_v1'
PRIVATE = cal.DATA/'native_cpu_shard_benchmark_v1'
PREFIXES_PER_SHARD = 6
N_SHARDS = 3


def digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def select_work(d, frames):
    """Select only from the original exposed-development input roster."""
    roster = []
    for rec in np.unique(d['recordings']):
        subsets = [q for q in d['groups'] if d['recordings'][q[0]] == rec]
        subsets.sort(key=lambda q: frames[q[0]])
        if len(subsets) < PREFIXES_PER_SHARD:
            continue
        m = len(subsets)//2
        q = subsets[m-PREFIXES_PER_SHARD//2:m+PREFIXES_PER_SHARD//2]
        roster.append((sum(map(len, q)), str(rec), [int(frames[a[0]]) for a in q]))
    roster.sort()
    if len(roster) < N_SHARDS:
        raise ValueError('Development recording roster is too small')
    return [dict(rows=n, recording=r, frames=f) for n, r, f in
            (roster[int((len(roster)-1)*q)] for q in (.2, .55, .85))]


class PrefixSource:
    def __init__(self, d, frames, rec, planned):
        self.d = d
        self.indices = {}
        for q in d['groups']:
            frame = int(frames[q[0]])
            if str(d['recordings'][q[0]]) == rec and frame in planned:
                if frame in self.indices or not (frames[q] == frame).all():
                    raise ValueError('Query group is not one recording and frame')
                self.indices[frame] = q
        if set(self.indices) != set(planned):
            raise ValueError('Incomplete development prefix shard')

    def get_inputs(self, frame):
        q = self.indices[frame]
        mask = np.ones((len(q), 8), dtype=bool)
        return dict(positions=self.d['history'][q], history_mask=mask, history_complete=mask.all(1),
                    observation_frames=np.arange(frame-7, frame+1), query_frame=frame,
                    agent_ids=self.d['agents'][q])


def dev():
    d = run.population()
    # These two named arrays are past-query identity only.
    index = run.old.prior.arrays(('agents', 'frames'))
    d['agents'] = index['agents']
    return d, index['frames']


def worker(slot, mode):
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise ValueError('Native arm64 required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    d, frames = dev()
    _, _, packet = cal.register()
    loaded = cal.networks(packet)
    work = run.read(PRIVATE/'work.json')[slot]
    source = PrefixSource(d, frames, work['recording'], work['frames'])
    neural, times = [], []
    native_predict = cal.native.predict_relative

    def exact_trace(*args, **kwargs):
        start = time.monotonic()
        result = native_predict(*args, **kwargs)
        times.append(time.monotonic()-start)
        neural.append(digest(result))
        return result

    cal.native.predict_relative = exact_trace
    start = time.monotonic()
    try:
        arrays = cal.infer(source, np.asarray(work['frames']), loaded, lambda **_: None)
    finally:
        cal.native.predict_relative = native_predict
    wall = time.monotonic()-start
    if len(neural) != 9 or len(arrays['agents']) != work['rows']:
        raise ValueError('Expected all three neural families per seed and the complete shard')
    run.once(PRIVATE/f'{mode}_{slot}.json', dict(slot=slot, pid=os.getpid(), rows=work['rows'],
        neural_hashes=neural, arrays={k: digest(v) for k, v in arrays.items()},
        family_seconds={k: sum(times[j] for j in range(i, 9, 3))
                        for i, k in enumerate(('pct', 'eqmotion', 'self'))},
        inference_seconds=wall, torch_threads=torch.get_num_threads()))


def phase(mode):
    flags = [sys.executable, str(Path(__file__).resolve()), '--worker', '--mode', mode]
    start = time.monotonic()
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    if mode == 'serial':
        for j in range(N_SHARDS):
            subprocess.run([*flags, '--slot', str(j)], cwd=ROOT, env=env, check=True)
    else:
        procs = [subprocess.Popen([*flags, '--slot', str(j)], cwd=ROOT, env=env) for j in range(N_SHARDS)]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise RuntimeError(f'Parallel workers failed: {codes}')
    seconds = time.monotonic()-start
    print(dict(mode=mode, wall_seconds=seconds), flush=True)
    return seconds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--mode', choices=('serial', 'parallel'))
    p.add_argument('--slot', type=int)
    a = p.parse_args()
    if a.worker:
        return worker(a.slot, a.mode)
    if (PUBLIC/'analysis.json').exists():
        raise ValueError('Benchmark exists; do not overwrite its measured timings')
    if any(PRIVATE.glob('*.json')):
        raise ValueError('Existing benchmark state; inspect before restarting')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    d, f = dev()
    work = select_work(d, f)
    run.once(PRIVATE/'work.json', work)
    serial, parallel = (phase(k) for k in ('serial', 'parallel'))
    rows = []
    for i in range(N_SHARDS):
        a, b = (run.read(PRIVATE/f'{mode}_{i}.json') for mode in ('serial', 'parallel'))
        for key in ('rows', 'neural_hashes', 'arrays'):
            if a[key] != b[key]:
                raise ValueError('Scheduling changed forecasts or sparse choices')
        rows.append(dict(rows=a['rows'], exact_neural_models=len(a['neural_hashes']),
            exact_output_arrays=len(a['arrays']), serial_family_seconds=a['family_seconds'],
            parallel_family_seconds=b['family_seconds'],
            serial_inference_seconds=a['inference_seconds'], parallel_inference_seconds=b['inference_seconds']))
    run.once(PUBLIC/'analysis.json', run.stable(dict(
        development_only=True, confirmation_inputs_opened=False, calibration_job_modified=False,
        processor=platform.machine(), record_shards=N_SHARDS, prefixes_per_shard=PREFIXES_PER_SHARD,
        inference_exact=True, serial_end_to_end_seconds=serial, parallel_end_to_end_seconds=parallel,
        measured_end_to_end_speedup=serial/parallel, process_threads=4, dataloader_workers=0,
        rows=rows, benchmark_code_sha256=run.sha(Path(__file__)),
        benchmark_parents=[run.artifact(p) for p in (cal.REG, Path(cal.__file__),
            run.old.prior.PARENT/'population.json')], no_runtime_amendment_yet=True)))


if __name__ == '__main__':
    main()
