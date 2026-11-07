"""Optionally parallelize confirmation inference without changing its scorer."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dronecrowd_confirmation as confirm

cal, run = confirm.cal, confirm.run
PUBLIC, PRIVATE = confirm.PUBLIC, confirm.PRIVATE
BENCH = cal.roles.HOME/'native_cpu_shard_benchmark_v1/analysis.json'
REG = PUBLIC/'runtime_amendment.json'
SCRIPT = Path(__file__)
N = 3


def assign(allowed, visible_rows):
    """Partition full recordings using input-independent workload metadata."""
    if len(set(allowed)) != len(allowed) or set(allowed) != set(visible_rows):
        raise ValueError('Expected one workload estimate per authorized recording')
    weights, shards = [0]*N, [[] for _ in range(N)]
    for rec in sorted(allowed, key=lambda r: (-visible_rows[r], r)):
        k = min(range(N), key=lambda i: (weights[i], i))
        shards[k].append(rec)
        weights[k] += visible_rows[rec]
    return shards


def register():
    b = run.read(BENCH)
    if (not b['development_only'] or not b['inference_exact'] or b['confirmation_inputs_opened']
            or b['process_threads'] != 4 or b['record_shards'] != N
            or b['measured_end_to_end_speedup'] <= 1
            or b['benchmark_code_sha256'] != run.sha(ROOT/'scripts/benchmark_m3w_native_cpu_shards.py')):
        raise ValueError('Require the matching exact development scheduling benchmark')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in (SCRIPT, BENCH, confirm.REG,
        Path(confirm.__file__), Path(cal.__file__))], process_count=N, threads_per_process=4,
        dataloader_workers=0, changes_scientific_protocol=False, changes_predictions=False,
        completion_verified_by_original_infer_runner=True, confirmation_input_opening=False)))


def metadata(manifest):
    allowed = manifest['assignments']['confirmation']['recordings']
    inv = run.read(cal.CACHE)
    info = {r['sequence_id']: r for r in inv['recordings'] if r['sequence_id'] in allowed}
    weights = {k: v['visible_rows'] for k, v in info.items()}
    return info, inv['provenance'], assign(allowed, weights)


def worker(slot, manifest, packet, admission):
    cal.torch.set_num_threads(4)
    cal.torch.set_num_interop_threads(1)
    info, provenance, shards = metadata(manifest)
    loaded = cal.networks(packet)
    started = time.monotonic()

    def beat(**v):
        row = dict(pid=os.getpid(), shard=slot, seconds=time.monotonic()-started, **v)
        run.kin.prior.source.atomic(PRIVATE/f'heartbeat_shard_{slot}.json', row)
        print(row, flush=True)

    for seq in shards[slot]:
        path = PRIVATE/f'forecast_{seq}.npz'
        mark = path.with_suffix('.json')
        if not mark.exists():
            source = confirm.reader(manifest, seq, info[seq], provenance)
            data = cal.infer(source, source.query_frames, loaded, beat)
            j = cal.model.ARMS.index(admission['arm'])
            take = [j, *(cal.model.ARMS.index(k) for k in confirm.MATCH)]
            for seed in cal.SEEDS:
                for k in ('index', 'paths'):
                    name = f'{k}_{seed}'
                    data[name] = data[name][take]
            cal.prior.h.prune.previous.history.feedback.write_array(path, **data)
            run.once(mark, dict(sequence=seq, admission=run.sha(confirm.ADMISSION), queries=len(data['frames']),
                rows=len(data['agents']), artifact=run.artifact(path)))
        saved = run.read(mark)
        if saved['sequence'] != seq or saved['admission'] != run.sha(confirm.ADMISSION):
            raise ValueError('Shard checkpoint differs from the registered confirmation')
        run.checked(saved['artifact'])
        beat(state='recording_frozen', recording=seq)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'infer', 'worker'))
    p.add_argument('--slot', type=int, choices=range(N))
    p.add_argument('--lock-fd', type=int)
    a = p.parse_args()
    register()
    if a.phase == 'register':
        return
    confirm.require_committed((*confirm.FILES, confirm.REG, REG, SCRIPT, BENCH))
    manifest, _, packet = confirm.register()
    admission = confirm.admit()
    confirm.require_committed((confirm.ADMISSION,))
    if a.phase == 'worker':
        os.fstat(a.lock_fd)
        return worker(a.slot, manifest, packet, admission)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'parallel_infer.lock').open('a') as coordinator:
        fcntl.flock(coordinator, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (PRIVATE/'process.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fd = lock.fileno()
            procs = [subprocess.Popen([sys.executable, str(SCRIPT), 'worker', '--slot', str(i),
                '--lock-fd', str(fd)], cwd=ROOT, pass_fds=(fd,)) for i in range(N)]
            codes = [p.wait() for p in procs]
            if any(codes):
                raise RuntimeError(f'Confirmation workers failed; resume verified recording checkpoints: {codes}')
        subprocess.run([sys.executable, str(Path(confirm.__file__)), 'infer'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
