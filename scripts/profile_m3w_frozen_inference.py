"""Measure the registered inference route on frozen development prefixes only."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dronecrowd_calibration as cal

run, native, np, torch = cal.run, cal.native, cal.np, cal.torch
PUBLIC = cal.PUBLIC.parent/'frozen_inference_profile_v1'
PRIVATE = cal.PRIVATE.parent/PUBLIC.name
LAYOUTS = ((4, 256), (2, 256), (8, 256), (4, 512), (4, 1024), (4, 256))


def read_prefixes():
    d = run.population()
    artifacts = run.read(native.POPULATION)['files']
    ref = {k: np.load(native.verified(artifacts[k]), mmap_mode='r', allow_pickle=False)
           for k in ('geometry', 'agents', 'frames', 'global_ids')}
    np.testing.assert_array_equal(d['global_ids'], ref['global_ids'])
    cells = {}
    for idx in d['groups']:
        group = str(d['sites'][idx[0]])
        cells.setdefault(group, []).append(idx)
    chosen = []
    for g, blocks in sorted(cells.items()):
        ordered = sorted(blocks, key=lambda q: (len(q), int(q[0])))
        if len(ordered) < 2:
            raise ValueError('Expected two distinct density probes in each source')
        chosen.extend((ordered[len(ordered)//2-1], ordered[-1]))
    h, g = [], []
    b = {k: [] for k in native.BASELINES}
    start = time.perf_counter()
    for idx in chosen:
        frame = int(ref['frames'][idx[0]])
        if not np.all(ref['frames'][idx] == frame):
            raise ValueError('Expected one query frame')
        past = d['history'][idx]
        mask = np.ones(past.shape[:2], dtype=bool)
        inp = dict(positions=past, history_mask=mask, history_complete=mask.all(1),
                   observation_frames=np.arange(frame-7, frame+1), query_frame=frame,
                   agent_ids=ref['agents'][idx])
        packed = native.prepare_prefix(inp)
        np.testing.assert_array_equal(packed.geometry, ref['geometry'][idx])
        np.testing.assert_array_equal(packed.agent_id, ref['agents'][idx])
        g.append(packed.geometry)
        h.append(past)
        for k, p in packed.baselines.items():
            b[k].append(p)
    seconds = time.perf_counter()-start
    return h, np.concatenate(g), {k: np.concatenate(v) for k, v in b.items()}, dict(
        queries=len(chosen), sources=len(cells), rows=sum(map(len, chosen)),
        agents_per_query=[min(map(len, chosen)), max(map(len, chosen))], prefix_pack_seconds=seconds,
        geometry_sha256=native.array_hash(np.concatenate(g)))


def pass_layout(h, geometry, baselines, networks, threads, batch):
    torch.set_num_threads(threads)
    end = np.r_[0, np.cumsum(list(map(len, h)))]
    paths, decisions, selected = {}, [], []
    by_model, decide_time = {}, 0.
    for seed, (nets, coef) in sorted(networks.items()):
        paths[seed] = {}
        for family, net in nets.items():
            start = time.perf_counter()
            paths[seed][family] = native.predict_relative(net, geometry, batch_size=batch)
            by_model[family] = by_model.get(family, 0.)+time.perf_counter()-start
        bank = np.stack([paths[seed][k] if k in nets else baselines[k] for k in run.NAMES], 1)
        cv, ols = baselines[cal.BASE], baselines[cal.OLS]
        start = time.perf_counter()
        for past, lo, hi in zip(h, end[:-1], end[1:], strict=True):
            choice = cal.model.decide(past, bank[lo:hi], cv[lo:hi], ols[lo:hi], coef)
            decisions.append(choice)
            selected.append(np.stack([run.model.selected_paths(row, bank[lo:hi], cv[lo:hi]) for row in choice]))
        decide_time += time.perf_counter()-start
    return paths, decisions, selected, dict(threads=torch.get_num_threads(), batch_size=batch,
        model_seconds=by_model, decision_seconds=decide_time, core_seconds=sum(by_model.values())+decide_time)


def compare(reference, result):
    a, ac, ap = reference[:3]
    b, bc, bp = result[:3]
    deltas = {}
    for seed, nets in a.items():
        for k, v in nets.items():
            delta = v.astype(float)-b[seed][k]
            d = deltas.setdefault(k, dict(changed_values=0, max_abs_difference_px=0., bitwise=True))
            d['changed_values'] += int(np.count_nonzero(delta))
            d['max_abs_difference_px'] = max(d['max_abs_difference_px'], float(np.abs(delta).max()))
            d['bitwise'] &= native.array_hash(v) == native.array_hash(b[seed][k])
    changed = sum(int(np.count_nonzero(v != w)) for v, w in zip(ac, bc, strict=True))
    changed_paths = sum(int(np.count_nonzero(v != w)) for v, w in zip(ap, bp, strict=True))
    return dict(network_parity=deltas, changed_actions=changed, changed_selected_path_values=changed_paths,
                exact=all(r['bitwise'] for r in deltas.values()) and not changed and not changed_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', action='store_true', required=True)
    parser.add_argument('--output', type=Path, default=PUBLIC/'profile.json',
                        help='New receipt path; measured timing is not deterministic.')
    args = parser.parse_args()
    if platform.machine() != 'arm64' or sys.platform != 'darwin':
        raise RuntimeError('The timing comparison is specific to native Apple Silicon')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output.exists():
            raise ValueError('Use a new output path for a separate timing run')
        _, _, packet = cal.register()
        weights = cal.networks(packet)
        h, g, b, roster = read_prefixes()
        start = time.perf_counter()
        outputs, reference = [], None
        for threads, batch in LAYOUTS:
            row = pass_layout(h, g, b, weights, threads, batch)
            if reference is None:
                reference = row
            report = dict(row[3], **compare(reference, row))
            outputs.append(report)
            status = dict(pid=os.getpid(), completed_layouts=len(outputs), elapsed=time.perf_counter()-start, **report)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', status)
            print(status, flush=True)
        run.once(args.output, run.stable(dict(roster=roster, layouts=outputs,
            registered_calibration_sha256=run.sha(cal.REG), checkpoint_lineage=run.read(cal.REG)['exact_predictors'],
            version=torch.__version__, interop_threads=torch.get_num_interop_threads(), workers=0,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, elapsed_seconds=time.perf_counter()-start,
            platform=platform.machine(), concurrent_calibration_unchanged=True, development_prefix_only=True,
            labels_read=False, external_windows_read=False, current_protocol_changed=False)))


if __name__ == '__main__':
    main()
