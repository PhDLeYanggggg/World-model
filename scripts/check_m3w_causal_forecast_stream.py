"""Verify causal input equivalence and actual Torch chunk recovery before external roles."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import resource
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 before numerical imports')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_m3w_external_three_seed_bank as prior
from scripts.audit_m3w_dronecrowd_cross_archive import DATA
from src.data_unification import m3w_causal_forecast_stream as stream
from src.data_unification.m3w_dronecrowd_cache import DroneCrowdSceneWindows
from src.evaluation.m3w_dronecrowd_windows import PROBES, compare_inputs
from src.world_model.m3w_european_source_forecast import pack_scene

np, torch, read, once, digest, verified = prior.np, prior.torch, prior.read, prior.once, prior.digest, prior.verified
PUBLIC = prior.HOME/'m3w_causal_forecast_stream_v1'
PRIVATE = DATA/'m3w_causal_forecast_stream_v1'


def open_record(meta, item):
    folder = DATA/'dronecrowd_recordings_v1'/item['sequence_id']
    if digest(folder/'metadata.json') != item['metadata_sha256']:
        raise ValueError('Changed recording')
    return DroneCrowdSceneWindows(folder, PROBES[0], expected_xml_sha256=item['source_xml_sha256'],
        expected_provenance=meta['provenance'])


def prefix_check():
    source = prior.HOME/'dronecrowd_recordings_v1/analysis.json'
    older = prior.HOME/'dronecrowd_frozen_inputs_v1/analysis.json'
    meta, proof = read(source), read(older)
    if proof['source_manifest_sha256'] != digest(source):
        raise ValueError('Expected unchanged recording lineage')
    old_rows = {r['sequence']: r for r in proof['records']}
    old_seconds = new_seconds = 0.
    counts = dict(queries=0, visible=0, eligible=0)
    for item in meta['recordings']:
        node = item['sequence_id']; reader = open_record(meta, item)
        expected = old_rows[node]; frames = [r['frame'] for r in expected['inspected']]
        if len(reader) != expected['query_frames']:
            raise ValueError('Changed clip-bounded query schedule')
        for row in expected['inspected']:
            frame = row['frame']
            t = time.perf_counter(); original = reader.get_inputs(frame); old_seconds += time.perf_counter()-t
            t = time.perf_counter(); prefix = stream.native_prefix(reader.record, frame); new_seconds += time.perf_counter()-t
            if not compare_inputs(original, prefix):
                raise ValueError('Bounded reader changed an observed annotation prefix')
        packets = list(stream.scene_packets(reader.record, frames, pack_scene))
        for expected_packet, packet in zip(expected['inspected'], packets):
            if {k:packet[k] for k in ('frame','visible','eligible')} != expected_packet:
                raise ValueError('Changed original per-query population')
            original, indices = pack_scene(stream.adapt_past(reader.get_inputs(packet['frame'])))
            np.testing.assert_array_equal(original, packet['geometry'])
            np.testing.assert_array_equal(packet['keys'][:, 1], reader.get_inputs(packet['frame'])['agent_ids'][indices])
        counts['queries'] += len(packets)
        for k in ('visible','eligible'): counts[k] += sum(r[k] for r in packets)
    if counts != dict(queries=336, visible=48362, eligible=48183):
        raise ValueError('Keep the full previously registered compatibility probe')
    once(PUBLIC/'prefix_analysis.json', dict(recordings=len(old_rows), **counts,
        recording_manifest_sha256=digest(source), prior_prefix_sha256=digest(older),
        prefix_reader_sha256=digest(Path(stream.__file__)), exact_input_packets=True,
        future_label_api_called=False, external_forecast_inference=False, scientific_roles_assigned=False))
    return dict(**counts, old_read_seconds=old_seconds, bounded_read_seconds=new_seconds)


def schedule_check():
    source = prior.HOME/'dronecrowd_recordings_v1/analysis.json'; meta = read(source)
    identity = dict(manifest_sha256=digest(source), stream_sha256=digest(Path(stream.__file__)),
        packer_sha256=digest(ROOT/'src/world_model/m3w_european_source_forecast.py'),
        batch_rows=128, window='obs8_pred12_stride1', past_queries_only=True)
    rows = []
    for item in meta['recordings']:
        node = item['sequence_id']; mark = PRIVATE/'schedule'/f'{node}.json'
        if not mark.exists():
            reader = open_record(meta, item); counts = dict(queries=0, visible=0, eligible=0, empty=0)
            def packets():
                for q in stream.scene_packets(reader.record, reader.query_frames, pack_scene):
                    counts['queries'] += 1; counts['visible'] += q['visible']; counts['eligible'] += q['eligible']
                    counts['empty'] += q['eligible'] == 0
                    yield q
            keys, geometry = hashlib.sha256(), hashlib.sha256()
            batches = indexed = 0
            for b in stream.fixed_batches(packets()):
                keys.update(b['keys'].data); geometry.update(b['geometry'].data)
                batches += 1; indexed += len(b['keys'])
            if counts['queries'] != len(reader) or indexed != counts['eligible']:
                raise ValueError('Batch packing changed row or query coverage')
            once(mark, dict(identity=identity, sequence=node, metadata_sha256=item['metadata_sha256'],
                **counts, batches=batches, keys_sha256=keys.hexdigest(), geometry_sha256=geometry.hexdigest()))
        row = read(mark)
        if row['identity'] != identity or row['sequence'] != node or row['metadata_sha256'] != item['metadata_sha256']:
            raise ValueError('Cannot resume a different past-only schedule')
        rows.append({k:v for k,v in row.items() if k != 'identity'})
        status = dict(pid=os.getpid(), complete=len(rows), total=len(meta['recordings']), sequence=node,
            eligible=rows[-1]['eligible'])
        stream.write_json(PRIVATE/'heartbeat.json', status)
        if len(rows) % 8 == 0:
            print(status, flush=True)
    sums = {k:sum(row[k] for row in rows) for k in ('queries','visible','eligible','empty','batches')}
    if len(rows) != 112 or sums['queries'] != 31472:
        raise ValueError('Expected full unassigned input roster')
    once(PUBLIC/'schedule_analysis.json', dict(identity=identity, **sums, recordings=rows,
        dense_geometry_written=False, future_label_api_called=False, external_forecast_inference=False,
        scientific_roles_assigned=False))
    return dict(recordings=len(rows), **sums, schedule_sha256=digest(PUBLIC/'schedule_analysis.json'))


def forecast_check(batches):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    covered = read(prior.HOME/'dronecrowd_combined_forecaster_coverage_v1/analysis.json')
    coverage = {r['key'] for r in covered['trials'] if not r['still_uncovered']}
    original = read(prior.OLD/'analysis.json')['prediction_receipts']
    proof_path = prior.PUBLIC/'analysis.json'
    probe = read(proof_path); trials = [t for t in probe['trials'] if t['fold'] == 2]
    if len(trials) != 6 or any(t['key'] not in coverage for t in trials):
        raise ValueError('Need both families and three actual matched fold-2 seeds')
    pop = prior.HOME/'moving_producer_crossfit_v1/population.json'
    data = {k:np.load(verified(read(pop)['files'][k]), mmap_mode='r', allow_pickle=False)
            for k in ('global_ids','geometry')}
    ids = data['global_ids']; results = []
    for trial in trials:
        start = time.monotonic(); key = trial['key']; checkpoint = trial['checkpoint']
        bank = next(b for b in original if b['sha256'] == trial['prediction_sha256'] and
                    b['checkpoint_sha256'] == checkpoint['sha256'])
        state = torch.load(verified(checkpoint), map_location='cpu', weights_only=False)
        ident = bank['identity']
        if state['step'] != 4000 or state['identity'] != ident:
            raise ValueError('Checkpoint training identity changed')
        model = prior.SourceForecaster(ident['architecture'], ident['baseline_index']); model.load_state_dict(state['model'])
        with np.load(verified(bank), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            starts = trial['original_prediction_batch_starts'][:batches]
            expected = z['relative_prediction'][np.asarray(starts)[:, None]+np.arange(128)]
        identity = dict(key=key, checkpoint_sha256=checkpoint['sha256'], parent_bank_sha256=bank['sha256'],
            geometry_receipt_sha256=digest(pop), stream_sha256=digest(Path(stream.__file__)),
            cpu_threads=4, interop_threads=1, torch=str(torch.__version__), batch_rows=128)
        directory = PRIVATE/'A'/key
        counts = dict(fresh_rows=0, reused_rows=0)
        for index, offset in enumerate(starts):
            ix = slice(offset, offset+128)
            def infer(g): return prior.predict(model, {'geometry':g}, np.arange(len(g)), 128)
            p, cached = stream.spool_batch(directory, index, identity, ids[ix], data['geometry'][ix], infer)
            np.testing.assert_array_equal(p, expected[index])
            counts['reused_rows' if cached else 'fresh_rows'] += len(p)
        result = dict(key=key, seconds=time.monotonic()-start, **counts)
        results.append(result); print(result, flush=True)
        stream.write_json(PRIVATE/'heartbeat.json', dict(pid=os.getpid(), completed_models=len(results), **result))
    return dict(models=len(results), batches_per_model=batches, exact_probe_rows=6*128*batches, results=results,
        fresh_rows=sum(r['fresh_rows'] for r in results), reused_rows=sum(r['reused_rows'] for r in results),
        original_replay_sha256=digest(proof_path), all_equal=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prefix', 'forecast', 'schedule'))
    p.add_argument('--batches', type=int, choices=(3,8), default=8)
    args = p.parse_args(); start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase == 'forecast': summary = forecast_check(args.batches)
        elif args.phase == 'prefix': summary = prefix_check()
        else: summary = schedule_check()
        receipt = dict(pid=os.getpid(), phase=args.phase, seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **summary,
            forecast_training=False, external_label_api_called=False, external_forecast_inference=False,
            scientific_roles_assigned=False, independent_confirmation=False, stage5c_executed=False, smc_enabled=False)
        once(PUBLIC/'executions'/f'{receipt["pid"]}_{time.time_ns()}.json', receipt)
        print(receipt, flush=True)


if __name__ == '__main__':
    main()
