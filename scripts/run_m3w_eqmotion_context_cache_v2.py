"""Extract original-packet frozen interaction features with complete exact replay."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_eqmotion_context_cache as old
from src.world_model.m3w_frozen_context_source import original_context
from src.world_model.m3w_frozen_interaction_capture import InteractionCapture
from src.world_model.m3w_native_forecast import pack_geometry

NAME = 'european_eqmotion_context_cache_v2'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_eqmotion_context_cache_v2.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_frozen_context_source.py',
         ROOT/'tests/test_m3w_frozen_context_source.py', PUBLIC/'protocol.md')
sha, once, write, artifact, checked = old.sha, old.once, old.write, old.artifact, old.checked


def admission(register=False):
    cfg = json.loads(CONFIG.read_text())
    previous, architecture, designs, controller = old.admission()
    if {k: v for k, v in cfg.items() if k != 'name'} != {k: v for k, v in previous.items() if k != 'name'}:
        raise ValueError('Only the original-packet input binding changes')
    directory = old.parent.forecaster.source.PRIVATE/'packed'
    receipt = json.loads((directory/'receipt.json').read_text())
    names = ('geometry', 'sites', 'recordings', 'frames', 'agents', 'history', 'origin')
    refs = {k: dict(path=str((directory/(k+'.npy')).relative_to(ROOT)), sha256=receipt['arrays'][k]) for k in names}
    packed = {k: np.load(checked(ref), mmap_mode='r', allow_pickle=False) for k, ref in refs.items()}
    source = original_context(controller, packed)
    diagnosis = old.PUBLIC/'input_parity_diagnosis.json'
    evidence = json.loads(diagnosis.read_text())
    if not evidence['all_original_input_predictions_exact'] or len(evidence['cells']) != 27:
        raise ValueError('Specific original-input diagnosis required before repair')
    definition = dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
                      prior_registration=artifact(old.REG), prior_failed_pilot=artifact(old.PRIVATE/'pilot.log'),
                      diagnosis=artifact(diagnosis), diagnosis_code=artifact(ROOT/'scripts/diagnose_m3w_eqmotion_context_parity.py'),
                      original_forecaster_packet=refs, old_runner=artifact(Path(old.__file__)),
                      passive_capture=artifact(ROOT/'src/world_model/m3w_frozen_interaction_capture.py'),
                      processed_rows=sum(len(d['full_ids']) for d in designs), stored_rows=897060,
                      predicted_latent_bytes=897060*64*4, new_optimizer_updates=0, future_labels_used=False,
                      independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    if register:
        once(REG, definition)
    if json.loads(REG.read_text()) != definition:
        raise ValueError('Frozen corrected extraction definition changed')
    if not register:
        for p in (*FILES, REG, ROOT/'scripts/diagnose_m3w_eqmotion_context_parity.py', diagnosis):
            blob = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))], cwd=ROOT, capture_output=True, check=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(p):
                raise ValueError('Commit complete correction before execution')
    return cfg, architecture, designs, source


def runtime(cfg, phase):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged cache cursor')))
    def guard():
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h phase boundary; resume unchanged')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file()) > cfg['private_cache_cap_bytes']:
            raise OSError('Preserve1GiB cache cap')
    def beat(**kw):
        guard()
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     phase=phase, elapsed_seconds=time.monotonic()-start, **kw)
        write(PRIVATE/'heartbeat.json', value)
        with (PRIVATE/'events.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    def receipt():
        return dict(pid=os.getpid(), seconds=time.monotonic()-start,
                    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, registration_sha256=sha(REG))
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, interop=1, workers=0, resource_probing=False)
    return beat, guard, receipt


def pilot(cfg, architecture, designs, source, beat, guard, receipt):
    if (PUBLIC/'pilot.json').exists():
        raise ValueError('Complete real pilot immutable')
    cells, projection = [], 0.
    for design in designs:
        model = old.load_model(architecture, design)
        ids = design['full_ids']
        n = (len(ids)+127)//128
        elapsed = rows = 0
        with np.load(checked(design['forecast']), allow_pickle=False) as z:
            expected = z['prediction'].copy()
        with InteractionCapture(model, suffix=cfg['layer_suffix']) as capture:
            module_name = capture.name
            for offset in sorted({0, n//2*128, (n-1)*128}):
                guard()
                batch = ids[offset:offset+128]
                inputs = pack_geometry(source['geometry'][batch])
                rng = torch.get_rng_state().clone()
                start = time.monotonic()
                p, h = capture.run(inputs)
                elapsed += time.monotonic()-start
                rows += len(batch)
                q, j = capture.run(inputs)
                with torch.no_grad():
                    plain = model(inputs)
                np.testing.assert_array_equal(p.numpy(), expected[offset:offset+128])
                for a, b in ((p, q), (p, plain), (h, j), (torch.get_rng_state(), rng)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        estimate = elapsed/rows*len(ids)
        projection += estimate
        cells.append(dict(trial=design['trial'], module=module_name, batches=3, processed_rows=rows,
                          actual_first_capture_seconds=elapsed, projected_full_capture_seconds=estimate,
                          parent_predictions_exact=True, repeated_hidden_exact=True, passive_forecast_and_RNG_exact=True))
        beat(state='pilot_producer_complete', trial=design['trial'], complete=len(cells), total=9)
        del model, expected
        gc.collect()
    once(PUBLIC/'pilot.json', dict(**receipt(), cells=cells, actual_optimizer_updates=0, all9_real_pilots_exact=True,
                                 projected_full_capture_seconds=projection, projection_excludes_loading_IO_and_final_hashes=True,
                                 stored_rows=897060, predicted_latent_bytes=897060*64*4, feature_lift_evaluated=False))


def cache(cfg, architecture, designs, source, beat, guard, receipt, resume, verify=False):
    end = PUBLIC/('verification.json' if verify else 'cache_freeze.json')
    if end.exists():
        raise ValueError('Complete cache phase immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['all9_real_pilots_exact']:
        raise ValueError('All nine real pilots required')
    refs = []
    for design in designs:
        trial = design['trial']
        home = PRIVATE/'arrays'/trial
        home.mkdir(parents=True, exist_ok=True)
        meta = PUBLIC/'caches'/(trial+'.json')
        ids, keep = design['full_ids'], design['keep_ids']
        identity = dict(registration_sha256=sha(REG), trial=trial, checkpoint=design['checkpoint'], forecast=design['forecast'],
                        kept_ids_sha256=old.parent.prior.frozen.fingerprint(keep),
                        full_ids_sha256=old.parent.prior.frozen.fingerprint(ids), dimension=64, batch_size=128)
        if meta.exists() and resume and not verify:
            saved = json.loads(meta.read_text())
            if saved['identity'] != identity:
                raise ValueError('Completed producer identity mismatch')
            checked(saved['latent'])
            checked(saved['ids'])
            refs.append(artifact(meta))
            continue
        cursor, array, idfile = home/'progress.json', home/'latent.npy', home/'ids.npy'
        offset = stored = 0
        if verify:
            if not (PUBLIC/'cache_freeze.json').exists():
                raise ValueError('Complete cache required before replay')
            saved = json.loads(meta.read_text())
            if saved['identity'] != identity:
                raise ValueError('Replay identity mismatch')
            data = np.load(checked(saved['latent']), mmap_mode='r', allow_pickle=False)
            np.testing.assert_array_equal(np.load(checked(saved['ids']), allow_pickle=False), keep)
        elif cursor.exists():
            if not resume:
                raise ValueError('Partial cache requires explicit unchanged resume')
            state = json.loads(cursor.read_text())
            if state['identity'] != identity:
                raise ValueError('Partial identity mismatch')
            offset, stored = state['next_offset'], state['stored_rows']
            data = np.load(array, mmap_mode='r+', allow_pickle=False)
            np.testing.assert_array_equal(np.load(idfile, allow_pickle=False), keep)
        else:
            if array.exists():
                raise ValueError('Unbound partial array cannot be reused')
            old.parent.prior.frozen.inputs.store_array(idfile, keep)
            data = np.lib.format.open_memmap(array, mode='w+', dtype=np.float32, shape=(len(keep), 64))
            data.flush()
            write(cursor, dict(identity=identity, next_offset=0, stored_rows=0))
        if data.shape != (len(keep), 64) or data.dtype != np.float32:
            raise ValueError('Fixed cache shape and dtype required')
        model = old.load_model(architecture, design)
        with np.load(checked(design['forecast']), allow_pickle=False) as z:
            expected = z['prediction'].copy()
        def save():
            data.flush()
            write(cursor, dict(identity=identity, next_offset=offset, stored_rows=stored))
        try:
            with InteractionCapture(model, suffix=cfg['layer_suffix']) as capture:
                for start in range(offset, len(ids), 128):
                    guard()
                    batch = ids[start:start+128]
                    p, h = capture.run(pack_geometry(source['geometry'][batch]))
                    np.testing.assert_array_equal(p.numpy(), expected[start:start+128])
                    positions, use = old.selected_positions(batch, keep)
                    values = h.numpy()[use]
                    if verify:
                        np.testing.assert_array_equal(values, data[positions])
                    else:
                        data[positions] = values
                    offset = start+len(batch)
                    stored += len(positions)
                    if (start//128+1) % cfg['checkpoint_every_batches'] == 0 or offset == len(ids):
                        if not verify:
                            save()
                        beat(state='latent_replay' if verify else 'latent_cache', trial=trial, processed_rows=offset,
                             total_processed_rows=len(ids), stored_rows=stored, total_stored_rows=len(keep))
        except (KeyboardInterrupt, TimeoutError, OSError):
            if not verify:
                save()
            raise
        if stored != len(keep) or not np.isfinite(data).all():
            raise ValueError('Every retained row must be finite and covered')
        if not verify:
            once(meta, dict(identity=identity, processed_rows=len(ids), stored_rows=len(keep), dimension=64, dtype='float32',
                            latent=artifact(array), ids=artifact(idfile), parent_forecast_every_row_exact=True,
                            future_labels_used=False, normalization_fitted=False, new_optimizer_updates=0))
        refs.append(artifact(meta))
        beat(state='producer_replayed' if verify else 'producer_complete', trial=trial, complete=len(refs), total=9)
        del model, data, expected
        gc.collect()
    freeze = dict(registration_sha256=sha(REG), caches=refs, stored_rows=897060,
                  processed_rows=sum(len(d['full_ids']) for d in designs), latent_bytes=897060*64*4,
                  future_labels_used=False, new_optimizer_updates=0, normalization_fitted=False,
                  independent_confirmation=False, feature_lift_evaluated=False, deployment_changed=False,
                  stage5c_executed=False, smc_enabled=False)
    once(PUBLIC/'cache_freeze.json', freeze)
    if verify:
        once(end, dict(**receipt(), all9latent_arrays_exact=True, all_parent_forecasts_exact=True,
                        cache_freeze=artifact(PUBLIC/'cache_freeze.json')))
    else:
        once(PUBLIC/'complete.json', dict(**receipt(), cache_freeze=artifact(PUBLIC/'cache_freeze.json'), all9complete=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'build', 'verify'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        values = admission(args.phase == 'register')
        if args.phase == 'register':
            return
        args0 = (*values, *runtime(values[0], args.phase))
        if args.phase == 'pilot':
            pilot(*args0)
        else:
            cache(*args0, args.resume, verify=args.phase == 'verify')


if __name__ == '__main__':
    main()
