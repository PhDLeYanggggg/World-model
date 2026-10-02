"""Replay frozen forests and test a single causal decoder factor on source rows."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 runtime required before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_component_calibration as parent
from scripts import verify_m3w_selected_set_readout as independent
from src.world_model import m3w_forest_projection as api

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_forest_projection_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_forest_projection_v1'
CONFIG = ROOT/'configs/m3w_european_forest_projection_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == payload, 'Immutable evidence differs: '+str(path)
    else:
        with path.open('x') as f:
            f.write(payload)


def registration():
    parent.registration()
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_forest_projection'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_forest_projection.py']
    return dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},
        parent_freeze_sha256=sha(parent.PUBLIC/'calibration_freeze.json'),
        parent_seed_freeze_sha256=sha(parent.parent.PUBLIC/'training_freeze.json'),
        source_heads=72, new_parameter_updates=0, independent_roles_read=False)


def check_scalars(actual, expected):
    count = 0
    assert actual.keys() == expected.keys()
    for key, a in actual.items():
        b = expected[key]
        if isinstance(a, float) and isinstance(b, float):
            assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (key, a, b)
        else:
            assert a == b, (key, a, b)
        count += 1
    return count


def forecasts(state, x, env):
    model_api = parent.forest.api
    z, support = model_api.causal_inputs(x, env, state['preprocess'])
    state['model'].set_params(n_jobs=1)
    pr = state['preprocess']
    extended = state['model'].predict(z)/model_api.FACTORS*pr['rms']*pr['scale']
    np.testing.assert_allclose(extended[:, 5:], parent.core.signed(extended[:, :5]), rtol=2e-6, atol=1e-7)
    raw = extended[:, :5]
    return raw, model_api.project_moments(raw, env), api.harm_first(raw, env), support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register', 'pilot', 'run'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    reg = registration()
    if args.phase == 'register':
        once(PUBLIC/'registration.json', reg)
        print(json.dumps(dict(registration_sha256=sha(PUBLIC/'registration.json'))))
        return
    assert json.loads((PUBLIC/'registration.json').read_text()) == reg
    parent.base.inter.committed(PUBLIC/'registration.json')
    cfg = json.loads(CONFIG.read_text())
    assert cfg['risk_budget'] == .02 and not cfg['threshold_search'] and not cfg['independent_roles_read']
    assert not (PUBLIC/'complete.json').exists(), 'Completed; use existing verified evidence'
    parent.core.torch.set_num_threads(cfg['cpu_threads'])
    parent.core.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fits, calibrations = parent.parent.docs(), parent.docs()
        _, _, data, jobs, oid, _, _, _ = parent.inner.old.load()
        loaded_seconds = time.monotonic()-start
        groups, refs = [], []
        used = arithmetic_checks = 0
        pilot_done = False
        for c in parent.parent.contexts(data, jobs, oid):
            for source in parent.inner.sources(c):
                group = c['name']+'_fit_'+source
                at, ids, x, env, y, _, upstream = parent.inner.training_arrays(c, data, source)
                _, val, partition = parent.forest.parent.api.source_partition(
                    data['recordings'][ids], data['frames'][ids], source)
                vid, ev, yv = ids[val], env[val], y[val]
                moving, rec, frames = c['moving'][at][val], data['recordings'][vid], data['frames'][vid]
                for seed in cfg['head_seeds']:
                    assert time.monotonic()-start < cfg['hard_runtime_limit_seconds'], '12h cap; preserve resume files'
                    old, cal = fits[group, seed], calibrations[group, seed]
                    state = joblib.load(ROOT/old['checkpoint']['path'])
                    assert state['identity']['upstream'] == upstream and old['partition'] == partition
                    raw, projected, new, support = forecasts(state, x[val], ev)
                    h = parent.base.inter.array_hash
                    assert h(vid) == old['validation']['ids_hash'] == cal['identity']['validation_ids_hash']
                    assert h(projected) == old['validation']['prediction_hashes']['forest'] == cal['identity']['prediction_hash']
                    assert h(yv) == cal['identity']['target_hash'] and h(ev) == cal['identity']['envelope_hash']
                    # Repeat actual tree inference, not just the aggregate readout.
                    again = forecasts(state, x[val], ev)
                    for left, right in zip((raw, projected, new, support), again):
                        np.testing.assert_array_equal(left, right)
                    result, action, retained = api.audit(raw, projected, new, yv, ev, moving, support, rec, frames)
                    replay, _, _ = api.audit(*again[:3], yv, ev, moving, again[3], rec, frames)
                    assert result == replay
                    assert h(action) == old['validation']['action_hashes']['forest']
                    arithmetic_checks += check_scalars(result['original'], old['validation']['completion_screen'])
                    for key, chosen in [('original', action), ('harm_first', retained), ('removed', action & ~retained)]:
                        arithmetic_checks += check_scalars(result[key], independent.scalar_bounds(yv, chosen, ev))
                    item = dict(group=group, source=source, head_seed=seed, checkpoint=old['checkpoint'],
                        registration_sha256=sha(PUBLIC/'registration.json'), result_source='fresh_run',
                        inputs_source='cached_verified', partition=partition,
                        hashes={k:h(a) for k, a in dict(ids=vid, targets=yv, raw=raw, projected=projected,
                            harm_first=new, original_action=action, harm_first_action=retained).items()}, result=result)
                    path = PUBLIC/'groups'/(group+'_head'+str(seed)+'.json')
                    if args.phase == 'run':
                        if path.exists() and not args.resume:
                            raise RuntimeError('Existing group requires --resume')
                        once(path, item)
                        used += path.stat().st_size
                        assert used < cfg['aggregate_output_cap_bytes']
                        refs.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
                    groups.append(item)
                    beat = dict(pid=os.getpid(), phase=args.phase, groups=len(groups),
                        seconds=time.monotonic()-start, last=group+'_head'+str(seed),
                        independent_arithmetic_checks=arithmetic_checks,
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024))
                    (PRIVATE/'heartbeat.json').write_text(json.dumps(beat)+'\n')
                    print(json.dumps(beat), flush=True)
                    if args.phase == 'pilot':
                        pilot_done = True
                        break
                if pilot_done:
                    break
            if pilot_done:
                break
        receipt = dict(beat, loading_seconds=loaded_seconds,
            cached_verified_checkpoints=len(groups), exact_full_inference_replay=True,
            exact_aggregate_replay=True, numerical_arrays_written=False, checkpoint_written=False,
            result_source='fresh_run_frozen_decoder_control', new_parameter_updates=0,
            architecture=platform.machine(), torch=parent.core.torch.__version__, cpu_threads=4, num_workers=0,
            available_disk_bytes=shutil.disk_usage(ROOT).free, cache_reserve_bytes=cfg['no_numeric_cache_below_reserve_bytes'])
        if args.phase == 'pilot':
            receipt['first_group_is_not_representative_runtime_upper_bound'] = True
            receipt['group_readout_bytes'] = len(json.dumps(groups[0]).encode())
            once(PUBLIC/'pilot.json', receipt)
        else:
            assert len(groups) == cfg['source_heads']
            summary = api.summarize(groups, cfg)
            once(PUBLIC/'summary.json', summary)
            once(PUBLIC/'complete.json', dict(receipt, groups=refs, summary_sha256=sha(PUBLIC/'summary.json')))
            print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
