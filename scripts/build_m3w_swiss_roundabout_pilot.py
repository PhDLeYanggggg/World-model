"""Prepare observed prefixes and unscored motion forecasts on the Swiss source."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import fetch_m3w_swiss_roundabouts as source
from src.data_unification import m3w_swiss_roundabout_prefix as model
from src.data_unification import m3w_swiss_roundabout_support as support


np = model.np
STRIDE = 12
LOCAL = source.ROOT/'data/stage_cvpr2027_experiments/swiss_roundabouts_source_intake_v1'


def read(path):
    return json.loads(path.read_text())


def save(path, values):
    tmp = path.with_suffix('.pending')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)
    return source.digest(path)


def beat(phase, name, count):
    tmp = LOCAL/'heartbeat.pending'
    tmp.write_text(json.dumps(dict(pid=os.getpid(), phase=phase, file=name, completed=count, total=18))+'\n')
    tmp.replace(LOCAL/'heartbeat.json')
    print(phase, count, '/ 18', name, flush=True)


def prefixes():
    sp, ap = (source.OUT/f'{k}.json' for k in ('source', 'support'))
    published, audit = read(sp), read(ap)
    if (audit['source_receipt_sha256'] != source.digest(sp)
            or audit['scan_code_sha256'] != source.digest(Path(support.__file__))
            or audit['roles_assigned'] or audit['forecasts_scored'] or audit['numeric_coordinates_parsed']):
        raise ValueError('Require the original structure-only source audit')
    binding = dict(source=source.digest(sp), audit=source.digest(ap), driver=source.digest(Path(__file__)),
                   reader=source.digest(Path(model.__file__)), engineering_stride=STRIDE)
    packets = {}
    for x in published['members']:
        name = x['key']
        if name not in audit['members']:
            continue
        raw = source.LOCAL/name
        if source.digest(raw) != x['sha256']:
            raise ValueError('Changed released coordinates')
        p, r = (LOCAL/f'{raw.stem}.{k}' for k in ('prefix.npz', 'prefix.json'))
        expected = audit['members'][name]['coverage'][str(STRIDE)]
        identity = dict(binding=binding, raw_sha256=x['sha256'], queries=expected['queries'], rows=expected['target_rows'])
        if not r.exists():
            with raw.open(encoding='utf-8-sig', newline='') as f:
                segments, _ = support.scan(f)
            wanted, meta = model.plan(segments, STRIDE)
            if len(meta['current_tick']) != identity['queries'] or len(meta['agent_id']) != identity['rows']:
                raise ValueError('Past roster changed from the structure-only audit')
            with raw.open(encoding='utf-8-sig', newline='') as f:
                v = model.pack(f, segments, wanted, meta)
            source.write(r, dict(identity, prefix_sha256=save(p, v)))
        checked = read(r)
        if checked != dict(identity, prefix_sha256=source.digest(p)):
            raise ValueError('Stale prefix artifact')
        packets[name] = checked
        beat('prefix', name, len(packets))
    if len(packets) != 18:
        raise ValueError('Incomplete prefix bank')
    source.write(source.OUT/'prefix_receipt.json', dict(result_source='fresh_observed_input_rebuild', files=packets,
        binding=binding, queries=sum(v['queries'] for v in packets.values()), rows=sum(v['rows'] for v in packets.values()),
        feature_array='history only; origin and roster identifiers are bookkeeping',
        origin_wgs84_lat_lon=model.ORIGIN, track_class_used_as_input=False,
        future_coordinates_parsed=False, formal_stride_selected=False, roles_assigned=False, forecasts_scored=False))


def forecasts():
    from scripts import build_m3w_songdo_first_day_vehicle_bank as donor

    torch = donor.torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    src = source.OUT/'prefix_receipt.json'
    d = read(src)
    if d['future_coordinates_parsed'] or d['track_class_used_as_input'] or d['roles_assigned'] or d['forecasts_scored']:
        raise ValueError('Require observed-only and unassigned prefixes')
    nets, checksums, freeze = donor.weights()
    binding = dict(prefix=source.digest(src), checkpoints=checksums, donor_freeze=freeze,
                   predictor=source.digest(Path(donor.__file__)), driver=source.digest(Path(__file__)))
    records = {}
    for name, x in d['files'].items():
        stem = Path(name).stem
        p, f, r = (LOCAL/f'{stem}.{k}' for k in ('prefix.npz', 'forecast.npz', 'forecast.json'))
        if source.digest(p) != x['prefix_sha256']:
            raise ValueError('Changed model inputs')
        identity = dict(binding=binding, prefix_sha256=x['prefix_sha256'], queries=x['queries'], rows=x['rows'])
        if not r.exists():
            with np.load(p, allow_pickle=False) as arrays:
                history = arrays['history']
            if history.shape != (x['rows'], 8, 2):
                raise ValueError('Unexpected history shape')
            values = donor.forecasts(history, nets)
            source.write(r, dict(identity, forecast_sha256=save(f, values)))
        checked = read(r)
        if checked != dict(identity, forecast_sha256=source.digest(f)):
            raise ValueError('Stale forecast artifact')
        records[name] = {k: v for k, v in checked.items() if k != 'binding'}
        beat('forecast', name, len(records))
    source.write(source.OUT/'forecast_receipt.json', dict(result_source='fresh_frozen_past_only_inference',
        files=records, binding=binding, queries=d['queries'], rows=d['rows'], seeds=list(nets),
        checkpoints_verified=len(checksums), forecast_families=['cv', 'ols4', 'median4', 'opendd_vehicle'],
        future_coordinates_parsed=False, formal_stride_selected=False, roles_assigned=False, forecasts_scored=False,
        training_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prefix', 'forecast'))
    args = parser.parse_args()
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use the native arm64 .venv-pytorch interpreter')
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prefixes if args.phase == 'prefix' else forecasts)()
