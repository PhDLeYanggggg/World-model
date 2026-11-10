"""Materialize past inputs and exercise frozen predictors; never join outcomes."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_songdo_first_day_vehicle_bank as transfer
from scripts import build_m3w_songdo_prefix_pack as storage
from scripts import intake_m3w_pneuma as source
from src.data_unification import m3w_pneuma_prefix as model


np, torch = transfer.np, transfer.torch
OUT = source.PUBLIC.parent/'pneuma_prefix_pilot_v1'
LOCAL = source.ROOT/'data/stage_cvpr2027_experiments/pneuma_prefix_pilot_v1'
STRIDE = 10


def read(path):
    return json.loads(path.read_text())


def beating(phase, n, name):
    tmp = LOCAL/'heartbeat.pending'
    tmp.write_text(json.dumps(dict(pid=os.getpid(), phase=phase, completed=n, total=6, file=name))+'\n')
    tmp.replace(LOCAL/'heartbeat.json')
    print(phase, n, '/ 6', name, flush=True)


def audit_inputs():
    sample, audit = (source.PUBLIC/f'{k}.json' for k in ('samples', 'support'))
    src, ref = read(sample), read(audit)
    if (ref['samples_sha256'] != storage.sha(sample) or set(src['members']) != set(ref['rows'])
            or src['predictive_roles_assigned'] or ref['predictive_roles_assigned'] or ref['forecasts_scored']
            or len(src['members']) != 6):
        raise ValueError('Need the verified split-neutral pNEUMA pilot')
    binding = dict(samples=storage.sha(sample), support=storage.sha(audit), driver=storage.sha(Path(__file__)),
                   prefix_parser=storage.sha(Path(model.__file__)), engineering_stride_ticks=STRIDE)
    return src['members'], ref['rows'], binding


def build():
    members, ref, binding = audit_inputs()
    files = {}
    for name, src in members.items():
        p = source.LOCAL/name
        if storage.sha(p) != src['sha256']:
            raise ValueError('Changed source member')
        stem = Path(name).stem
        dest, record = (LOCAL/f'{stem}.{s}' for s in ('prefix.npz', 'prefix.json'))
        want = ref[name]['coverage'][str(STRIDE)]
        if not record.exists():
            with p.open(encoding='utf-8-sig', newline='') as stream:
                wanted, starts, counts = model.plan(stream, STRIDE)
            if (len(starts), int(counts.sum())) != (want['past_selected_queries'], want['selected_agents']):
                raise ValueError('Past roster does not match the structural audit')
            with p.open(encoding='utf-8-sig', newline='') as stream:
                data = model.pack(stream, wanted, starts, counts, STRIDE)
            storage.write_json(record, dict(binding=binding, raw_sha256=src['sha256'], prefix_sha256=storage.save_npz(dest, data),
                                            queries=len(starts), agents=int(counts.sum())))
        row = read(record)
        if row != dict(binding=binding, raw_sha256=src['sha256'], prefix_sha256=storage.sha(dest),
                       queries=want['past_selected_queries'], agents=want['selected_agents']):
            raise ValueError('Stale prefix receipt')
        files[name] = row
        beating('prefix', len(files), name)
    storage.write_json(OUT/'prefix_receipt.json', dict(result_source='fresh_past_only_pilot', binding=binding, files=files,
        queries=sum(r['queries'] for r in files.values()), agents=sum(r['agents'] for r in files.values()),
        coordinate_frame='fixed WGS84 horizontal tangent plane; ellipsoid height set to zero',
        coordinate_origin_lat_lon=model.ORIGIN_LAT_LON, coordinate_footage_accuracy_verified=False,
        observed_steps=model.HISTORY, prediction_steps=model.FUTURE, engineering_stride_ticks=STRIDE,
        future_position_numbers_parsed=0, future_arrays_written=0, predictive_roles_assigned=False,
        formal_clock_selected=False, forecasts_scored=False, new_fit=False))


def predict():
    _, _, bound = audit_inputs()
    receipt = OUT/'prefix_receipt.json'
    ref = read(receipt)
    if (ref['binding'] != bound or ref['future_position_numbers_parsed'] or ref['future_arrays_written']
            or ref['predictive_roles_assigned'] or ref['forecasts_scored']):
        raise ValueError('Need verified unscored pilot prefixes')
    nets, weights, frozen = transfer.weights()
    binding = dict(prefix_receipt=storage.sha(receipt), donor_freeze=frozen, checkpoints=weights,
                   predictor=storage.sha(Path(transfer.__file__)))
    files = {}
    for name, src in ref['files'].items():
        stem = Path(name).stem
        p = LOCAL/f'{stem}.prefix.npz'
        if storage.sha(p) != src['prefix_sha256']:
            raise ValueError('Changed pilot prefix')
        dest, record = (LOCAL/f'{stem}.{s}' for s in ('forecast.npz', 'forecast.json'))
        if not record.exists():
            with np.load(p, allow_pickle=False) as a:
                h, offsets = a['history'], a['offsets']
            if h.shape != (src['agents'], model.HISTORY, 2) or len(offsets) != src['queries']+1:
                raise ValueError('Changed pilot roster')
            forecasts = transfer.forecasts(h, nets)
            storage.write_json(record, dict(prefix_sha256=src['prefix_sha256'], binding=binding,
                forecast_sha256=storage.save_npz(dest, forecasts), queries=src['queries'], agents=src['agents']))
        row = read(record)
        if row != dict(prefix_sha256=src['prefix_sha256'], binding=binding, forecast_sha256=storage.sha(dest),
                       queries=src['queries'], agents=src['agents']):
            raise ValueError('Unbound forecast packet')
        files[name] = {k: v for k, v in row.items() if k != 'binding'}
        beating('forecast', len(files), name)
    storage.write_json(OUT/'forecast_receipt.json', dict(result_source='fresh_frozen_past_only_inference', binding=binding,
        files=files, queries=ref['queries'], agents=ref['agents'], families=['cv', 'ols4', 'median4', 'opendd_completion'],
        neural_seeds=list(nets), checkpoints_verified=len(weights), future_index_loaded=False,
        future_coordinates_loaded=False, forecasts_scored=False, train_updates=0, policy_selected=False,
        predictive_roles_assigned=False, formal_clock_selected=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('prefix', 'forecast'))
    a = cli.parse_args()
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 .venv-pytorch')
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (build if a.phase == 'prefix' else predict)()
