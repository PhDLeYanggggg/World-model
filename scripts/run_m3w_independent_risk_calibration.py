"""Frozen models, dedicated calibration role, unchanged recording-risk contract."""
import argparse
import fcntl
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
import zipfile

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 Python required before Torch')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_selection_cardinal_cost as parent
from scripts import build_m3w_model_selection_inputs as old_inputs
from src.data_unification import m3w_selection_feature_bridge as bridge
from src.world_model import m3w_independent_risk_calibration as api
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv
from src.evaluation.m3w_european_squares_intake import safe_zip_member

prior = parent.prior
sha, artifact, checked, write, once = parent.sha, parent.artifact, parent.checked, parent.write, parent.once
NAME = 'european_independent_risk_calibration_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_independent_risk_calibration_v1.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_independent_risk_calibration.py',
         ROOT/'tests/test_m3w_independent_risk_calibration.py', PUBLIC/'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text()); _, producers, _ = parent.admission()
    fits = parent.fitted(); roles = json.loads(old_inputs.ROLES.read_text())
    rows = sorted((r for r in roles['recordings'] if r['role'] == 'risk_calibration_reserved'),
                  key=lambda r: r['source_member'])
    if (len(rows) != 102 or len({r['locality_group'] for r in rows}) != 12
            or cfg['multipliers'] != list(api.MULTIPLIERS) or cfg['score_family'] != 'mean_absolute'
            or cfg['per_recording_query_cap'] != 256 or cfg['positive_harm_budget'] != .02
            or cfg['easy_net_budget'] != .02 or len(producers) != 9
            or any(cfg[k] for k in ('new_training', 'confirmation_access', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Fixed full role, frozen score family and original budgets required')
    for r in rows:
        api.require_calibration_role(roles, r['source_member'])
    source = json.loads(old_inputs.MANIFEST.read_text())['private_file']
    dependencies = ('src/data_unification/m3w_model_selection_inputs.py',
        'src/data_unification/m3w_selection_feature_bridge.py', 'src/data_unification/m3w_european_squares_source.py',
        'src/world_model/m3w_frame_action_allocation.py', 'src/world_model/m3w_selection_controller.py',
        'src/world_model/m3w_multiaction_cost.py', 'src/world_model/m3w_detector_forecaster.py',
        'src/evaluation/m3w_selection_forecast_readout.py', 'scripts/run_m3w_selection_forecast.py')
    authorization = ROOT/'outputs/publication_readiness_2026_09/delegated_research_authorization_20260923.json'
    value = dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        dependencies={p: sha(ROOT/p) for p in dependencies}, roles=artifact(old_inputs.ROLES),
        archive=source, authorization=artifact(authorization), parent_training=artifact(parent.PUBLIC/'training_freeze.json'),
        frozen_models=[dict(trial=r['trial'], producer=r['checkpoint'],
            critic=fits[r['trial'], 'mean_absolute']['checkpoint'], baseline_index=r['baseline_index'],
            normalizer=r['normalizer'], easy_cut=r['easy_cut'], hard_cut=r['hard_cut']) for r in producers],
        calibration_recordings=102, calibration_localities=12, source_role_manifest_modified=False,
        score_family='mean_absolute', new_optimizer_updates=0, confirmation_access=False,
        lambda_grid=cfg['multipliers'], policy_names=list(prior.allocator.ARMS),
        labels_only_after_all_policies_frozen=True, population_risk_certificate=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    if register:
        once(REG, value)
    if json.loads(REG.read_text()) != value:
        raise ValueError('Calibration registration changed')
    if not register:
        for p in (*FILES, REG):
            blob = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))],
                                  cwd=ROOT, check=True, capture_output=True).stdout
            if hashlib.sha256(blob).hexdigest() != sha(p):
                raise ValueError('Commit fixed calibration definition before raw/predictive access')
    return cfg, roles, rows, producers, fits, source


def runtime(cfg, phase):
    torch.set_num_threads(4); torch.set_num_interop_threads(1); start = time.monotonic()
    def stop(*_):
        raise KeyboardInterrupt('Resume unchanged phase from atomic checkpoints')
    signal.signal(signal.SIGTERM, stop)
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB reserve; do not reduce cohort')
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h phase boundary; unchanged resume required')
    def beat(**kw):
        guard()
        size = sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())
        if size > cfg['private_cache_cap_bytes']:
            raise OSError('Registered6GiB private cache cap')
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), phase=phase, **kw)
        write(PRIVATE/'heartbeat.json', row); print(json.dumps(row), flush=True)
    def receipt():
        return dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            registration_sha256=sha(REG))
    beat(state='runtime_ready', architecture=platform.machine(), torch=torch.__version__, threads=4, workers=0)
    return beat, guard, receipt


def home(row):
    return PRIVATE/'records'/hashlib.sha256(row['source_member'].encode()).hexdigest()


def archive_open(source):
    path = ROOT/source['path']
    if path.stat().st_size != source['bytes'] or sha(path) != source['sha256']:
        raise ValueError('Raw archive identity changed')
    return zipfile.ZipFile(path)


def raw_record(z, row):
    info = z.getinfo(row['source_member'])
    if safe_zip_member(info) != row['source_member']:
        raise ValueError('Unsafe ZIP member')
    with z.open(info) as f:
        raw, _ = read_raw_csv(f)
    if len(raw) != row['raw_rows'] or hashlib.sha256(raw.tobytes()).hexdigest() != row['rows_sha256']:
        raise ValueError('Raw structured rows changed')
    return raw


def record_data(row, roles, index):
    inputs, features, queries = api.load_record(home(row), roles, row['source_member'], sha(REG))
    n = len(features['frames']); rid = 191+index
    features.update(ids=(np.int64(rid) << np.int64(32))+np.arange(n, dtype=np.int64),
                    recordings=np.full(n, rid, np.int64), sites=np.full(n, row['locality_group']))
    return inputs, features, queries


def build_one(z, row, roles, cfg, *, verify=False):
    raw = raw_record(z, row)
    queries, support = old_inputs.api.source.choose_query_frames(raw, maximum=cfg['per_recording_query_cap'])
    if support['all_past_eligible_targets'] != row['past_eligible_k8_stride12']:
        raise ValueError('Past support differs from original role inventory')
    values = old_inputs.api.build_inputs(raw, queries); features = bridge.pack_record(values, raw)
    checks = old_inputs.prefix_checks(raw, values, queries)
    for qi in sorted({0, len(queries)//2, len(queries)-1}):
        q = int(queries[qi]); prefix = raw[raw['frame'] <= q]
        short = bridge.pack_record(old_inputs.api.build_inputs(prefix, [q]), prefix)
        at = features['frames'] == q
        for key in bridge.FIELDS:
            if key != 'input_rows':
                np.testing.assert_array_equal(short[key], features[key][at])
    del raw
    arrays = {}; directory = home(row); directory.mkdir(parents=True, exist_ok=True)
    for key, value in dict(**{'input_'+k: v for k, v in values.items()},
                           **{'feature_'+k: v for k, v in features.items()}, index_query_frames=queries).items():
        path = directory/(key+'.npy')
        if verify:
            np.testing.assert_array_equal(np.load(path, mmap_mode='r', allow_pickle=False), value)
        else:
            if shutil.disk_usage(ROOT).free-value.nbytes < cfg['disk_reserve_bytes']:
                raise OSError('Preserve disk reserve')
            old_inputs.store_array(path, value)
        arrays[key] = dict(name=path.name, shape=list(value.shape), dtype=str(value.dtype), sha256=sha(path))
    result = dict(role='risk_calibration_inputs_only', source_member=row['source_member'],
        locality_group=row['locality_group'], rows_sha256=row['rows_sha256'], registration_sha256=sha(REG),
        arrays=arrays, support=support, prefix_queries_checked=checks, bridge_prefix_queries_checked=checks,
        input_statistics=old_inputs.api.input_statistics(values), future_labels_constructed=False)
    if verify:
        if result != json.loads((directory/'receipt.json').read_text()):
            raise ValueError('Full raw/input/bridge replay mismatch')
    else:
        once(directory/'receipt.json', result)
    api.load_record(directory, roles, row['source_member'], sha(REG))
    return artifact(directory/'receipt.json')


def make_models(producers, fits):
    models = {}
    for row in producers:
        state = prior.frozen.checked_state(row); torch.manual_seed(row['seed'])
        model = prior.frozen.producers.api.ActiveDetectorForecaster(row['baseline_index'], 'history', width=64, heads=4, layers=2)
        model.load_state_dict(state['model']); model.eval()
        models[row['trial']] = (model, prior.api.read_checkpoint(checked(fits[row['trial'], 'mean_absolute']['checkpoint'])))
    return models


def inference(row, data, models, cfg, guard):
    model, critic = models[row['trial']]; norm = row['normalizer']
    view = prior.DetectorGeometry(data['geometry'], data['detector_history'],
                                 np.asarray(norm['mean']), np.asarray(norm['std']), norm['clip'])
    n = len(data['ids']); out = np.empty((n, 12, 2), np.float64)
    for lo in range(0, n, cfg['batch_size']):
        guard(); ix = np.arange(lo, min(lo+cfg['batch_size'], n))
        with torch.no_grad():
            out[ix] = model(prior.pack_geometry(view[ix])).numpy().astype(float)+data['origin'][ix, None]
    paths, causal = causal_bank(row, data, out)
    return out, prior.api.predict(critic, causal), causal['allowed']


def causal_bank(row, data, neural):
    paths = np.concatenate((np.stack([prior.baseline_numpy(data['history'], k) for k in range(6)], 1), neural[:, None]), 1)
    causal = prior.api.features(geometry=data['geometry'], detector_history=data['detector_history'],
        rollouts=paths-data['origin'][:, None, None], floor_index=row['baseline_index'],
        recordings=data['recordings'], frames=data['frames'], ids=data['ids'], sites=data['sites'])
    return paths, causal


def infer_one(r, row, data, models, cfg, guard, *, verify=False):
    directory = home(r)/row['trial']; path = directory/'inference.json'
    if path.exists() and not verify:
        ref = json.loads(path.read_text())
        if ref['registration_sha256'] != sha(REG) or ref['input_receipt'] != artifact(home(r)/'receipt.json'):
            raise ValueError('Inference identity changed')
        for k in ('neural', 'moments', 'allowed'):
            checked(ref[k])
        return artifact(path)
    neural, moments, allowed = inference(row, data, models, cfg, guard)
    directory.mkdir(parents=True, exist_ok=True)
    refs = {}
    for key, value in dict(neural=neural, moments=moments, allowed=allowed).items():
        dest = directory/(key+'.npy')
        if verify:
            np.testing.assert_array_equal(np.load(dest, mmap_mode='r', allow_pickle=False), value)
        else:
            guard(); old_inputs.store_array(dest, value)
        refs[key] = artifact(dest)
    result = dict(registration_sha256=sha(REG), input_receipt=artifact(home(r)/'receipt.json'), trial=row['trial'],
                  rows=len(neural), batch_size=cfg['batch_size'], future_labels_read=False, **refs)
    if verify:
        if result != json.loads(path.read_text()):
            raise ValueError('Fresh model/cost inference replay differs')
    else:
        once(path, result)
    return artifact(path)


def policy_one(r, row, data, cfg, beat, guard):
    directory = home(r)/row['trial']; final = directory/'policy.json'
    ref = json.loads((directory/'inference.json').read_text())
    if final.exists():
        old = json.loads(final.read_text()); checked(old['choices'])
        if old['inference'] != artifact(directory/'inference.json') or old['registration_sha256'] != sha(REG):
            raise ValueError('Frozen policy inputs changed')
        return artifact(final)
    p = np.load(checked(ref['moments']), allow_pickle=False); ok = np.load(checked(ref['allowed']), allow_pickle=False)
    progress = directory/'policy_progress.json'; matrix = directory/'choices.npy'
    info = []; choices = np.full((13, 4, len(p)), -1, np.int8)
    if progress.exists():
        old = json.loads(progress.read_text())
        if old['inference'] != artifact(directory/'inference.json') or old['registration_sha256'] != sha(REG):
            raise ValueError('Policy resume identity changed')
        checked(old['choices']); choices = np.load(matrix, allow_pickle=False); info = old['allocation']
    for j, multiplier in enumerate(api.MULTIPLIERS):
        if j < len(info):
            continue
        inner = directory/('query_checkpoint_'+str(multiplier)+'.gz')
        guard(); q = api.coefficient_view(p, ok, multiplier)
        out, _, metadata = prior.allocator.allocate(q, ok, data['recordings'], data['frames'], data['ids'],
            seconds=cfg['solver_seconds_per_query'],
            checkpoint=lambda v: prior.api.save_checkpoint(inner, v),
            heartbeat=lambda **v: beat(state='query_allocation', trial=row['trial'], multiplier=multiplier, **v),
            resume_state=prior.api.read_checkpoint(inner) if inner.exists() else None)
        choices[j] = np.stack([out[a] for a in prior.allocator.ARMS]).astype(np.int8)
        info.append(metadata); old_inputs.store_array(matrix, choices)
        write(progress, dict(inference=artifact(directory/'inference.json'), registration_sha256=sha(REG),
                             choices=artifact(matrix), allocation=info))
        if inner.exists():
            inner.unlink()
    once(final, dict(registration_sha256=sha(REG), inference=artifact(directory/'inference.json'),
        choices=artifact(matrix), policy_names=list(prior.allocator.ARMS), multipliers=list(api.MULTIPLIERS),
        allocation=info, labels_read=False))
    return artifact(final)


def main(phase, resume=False):
    cfg, roles, rows, producers, fits, source = admission(phase == 'register')
    if phase == 'register':
        return
    beat, guard, receipt = runtime(cfg, phase)
    finish = PUBLIC/(phase+'.json')
    if finish.exists():
        raise ValueError('Completed phase immutable')
    if phase in ('pilot', 'build', 'verify_inputs'):
        if phase == 'build' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
            raise ValueError('Real largest-record pilot required')
        selected = [max(rows, key=lambda r: r['raw_rows'])] if phase == 'pilot' else rows
        refs = []; built = cached = 0
        with archive_open(source) as z:
            for i, r in enumerate(selected):
                guard(); path = home(r)/'receipt.json'
                if path.exists() and phase != 'verify_inputs':
                    if not resume:
                        raise ValueError('Explicit resume for existing recording required')
                    api.load_record(home(r), roles, r['source_member'], sha(REG)); refs.append(artifact(path)); cached += 1
                else:
                    beat(state='raw_input_bridge', recording=i+1, total=len(selected))
                    refs.append(build_one(z, r, roles, cfg, verify=phase == 'verify_inputs')); built += 1
                beat(state='input_complete', recording=i+1, total=len(selected))
        extra = {}
        if phase == 'pilot':
            r = selected[0]; _, data, _ = record_data(r, roles, rows.index(r))
            models = make_models(producers[:1], fits)
            infer_one(r, producers[0], data, models, cfg, guard)
            before = time.monotonic(); policy_one(r, producers[0], data, cfg, beat, guard)
            extra = dict(engineering_pass=True, largest_raw_rows=r['raw_rows'], target_rows=len(data['ids']),
                one_model_full_grid_seconds=time.monotonic()-before,
                projection_is_not_full_completion=True, future_labels_constructed=False)
        once(finish, dict(**receipt(), record_receipts=refs, freshly_parsed_recordings=built,
                          cached_verified_recordings=cached, **extra))
    elif phase in ('infer', 'verify_inference', 'policy'):
        if not (PUBLIC/'verify_inputs.json').exists():
            raise ValueError('Complete raw/input replay required')
        models = make_models(producers, fits) if phase != 'policy' else None
        refs = []
        for i, r in enumerate(rows):
            _, data, _ = record_data(r, roles, i)
            for row in producers:
                guard()
                ref = (policy_one(r, row, data, cfg, beat, guard) if phase == 'policy' else
                       infer_one(r, row, data, models, cfg, guard, verify=phase == 'verify_inference'))
                refs.append(ref)
            beat(state='record_models_complete', recording=i+1, total=len(rows), model_record_views=len(refs))
        once(finish, dict(**receipt(), model_record_receipts=refs, future_labels_read=False,
                          new_optimizer_updates=0, full_cohort=102, all_nine_models=True))
    elif phase == 'labels':
        freeze = json.loads((PUBLIC/'policy.json').read_text())
        if len(freeze['model_record_receipts']) != 918:
            raise ValueError('All13x4 policies for all102 recordings and9producers must freeze before labels')
        for ref in freeze['model_record_receipts']:
            p = json.loads(checked(ref).read_text()); checked(p['choices'])
        refs = []
        with archive_open(source) as z:
            for i, r in enumerate(rows):
                guard(); path = home(r)/'labels.json'
                if path.exists():
                    if not resume:
                        raise ValueError('Explicit label resume required')
                    ref = json.loads(path.read_text())
                    if ref['policy_freeze'] != artifact(PUBLIC/'policy.json'):
                        raise ValueError('Label phase bound to all frozen policies')
                    checked(ref['target']); checked(ref['valid']); refs.append(artifact(path)); continue
                raw = raw_record(z, r); inputs, data, queries = record_data(r, roles, i)
                past, labels = old_inputs.api.source.build_recording(raw, queries)
                for k in inputs:
                    np.testing.assert_array_equal(inputs[k], past[k])
                target = labels['future_xy'][data['input_rows']]; valid = labels['future_valid'][data['input_rows']]
                refs_one = {}
                for k, value in dict(target=target, valid=valid).items():
                    dest = home(r)/(k+'.npy'); old_inputs.store_array(dest, value); refs_one[k] = artifact(dest)
                once(path, dict(policy_freeze=artifact(PUBLIC/'policy.json'), rows=len(target),
                    wholly_unknown_rows=int((~valid.any(1)).sum()), partial_future_rows=int((valid.any(1)&~valid.all(1)).sum()), **refs_one))
                refs.append(artifact(path)); beat(state='labels_constructed_after_freeze', recording=i+1, total=len(rows))
        once(finish, dict(**receipt(), label_receipts=refs, policy_freeze=artifact(PUBLIC/'policy.json'), confirmation_access=False))
    elif phase in ('eval', 'verify'):
        evaluate(cfg, roles, rows, producers, beat, guard, receipt, verify=phase == 'verify')
    else:
        raise ValueError('Unknown phase')


def evaluate(cfg, roles, rows, producers, beat, guard, receipt, *, verify):
    complete = json.loads((PUBLIC/'labels.json').read_text())
    if len(complete['label_receipts']) != 102:
        raise ValueError('Full calibration label cohort required')
    curves = [dict(multiplier=m, policies={a: dict(required_record_views=0, supported_record_views=0,
        known_all_violations=0, known_easy_violations=0, unsupported_views=0,
        worst_easy_net_percent=None, by_locality={}) for a in prior.allocator.ARMS}) for m in api.MULTIPLIERS]
    metric_refs = []
    for i, r in enumerate(rows):
        _, data, _ = record_data(r, roles, i); lr = json.loads((home(r)/'labels.json').read_text())
        target = np.load(checked(lr['target']), allow_pickle=False); valid = np.load(checked(lr['valid']), allow_pickle=False)
        for row in producers:
            guard(); directory = home(r)/row['trial']; ref = json.loads((directory/'inference.json').read_text())
            paths, causal = causal_bank(row, data, np.load(checked(ref['neural']), allow_pickle=False))
            pf = json.loads((directory/'policy.json').read_text()); choices = np.load(checked(pf['choices']), allow_pickle=False)
            p = np.load(checked(ref['moments']), allow_pickle=False)
            np.testing.assert_array_equal(causal['allowed'], np.load(checked(ref['allowed']), allow_pickle=False))
            floor = paths[:, row['baseline_index']]; be, _ = prior.metrics.errors(floor, target, valid)
            known = valid.any(1); denominator = float(be[known].sum()); scored = []
            for j, multiplier in enumerate(api.MULTIPLIERS):
                q = api.coefficient_view(p, causal['allowed'], multiplier)
                for ai, arm in enumerate(prior.allocator.ARMS):
                    c = choices[j, ai]
                    for ix in causal['groups']:
                        if not prior.allocator.feasible(q[ix], causal['allowed'][ix], c[ix]):
                            raise ValueError('Frozen policy feasibility violated')
                        if arm == 'joint_matched' and (c[ix] >= 0).sum() != (choices[j, 0, ix] >= 0).sum():
                            raise ValueError('Per-query intervention counts differ')
                    pred = prior.selected_paths(paths, c, row['baseline_index'])
                    score = prior.metrics.policy(pred, floor, paths[:, 1], target, valid, data['recordings'],
                                                 easy_cut=row['easy_cut'], hard_cut=row['hard_cut'])
                    scored.append(dict(multiplier=multiplier, arm=arm, metrics=score))
                    view = next(iter(score['recordings'].values())); bucket = curves[j]['policies'][arm]
                    bucket['required_record_views'] += 1; bucket['supported_record_views'] += int(view['supported'])
                    bucket['known_all_violations'] += int(view['known_all_risk'] is not None and view['known_all_risk'] > .02)
                    bucket['known_easy_violations'] += int(view['known_easy_risk'] is not None and view['known_easy_risk'] > .02)
                    bucket['unsupported_views'] += int(view['completion_all_risk_upper'] is None or view['completion_easy_risk_upper'] is None
                                                      or view['known_easy_net_degradation'] is None)
                    easy = view['known_easy_net_degradation']
                    if easy is not None:
                        old = bucket['worst_easy_net_percent']
                        bucket['worst_easy_net_percent'] = max(old if old is not None else -float('inf'), 100*easy)
                    site = bucket['by_locality'].setdefault(r['locality_group'], dict(denominator=0., known_numerator=0.,
                        lower_numerator=0., rows=0, switched=0))
                    pe, _ = prior.metrics.errors(pred, target, valid)
                    value = float((be[known]-pe[known]).sum())
                    distance = float(np.linalg.norm(pred-floor, axis=-1).max(1)[~known].sum())
                    site['denominator'] += denominator; site['known_numerator'] += value
                    site['lower_numerator'] += value-distance; site['rows'] += len(pred)
                    site['switched'] += int(np.any(pred != floor, axis=(1, 2)).sum())
            dest = directory/'metrics.json'
            result = dict(trial=row['trial'], source_member=r['source_member'], policy=artifact(directory/'policy.json'), scores=scored)
            if verify:
                if result != json.loads(dest.read_text()):
                    raise ValueError('Complete metric replay mismatch')
            else:
                once(dest, result)
            metric_refs.append(artifact(dest))
        beat(state='full_grid_metrics', recording=i+1, total=len(rows), metric_views=(i+1)*9*13*4)
    selection = api.select_multiplier(curves, 918)
    for curve in curves:
        for bucket in curve['policies'].values():
            sites = bucket['by_locality']; rng = np.random.default_rng(cfg['bootstrap_seed'])
            for kind in ('known', 'lower'):
                values = [100*s[kind+'_numerator']/s['denominator'] if s['denominator'] > 0 else None for s in sites.values()]
                if None in values:
                    stat = dict(point=None, CI95=None, status='not_estimable')
                else:
                    draws = rng.choice(values, size=(cfg['bootstrap_resamples'], len(values))).mean(1)
                    stat = dict(point=float(np.mean(values)), CI95=np.quantile(draws, [.025, .975]).tolist(),
                        status='conditional_in_sample_calibration_not_confirmation', independent_locality_units=len(values))
                bucket[kind+'_gain_percent'] = stat
    summary = dict(selection=selection, curves=curves, unique_recordings=102, independent_localities=12,
        repeated_pipeline_record_views=918, metric_groups=len(metric_refs)*13*4,
        calibration_outcomes_opened=True, new_optimizer_updates=0, confirmation_access=False,
        population_risk_certificate=False, complete12_ADE_claim=False, t50_or_metric_claim=False,
        deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)
    if verify:
        if summary != json.loads((PUBLIC/'summary.json').read_text()):
            raise ValueError('Calibration parameter/curves replay mismatch')
        once(PUBLIC/'verify.json', dict(**receipt(), all47736_metric_groups_exact=True,
            policy_feasibility_and_per_query_count_replayed=True, milp_reoptimized=False,
            summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json', summary)
        once(PUBLIC/'eval.json', dict(**receipt(), private_metric_receipts=metric_refs, summary=artifact(PUBLIC/'summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'build', 'verify_inputs', 'infer',
                                         'verify_inference', 'policy', 'labels', 'eval', 'verify'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True); PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args.phase, args.resume)
