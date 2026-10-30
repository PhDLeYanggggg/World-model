"""Matched gain/harm learning on frozen old/new actual active proposal banks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_detector_cost as old
from scripts import run_m3w_active_detector_forecaster as producer
from src.world_model import m3w_active_bank_cost as api

NAME = 'european_active_bank_cost_v1'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09' / NAME
PRIVATE = ROOT / 'data/stage_cvpr2027_experiments' / NAME
CONFIG = ROOT / 'configs/m3w_active_bank_cost_v1.json'
REG = PUBLIC / 'registration.json'
engine, scene = old.parent.parent.parent, old.parent.parent
sha, once, write = old.sha, old.once, old.write
FILES = (Path(__file__), CONFIG, ROOT / 'src/world_model/m3w_active_bank_cost.py',
         ROOT / 'tests/test_m3w_active_bank_cost.py', ROOT / 'tests/test_m3w_active_bank_runner.py', PUBLIC / 'protocol.md')


def admission(register=False):
    cfg = json.loads(CONFIG.read_text())
    pc, values, _, store = old.admission()
    legacy = old.models()
    _, source, jobs, inputs, controls = producer.admission()
    producers = producer.models()
    contexts = values[2]['contexts']
    if (cfg['training'] != pc['training'] or cfg['seeds'] != pc['seeds'] or cfg['risk_budget'] != .02
            or cfg['variants'] != list(api.VARIANTS) or cfg['forecaster_seed'] != 43
            or cfg['representation'] != 'history' or cfg['supervision'] != 'known_only'
            or len(contexts) != 6 or any(cfg[k] for k in
                ('held_scored', 'independent_roles_opened', 'deployment_changed', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Original six contexts, three seeds, objective, budget and roles required')
    for key in ('sites', 'recordings', 'agents', 'frames', 'origin', 'history'):
        np.testing.assert_array_equal(source[key], values[3][key])
    old_check = json.loads((old.PUBLIC / 'verification.json').read_text())
    new_check = json.loads((producer.PUBLIC / 'verification.json').read_text())
    if (not old_check['exact_replay'] or old_check['groups'] != 288
            or old_check['summary_sha256'] != sha(old.PUBLIC / 'summary.json')
            or not new_check['all36groups_exact'] or not new_check['all9fresh_parent_predictions_exact']
            or new_check['summary_sha256'] != sha(producer.PUBLIC / 'summary.json')):
        raise ValueError('Complete verified old heads and actual active producers required')
    refs = []
    for key in sorted({name.split('_controller')[0] for name in contexts}):
        job = next(j for j in jobs if j['key'] == key)
        if job['old_identity']['seed'] != 43 or key not in values[5]['controls']:
            raise ValueError('The actual consumer producer seed43 is fixed')
        reference = values[5]['controls'][key]['ADE_only_prediction']
        expected = next(r['dimensionless'] for r in json.loads(
            (producer.parent.PUBLIC / 'prediction_freeze.json').read_text())['predictions'] if r['key'] == key)
        if reference != expected:
            raise ValueError('The old bank is not the actual matched dimensionless producer')
        refs.append(dict(key=key, old_prediction=reference,
            new_checkpoint=producers[key, 'history']['checkpoint'], disabled_checkpoint=producers[key, 'disabled']['checkpoint']))
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        old_training_freeze_sha256=sha(old.PUBLIC / 'training_freeze.json'),
        old_readout_verification_sha256=sha(old.PUBLIC / 'verification.json'),
        producer_training_freeze_sha256=sha(producer.PUBLIC / 'training_freeze.json'),
        producer_readout_verification_sha256=sha(producer.PUBLIC / 'verification.json'),
        detector_store_complete_sha256=store.complete_sha256,
        contexts={k: [x['packet']['identity']['source'] for x in items] for k, items in sorted(contexts.items())},
        producers=refs, fits=54, legacy_exact_fits=36, steps_per_fit=2000,
        primary_optimizer_updates=108000, extra_full_refit_updates=2000, pilot_updates=160,
        parameters=30405, causal_fields=462, forecaster_seed=43, representation='history',
        sampler='original_causal_draw_groups_including_unknowns_reused_exactly',
        preprocessing='original_physical_fitting_and_original_causal_detector_normalizers_frozen',
        source_eligibility='original_causal_coverage_gate_frozen_new_real_action_identity_recomputed',
        candidate_features_scene_costs_rebuilt=True, physical_columns_unchanged=True,
        cache_prediction_batch=128, cache_future_targets_used=False, risk_budget=.02,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel, digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT / rel) != digest:
                raise ValueError('Registered matched-bank experiment changed: ' + rel)
            engine.committed(ROOT / rel)
        engine.committed(REG)
    return cfg, values, legacy, store, (source, jobs, inputs, producers)


def runtime(cfg):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume the same atomic checkpoints')))
    def guard():
        if time.monotonic() - start > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12h boundary; resume the full registered roster')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Private checkpoint cap exceeded')
    def beat(**event):
        guard()
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        write(PRIVATE / 'heartbeat.json', value)
        with (PRIVATE / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)
    beat(state='runtime_ready', architecture=platform.machine(), device='cpu', threads=4, workers=0, resource_probing=False)
    return start, beat, guard


def cache_refs():
    path = PUBLIC / 'bank_complete.json'
    engine.committed(path)
    complete = json.loads(path.read_text())
    if complete['registration_sha256'] != sha(REG) or len(complete['banks']) != 3 or complete['future_targets_used']:
        raise ValueError('Complete frozen past-only three-producer bank required')
    output = {}
    for ref in complete['banks']:
        receipt = ROOT / ref['path']
        engine.committed(receipt)
        row = json.loads(receipt.read_text())
        if sha(receipt) != ref['sha256'] or sha(ROOT / row['prediction']['path']) != row['prediction']['sha256']:
            raise ValueError('Frozen neural bank changed')
        if not row['fresh_disabled_exact_vs_original_bank'] or row['future_targets_used']:
            raise ValueError('Actual old-bank replay and no target use required')
        output[row['producer']] = row
    if set(output) != {r['key'] for r in json.loads(REG.read_text())['producers']}:
        raise ValueError('All original consumer producers required')
    return output


def build_banks(resume=False):
    cfg, values, _, _, assets = admission()
    if (PUBLIC / 'bank_complete.json').exists():
        raise ValueError('Completed bank freeze is immutable')
    source, jobs, inputs, models = assets
    start, beat, guard = runtime(cfg)
    refs = []
    for record in json.loads(REG.read_text())['producers']:
        key = record['key']
        receipt = PUBLIC / 'banks' / (key + '.json')
        if receipt.exists():
            if not resume:
                raise ValueError('Partial bank cache requires explicit --resume')
            row = json.loads(receipt.read_text())
            if row['registration_sha256'] != sha(REG) or sha(ROOT / row['prediction']['path']) != row['prediction']['sha256']:
                raise ValueError('Completed bank receipt changed')
            refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
            continue
        job = next(j for j in jobs if j['key'] == key)
        ids = job['design']['held_ids']
        previous = engine.bank(record['old_prediction'], ids)
        learned = None
        for arm in ('disabled', 'history'):
            prepared, norm = producer.prepare_arm(source, inputs, job, arm)
            ref = models[key, arm]
            if ref['normalizer'] != norm:
                raise ValueError('Frozen producer normalization changed')
            state = torch.load(ROOT / ref['checkpoint']['path'], map_location='cpu', weights_only=False)
            torch.manual_seed(43)
            model = producer.api.ActiveDetectorForecaster(job['design']['baseline_index'], arm,
                **json.loads(producer.CONFIG.read_text())['architecture'])
            model.load_state_dict(state['model'])
            with producer.api.installed():
                prediction = producer.api.native.predict(model, prepared, ids, 128)
            if arm == 'disabled':
                np.testing.assert_array_equal(prediction, previous)
            else:
                learned = prediction
            beat(state='past_only_bank_inferred', producer=key, representation=arm, rows=len(ids), targets_read=False)
            del model, state, prepared
        path = PRIVATE / 'banks' / (key + '.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, ids=ids, prediction=learned)
        os.replace(temporary, path)
        once(receipt, dict(producer=key, rows=len(ids), prediction=dict(path=str(path.relative_to(ROOT)),
            sha256=sha(path), bytes=path.stat().st_size), ids_sha256=engine.fingerprint(ids),
            new_checkpoint=record['new_checkpoint'], old_prediction=record['old_prediction'],
            fresh_disabled_exact_vs_original_bank=True, future_targets_used=False,
            registration_sha256=sha(REG), result_source='fresh_past_only_actual_producer_inference'))
        refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
    once(PUBLIC / 'bank_complete.json', dict(pid=os.getpid(), seconds=time.monotonic() - start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, banks=refs,
        registration_sha256=sha(REG), fresh_disabled_exact_producers=3, future_targets_used=False,
        controller_held_outcomes_scored=False, independent_roles_opened=False))
    beat(state='all_three_past_only_banks_frozen')


def scene_rows(source, ids, candidate, packed, polygons):
    values = np.empty((len(ids), 18), np.float32)
    for recording in np.unique(source['recordings'][ids]):
        at = np.flatnonzero(source['recordings'][ids] == recording)
        polygon = polygons[packed['records'][int(recording)]]['vertices']
        values[at] = scene.scene_features(history=source['history'][ids[at]], polygon=polygon,
            rollout=candidate[at].astype(float) + source['origin'][ids[at], None])
    return values


def prepared_context(context, items, values, store, banks, beat):
    original_packets = old.parent.packets(context, items, values, beat)
    original = old.parent.api.training_data(original_packets, scene)
    original = old.data_for(original_packets, store, 'history', original)
    key = context.split('_controller')[0]
    job = next(j for j in values[4] if j['key'] == key)
    with np.load(ROOT / banks[key]['prediction']['path'], allow_pickle=False) as archive:
        bank_ids, predictions = archive['ids'], archive['prediction']
    np.testing.assert_array_equal(bank_ids, job['design']['held_ids'])
    source = values[3]
    new_packets = {}
    for name, packet in sorted(original_packets.items()):
        ids = packet['ids']
        # Assert this source is inside the old fitting-only detector whitelist.
        store.read(ids, 'history')
        candidate = engine.aligned_predictions(ids, bank_ids, predictions)
        switch = packet['args'][0][:, -1].astype(bool)
        old_candidate = packet['rollouts'][:, 6].astype(np.float32)
        neutral_scene = scene_rows(source, ids, old_candidate, values[7], values[8])
        neutral = api.causal_neural_fields(geometry=source['geometry'][ids], cv=packet['rollouts'][:, 1],
            floor=packet['floor_rollout'], candidate=old_candidate, floor_switch=switch, scene=neutral_scene)
        np.testing.assert_array_equal(neutral['x'], packet['x'][:, 6])
        np.testing.assert_array_equal(neutral['envelope'], packet['envelope'][:, 6])
        np.testing.assert_array_equal(neutral_scene, packet['scene'][:, 6])
        extra_scene = scene_rows(source, ids, candidate, values[7], values[8])
        fields = api.causal_neural_fields(geometry=source['geometry'][ids], cv=packet['rollouts'][:, 1],
            floor=packet['floor_rollout'], candidate=candidate, floor_switch=switch, scene=extra_scene)
        ade, fde = engine.native_errors(candidate.astype(float) + source['origin'][ids, None],
            source['target_eval'][ids], source['valid'][ids], np.ones(len(ids)))
        labels = engine.targets(packet['cv_ade'], packet['floor_ade'], ade, job['design']['easy_cut'])
        eligible = engine.old.base.shared.eligible(packet['args'])
        new_packets[name] = api.replace_packet(packet, fields, causal_eligible=eligible, labels=labels, ade=ade, fde=fde)
        beat(state='matched_bank_source_prepared', context=context, source=name, rows=len(ids),
            neutral_causal_rebuild_exact=True, labels_used_for_supervision_only=True, held_scored=False)
    changed = api.frozen_training_data(original, original_packets, new_packets)
    return original_packets, new_packets, original, changed


def checkpoint(context, seed, variant, folder='heads'):
    return PRIVATE / folder / f'{context}_seed{seed}_{variant}.pt.gz'


def fit(data, cfg, context, seed, variant, path, beat, guard, *, resume=False, stop_at=None):
    identity = dict(registration_sha256=sha(REG), context=context, variant=variant,
                    bank_freeze_sha256=sha(PUBLIC / 'bank_complete.json'))
    return api.fit(data, variant=variant, settings=cfg['training'], seed=seed, identity=identity, path=path,
        heartbeat=lambda **kw: beat(context=context, variant=variant, **kw), guard=guard, resume=resume, stop_at=stop_at)


def pilot(resume=False):
    cfg, values, _, store, _ = admission()
    banks = cache_refs()
    if (PUBLIC / 'pilot.json').exists():
        raise ValueError('Completed pilot immutable')
    start, beat, guard = runtime(cfg)
    context, items = sorted(values[2]['contexts'].items())[0]
    op, npackets, original, changed = prepared_context(context, items, values, store, banks, beat)
    states, cells = [], []
    for variant in cfg['variants']:
        data = changed if variant == 'new_neural' else original
        path = checkpoint(context, 17, variant, 'pilot')
        if not path.exists():
            fit(data, cfg, context, 17, variant, path, beat, guard, stop_at=10)
        state = fit(data, cfg, context, 17, variant, path, beat, guard, resume=True, stop_at=20)
        direct = fit(data, cfg, context, 17, variant, checkpoint(context, 17, variant, 'pilot_direct'),
                     beat, guard, resume=resume, stop_at=20)
        api.assert_scientific_equal(state, direct)
        if variant != 'new_neural':
            with old.api.installed('history'):
                prior = old.parent.api.fit(original, mode='known_only', bank=api.BANKS[variant], settings=cfg['training'],
                    seed=17, identity=dict(engineering_original_control=True), path=checkpoint(context, 17, variant, 'pilot_original'),
                    heartbeat=beat, guard=guard, resume=resume, stop_at=20)
            api.assert_scientific_equal(state, prior)
        if state['parameters'] != 30405 or state['unknown_rows_drawn'] == 0 or state['unknown_rows_with_loss']:
            raise ValueError('Real common unknown-row draws and exact old model size required')
        if not any(not torch.equal(value, state['initial_model'][name]) for name, value in state['model'].items()):
            raise ValueError('No actual cost-head update')
        states.append(state)
        cells.append(dict(variant=variant, seconds=state['seconds'], updates=20, exact_recovery=True,
            parameters=state['parameters'], unknown_rows_drawn=state['unknown_rows_drawn'],
            unknown_rows_with_loss=state['unknown_rows_with_loss']))
    for other in states[1:]:
        api.assert_common_design(states[0], other)
    once(PUBLIC / 'pilot.json', dict(pid=os.getpid(), seconds=time.monotonic() - start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, cells=cells, engineering_pass=True,
        old_controls_exact=True, three_recoveries_exact=True, real_optimizer_updates=160,
        projected_primary_optimizer_seconds=float(np.mean([r['seconds'] for r in cells])) * 100 * 54,
        projection_excludes_packet_preparation_refit_and_readout=True))
    beat(state='real_full_context_pilot_complete', updates=160)


def train(resume=False):
    cfg, values, legacy, store, _ = admission()
    banks = cache_refs()
    if (PUBLIC / 'training_freeze.json').exists():
        raise ValueError('Completed54fit freeze immutable')
    if not json.loads((PUBLIC / 'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real full-context pilot required')
    start, beat, guard = runtime(cfg)
    refs, neutral, refit = [], 0, False
    for context, items in sorted(values[2]['contexts'].items()):
        packets, changed_packets, original, changed = prepared_context(context, items, values, store, banks, beat)
        for seed in cfg['seeds']:
            states = []
            for variant in cfg['variants']:
                receipt = PUBLIC / 'fits' / f'{context}_seed{seed}_{variant}.json'
                path = checkpoint(context, seed, variant)
                if receipt.exists():
                    if not resume:
                        raise ValueError('Completed bank fits require explicit --resume')
                    row = json.loads(receipt.read_text())
                    if (row['registration_sha256'] != sha(REG) or row['variant'] != variant or row['step'] != 2000
                            or sha(ROOT / row['checkpoint']['path']) != row['checkpoint']['sha256']):
                        raise ValueError('Completed bank fit changed')
                    state = api.completion.base.read_checkpoint(path)
                else:
                    data = changed if variant == 'new_neural' else original
                    state = fit(data, cfg, context, seed, variant, path, beat, guard, resume=path.exists() and resume)
                    if variant != 'new_neural':
                        prior = api.completion.base.read_checkpoint(ROOT / legacy[context, seed, api.BANKS[variant], 'history']['checkpoint']['path'])
                        api.assert_scientific_equal(state, prior)
                    replay = PUBLIC / 'first_full_refit.json'
                    if variant == 'new_neural' and not replay.exists():
                        direct_path = checkpoint(context, seed, variant, 'full_refit')
                        direct = fit(changed, cfg, context, seed, variant, direct_path, beat, guard,
                                     resume=direct_path.exists() and resume)
                        api.assert_scientific_equal(state, direct)
                        once(replay, dict(context=context, seed=seed, variant=variant, updates=2000,
                            scientific_state_exact=True, checkpoint_sha256=sha(path), refit_checkpoint_sha256=sha(direct_path)))
                    row = dict(context=context, seed=seed, variant=variant, step=state['step'], parameters=state['parameters'],
                        seconds=state['seconds'], checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                        legacy_scientific_state_exact=variant != 'new_neural', draw_hash=state['draw_hash'],
                        unknown_rows_drawn=state['unknown_rows_drawn'], unknown_rows_with_loss=state['unknown_rows_with_loss'],
                        first_trace=state['trace'][0], final_trace=state['trace'][-1], input_hashes=state['input_hashes'],
                        registration_sha256=sha(REG), result_source='fresh_native_arm64_matched_active_bank_Torch_cost_training',
                        held_scored=False, independent_roles_opened=False)
                    once(receipt, row)
                if variant != 'new_neural':
                    if not row['legacy_scientific_state_exact']:
                        raise ValueError('Original complete control not verified')
                    neutral += 1
                if state['step'] != 2000 or state['parameters'] != 30405 or state['unknown_rows_with_loss']:
                    raise ValueError('Complete original cost budget and known-only supervision required')
                if (PUBLIC / 'first_full_refit.json').exists():
                    refit = json.loads((PUBLIC / 'first_full_refit.json').read_text())['scientific_state_exact']
                if variant == 'new_neural' and context == sorted(values[2]['contexts'])[0] and seed == 17:
                    replay = json.loads((PUBLIC / 'first_full_refit.json').read_text())
                    if (replay['checkpoint_sha256'] != sha(path)
                            or replay['refit_checkpoint_sha256'] != sha(checkpoint(context, seed, variant, 'full_refit'))):
                        raise ValueError('Completed full-refit checkpoints changed')
                if states:
                    api.assert_common_design(states[0], state)
                states.append(state)
                refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                beat(state='bank_endpoint_complete', context=context, seed=seed, variant=variant,
                     fits=len(refs), total=54, legacy_exact=neutral)
        del packets, changed_packets, original, changed
    if len(refs) != 54 or neutral != 36 or not refit:
        raise ValueError('Complete54fits/36exact controls/full new-bank refit required')
    once(PUBLIC / 'training_freeze.json', dict(pid=os.getpid(), seconds=time.monotonic() - start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fits=refs, count=54,
        primary_optimizer_updates=108000, extra_full_refit_updates=2000, pilot_updates=160,
        legacy_exact_endpoints=36, registration_sha256=sha(REG), bank_freeze_sha256=sha(PUBLIC / 'bank_complete.json'),
        first_full_refit_sha256=sha(PUBLIC / 'first_full_refit.json'), held_scored=False,
        independent_roles_opened=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False))
    beat(state='all54bank_cost_heads_complete', primary_optimizer_updates=108000)


def models():
    path = PUBLIC / 'training_freeze.json'
    engine.committed(path)
    frozen = json.loads(path.read_text())
    if (frozen['count'] != 54 or frozen['legacy_exact_endpoints'] != 36
            or frozen['registration_sha256'] != sha(REG) or frozen['bank_freeze_sha256'] != sha(PUBLIC / 'bank_complete.json')
            or frozen['first_full_refit_sha256'] != sha(PUBLIC / 'first_full_refit.json')):
        raise ValueError('Complete committed matched-bank freeze required')
    result = {}
    for ref in frozen['fits']:
        path = ROOT / ref['path']
        engine.committed(path)
        row = json.loads(path.read_text())
        if sha(path) != ref['sha256'] or sha(ROOT / row['checkpoint']['path']) != row['checkpoint']['sha256'] or row['step'] != 2000:
            raise ValueError('Frozen matched-bank endpoint changed')
        key = row['context'], row['seed'], row['variant']
        if key in result:
            raise ValueError('Repeated bank endpoint')
        result[key] = row
    reg = json.loads(REG.read_text())
    if set(result) != {(c, s, v) for c in reg['contexts'] for s in (17, 29, 43) for v in api.VARIANTS}:
        raise ValueError('All54fixed context/seed/variant endpoints required')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'bank', 'pilot', 'train'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register':
            admission(register=True)
        elif args.phase == 'bank':
            build_banks(args.resume)
        elif args.phase == 'pilot':
            pilot(args.resume)
        else:
            train(args.resume)
