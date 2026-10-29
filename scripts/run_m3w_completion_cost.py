"""Paired real Torch completion-cost fitting with full recovery and frozen endpoints."""
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
    raise RuntimeError('Use native arm64 .venv-pytorch before importing Torch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_static_scene_cost as parent
from src.world_model import m3w_completion_cost as api

NAME = 'european_completion_cost_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_completion_cost_v1.json'; REG = PUBLIC/'registration.json'
PREVIOUS = ROOT/'outputs/publication_readiness_2026_09/european_forecast_amplitude_v1'
sha, once, json_write = parent.sha, parent.once, parent.json_write
FILES = (CONFIG, Path(__file__), ROOT/'src/world_model/m3w_completion_cost.py',
    ROOT/'tests/test_m3w_completion_cost.py', PUBLIC/'protocol.md', PUBLIC/'prior_controls.md')


def load():
    cfg = json.loads(CONFIG.read_text()); values = parent.admission()
    prior = json.loads((PREVIOUS/'gates.json').read_text())
    check = json.loads((PREVIOUS/'verification.json').read_text())
    if (cfg['training'] != values[0]['training'] or cfg['seeds'] != [17, 29, 43]
            or cfg['modes'] != list(api.MODES) or cfg['banks'] != list(api.BANKS)
            or cfg['risk_budget'] != .02 or cfg['held_scored'] or cfg['independent_roles_opened']
            or not check['exact_replay'] or check['summary_sha256'] != sha(PREVIOUS/'summary.json')
            or prior['at_least_one_fitting_full_risk_and_completion_gain']):
        raise ValueError('Verified prior failure, fixed paired training and unchanged roles required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)): sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG),
        parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        prior_amplitude_verification_sha256=sha(PREVIOUS/'verification.json'),
        prior_amplitude_summary_sha256=sha(PREVIOUS/'summary.json'),
        contexts=sorted(values[2]['contexts']), seeds=cfg['seeds'], modes=cfg['modes'], banks=cfg['banks'],
        fits=72, steps_per_fit=2000, primary_optimizer_updates=144000,
        parameters=26309, causal_fields=398, forecaster_seed=43, scope='original_fitting_only',
        sampling='common_source_balanced_all_past_eligible_rows_known_and_unknown',
        normalization='unchanged_physical_fitting_only',
        target_intervention='known_only_loss_mask_vs_explicit_finite_completion_endpoints',
        unknown_outcomes_imputed=False, future_support_inference_input=False,
        prior_cached_control_is_matched_to_new_sampler=False, risk_budget=.02,
        held_scored=False, independent_roles_opened=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False))
    return cfg, values


def admission():
    cfg, values = load()
    for rel, digest in json.loads(REG.read_text())['bindings'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError('Registered completion experiment changed: '+rel)
        parent.parent.committed(ROOT/rel)
    parent.parent.committed(REG)
    return cfg, values


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def runtime(cfg):
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume existing atomic checkpoints')))
    def guard():
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']:
            raise OSError('Preserve 10 GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['checkpoint_cap_bytes']:
            raise OSError('Registered private checkpoint cap exceeded')
        if time.monotonic()-started > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('12-hour boundary; resume the same full scientific roster')
    def beat(**event):
        guard(); value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event)
        json_write(PRIVATE/'heartbeat.json', value)
        with (PRIVATE/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    guard(); beat(state='runtime_ready', architecture=platform.machine(), python=sys.executable,
                  torch_threads=torch.get_num_threads(), workers=0, device='cpu', resource_probing=False)
    return started, beat, guard


def packets(context, items, values, beat):
    _, _, _, source, jobs, identity, _, packed, polygons = values
    scopes = parent.scene_packets(parent.parent.prepare_context(context, items, source, jobs, identity, beat,
        include_held=False), source, packed, polygons)
    if set(scopes) != {'fit'}:
        raise ValueError('Fitting-only data required')
    return scopes['fit']


def checkpoint(context, seed, bank, mode, folder='heads'):
    return PRIVATE/folder/f'{context}_seed{seed}_{bank}_{mode}.pt.gz'


def cells(cfg):
    return [(bank, mode) for bank in cfg['banks'] for mode in cfg['modes']]


def known_legacy_neutral(data, state, settings, context, beat, guard, resume):
    legacy = dict(data, allowed=data['allowed'] & data['known'][:, None])
    legacy['input_hashes'] = dict(data['input_hashes'],
        engineering_known_masked_allowed=parent.fingerprint(legacy['allowed'].numpy()))
    path = checkpoint(context, 17, 'physical_neural', 'legacy_masked_common_stream', 'pilot')
    with parent.api.installed():
        old = parent.parent.api.fit(legacy, arm='physical_neural', settings=settings, seed=17,
            identity=dict(engineering_neutral=True, common_sampler_not_original_endpoint=True),
            path=path, stop_at=20, resume=path.exists() and resume, heartbeat=beat, guard=guard)
    for key in ('model', 'initial_model', 'optimizer', 'preprocess', 'draw_hash', 'row_draws',
                'sampler_rng', 'torch_rng', 'parameters'):
        api.base.exact(state[key], old[key])
    for left, right in zip(state['trace'], old['trace']):
        for key in right:
            api.base.exact(left[key], right[key])
    return dict(model_Adam_draws_RNG_exact=True,
        common_sampler_not_original_earlier_endpoint=True,
        legacy_hardcoded_unknown_counter_not_used=True, measured_unknown_rows_drawn=state['unknown_rows_drawn'])


def pilot(resume=False):
    cfg, values = admission(); finish = PUBLIC/'pilot.json'
    if finish.exists():
        raise ValueError('Completed pilot immutable')
    started, beat, guard = runtime(cfg)
    context, items = sorted(values[2]['contexts'].items())[0]
    data = api.training_data(packets(context, items, values, beat), parent)
    settings = dict(cfg['training'], heartbeat_every=10, checkpoint_every=10)
    states, records = {}, []
    for bank, mode in cells(cfg):
        path = checkpoint(context, 17, bank, mode, 'pilot')
        kw = dict(data=data, mode=mode, bank=bank, settings=settings, seed=17,
            identity=dict(context=context, registration_sha256=sha(REG), engineering_pilot=True),
            heartbeat=beat, guard=guard)
        if not path.exists():
            api.fit(**kw, path=path, stop_at=cfg['pilot_resume_at'])
        state = api.fit(**kw, path=path, stop_at=cfg['pilot_steps'], resume=True)
        direct_path = checkpoint(context, 17, bank, mode, 'pilot_direct')
        direct = api.fit(**kw, path=direct_path, stop_at=cfg['pilot_steps'], resume=direct_path.exists() and resume)
        api.exact_training(state, direct)
        if not any(not torch.equal(state['initial_model'][k], v) for k, v in state['model'].items()):
            raise ValueError('No real parameter update')
        if state['unknown_rows_drawn'] == 0 or (mode == 'completion_bounds' and state['unknown_rows_with_loss'] == 0):
            raise ValueError('Pilot did not exercise actual unknown supervision')
        states[bank, mode] = state
        records.append(dict(bank=bank, mode=mode, steps=state['step'], seconds=state['seconds'],
            parameters=state['parameters'], unknown_rows_drawn=state['unknown_rows_drawn'],
            unknown_rows_with_loss=state['unknown_rows_with_loss'], checkpoint_bytes=path.stat().st_size,
            first_trace=state['trace'][0], final_trace=state['trace'][-1], exact_resume=True))
    for state in list(states.values())[1:]:
        api.exact_paired_design(next(iter(states.values())), state)
    neutral = known_legacy_neutral(data, states['physical_neural', 'known_only'], settings, context, beat, guard, resume)
    once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started,
        architecture=platform.machine(), real_torch=True, device='cpu', cpu_threads=4, num_workers=0,
        engineering_pass=True, full_first_context=True, cells=records, legacy_neutral=neutral,
        actual_engineering_optimizer_updates=180, draw_pool=data['draw_pool'],
        projected_primary_optimizer_seconds=sum(r['seconds'] for r in records)*100*18,
        projected_primary_checkpoint_bytes=sum(r['checkpoint_bytes'] for r in records)*18,
        peak_RSS_bytes=rss_bytes(), held_scored=False, independent_roles_opened=False))
    beat(state='pilot_complete', cells=4, actual_optimizer_updates=180)


def train(resume=False):
    cfg, values = admission(); finish = PUBLIC/'training_freeze.json'
    if finish.exists():
        raise ValueError('Completed 72-fit training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real pilot and exact recovery required')
    started, beat, guard = runtime(cfg); refs = []
    first_context = sorted(values[2]['contexts'])[0]
    for context, items in sorted(values[2]['contexts'].items()):
        data = api.training_data(packets(context, items, values, beat), parent)
        for seed in cfg['seeds']:
            states = []
            for bank, mode in cells(cfg):
                path = checkpoint(context, seed, bank, mode)
                kw = dict(data=data, mode=mode, bank=bank, settings=cfg['training'], seed=seed,
                    identity=dict(context=context, registration_sha256=sha(REG)), heartbeat=beat, guard=guard)
                state = api.fit(**kw, path=path, resume=path.exists() and resume)
                if state['step'] != 2000 or state['parameters'] != 26309:
                    raise ValueError('Unchanged full model and training budget required')
                if states:
                    api.exact_paired_design(states[0], state)
                replay = PUBLIC/'first_full_replay.json'
                if (context, seed, bank, mode) == (first_context, 17, 'physical_only', 'known_only'):
                    if not replay.exists():
                        direct_path = checkpoint(context, seed, bank, mode, 'full_refit')
                        direct = api.fit(**kw, path=direct_path, resume=direct_path.exists() and resume)
                        api.exact_training(state, direct)
                        once(replay, dict(context=context, bank=bank, mode=mode, seed=seed, updates=2000,
                            scientific_state_exact=True, checkpoint_bytes_identical_claim=False,
                            primary_seconds=state['seconds'], direct_seconds=direct['seconds'],
                            registration_sha256=sha(REG)))
                    elif not json.loads(replay.read_text())['scientific_state_exact']:
                        raise ValueError('First full refit not verified')
                receipt = PUBLIC/'fits'/f'{context}_seed{seed}_{bank}_{mode}.json'
                value = dict(context=context, seed=seed, bank=bank, mode=mode, step=state['step'],
                    checkpoint=dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size),
                    parameters=state['parameters'], seconds=state['seconds'], draw_hash=state['draw_hash'],
                    row_draws=state['row_draws'], unknown_rows_drawn=state['unknown_rows_drawn'],
                    unknown_rows_with_loss=state['unknown_rows_with_loss'], known_rows_with_loss=state['known_rows_with_loss'],
                    draw_pool=state['draw_pool'], first_trace=state['trace'][0], final_trace=state['trace'][-1],
                    input_hashes=state['input_hashes'], target_columns=state['target_columns'],
                    paired_design_exact=True, unknown_outcomes_imputed=False,
                    registration_sha256=sha(REG), result_source='fresh_native_arm64_torch_cost_head_training')
                once(receipt, value); refs.append(dict(path=str(receipt.relative_to(ROOT)), sha256=sha(receipt)))
                states.append(state)
                beat(state='fit_complete', fits=len(refs), total=72, context=context, seed=seed, bank=bank, mode=mode,
                     unknown_rows_drawn=state['unknown_rows_drawn'], unknown_rows_with_loss=state['unknown_rows_with_loss'])
        del data
    if len(refs) != 72:
        raise ValueError('Every registered primary fit required')
    once(finish, dict(pid=os.getpid(), seconds=time.monotonic()-started, fits=refs, count=72,
        optimizer_updates=144000, extra_full_refit_updates=2000, pilot_updates=180,
        registration_sha256=sha(REG), first_full_replay_sha256=sha(PUBLIC/'first_full_replay.json'),
        peak_RSS_bytes=rss_bytes(), cpu_threads=4, num_workers=0, device='cpu',
        architecture=platform.machine(), held_scored=False, independent_roles_opened=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False))
    beat(state='all72training_complete', optimizer_updates=144000, held_scored=False)


def models():
    parent.parent.committed(PUBLIC/'training_freeze.json')
    frozen = json.loads((PUBLIC/'training_freeze.json').read_text())
    if frozen['count'] != 72 or frozen['held_scored'] or frozen['registration_sha256'] != sha(REG):
        raise ValueError('Committed full 72-fit freeze required')
    if (sha(PUBLIC/'first_full_replay.json') != frozen['first_full_replay_sha256']
            or not json.loads((PUBLIC/'first_full_replay.json').read_text())['scientific_state_exact']):
        raise ValueError('First full exact-refit evidence changed')
    result = {}
    for ref in frozen['fits']:
        path = ROOT/ref['path']; row = json.loads(path.read_text())
        parent.parent.committed(path)
        if sha(path) != ref['sha256'] or sha(ROOT/row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Frozen completion endpoint changed')
        key = row['context'], row['seed'], row['bank'], row['mode']
        if key in result or row['step'] != 2000 or row['unknown_rows_drawn'] == 0:
            raise ValueError('Unique complete endpoint with real unknown-row exposure required')
        if (row['mode'] == 'known_only' and row['unknown_rows_with_loss'] != 0
                or row['mode'] == 'completion_bounds' and row['unknown_rows_with_loss'] == 0):
            raise ValueError('Supervision-mode accounting differs')
        result[key] = row
    reg = json.loads(REG.read_text())
    expected = {(context, seed, bank, mode) for context in reg['contexts'] for seed in reg['seeds']
                for bank in reg['banks'] for mode in reg['modes']}
    if set(result) != expected:
        raise ValueError('All 72 registered context/seed/bank/mode endpoints required')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'train'))
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'register':
            load()
        elif args.phase == 'pilot':
            pilot(args.resume)
        else:
            train(args.resume)
