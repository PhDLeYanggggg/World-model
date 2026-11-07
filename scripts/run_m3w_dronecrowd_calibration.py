"""Freeze 40 calibration recordings' forecasts before evaluating their labels."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import freeze_m3w_dronecrowd_transport_candidate as prior
from scripts import run_m3w_native_inference_rehearsal as native
from scripts import freeze_m3w_dronecrowd_calibration_roles as roles
from scripts.run_m3w_dronecrowd_blind_preflight import CACHE, DATA, PROBES, DroneCrowdSceneWindows
from src.world_model import m3w_dronecrowd_calibration as model

run, np, torch = prior.run, prior.np, native.torch
BASE, OLS = prior.BASE, prior.OLS
PUBLIC = roles.HOME/'dronecrowd_calibration_readout_v1'
PRIVATE = DATA/'dronecrowd_calibration_readout_v1'
REG = PUBLIC/'registration.json'
SEEDS = tuple(run.kin.prior.SEEDS)
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_dronecrowd_calibration.py')
PAIRS = {f'{a}_vs_cv': (BASE, a, BASE) for a in (OLS, *model.ARMS)}
PAIRS.update({f'{a}_vs_ols4': (OLS, a, BASE) for a in model.ARMS})
PAIRS.update({f'joint_vs_{a}': (a, 'joint', BASE) for a in prior.model.MATCHED[:-1]})


def register():
    manifest = run.read(roles.DEST)
    if manifest != run.stable(roles.build()):
        raise ValueError('Require the approved immutable calibration overlay')
    cut = prior.register()
    previous = prior.PRIVATE/'freeze.json'
    if run.read(prior.PUBLIC/'freeze_execution.json')['artifact'] != run.artifact(previous):
        raise ValueError('Require sealed development controller')
    ckpt = prior.lineage.verify_bank()
    parents = (roles.DEST, CACHE, prior.REG, prior.PUBLIC/'freeze_execution.json',
               native.PUBLIC/'analysis.json', Path(run.model.__file__),
               Path(prior.model.__file__), Path(native.prepare_prefix.__code__.co_filename),
               Path(run.parent.readout.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in parents], exact_predictors=ckpt,
        cut=cut, seeds=SEEDS, arm_order=(BASE, OLS, *model.ARMS), contrasts=PAIRS,
        normalized_gain_thresholds=model.MARGINS, original_budget=.02,
        batch_size=256, cpu_threads=4, workers=0, scope='calibration_only',
        confirmation_access=False, all_predictions_frozen_before_any_calibration_labels=True,
        threshold_rule='all_120_pass_and_positive_paired_CV_lower_interval_then_maximize_lower_point')))
    return manifest, cut, run.read(previous)


def networks(packet):
    out = {}
    kinematic = {r['seed']: r for r in run.read(native.KIN/'training_freeze.json')['fits'] if r['arm'] == 'self'}
    arch = {(r['family'], r['seed']): r for r in run.read(native.ARCHIVE)['trials'] if r['fold'] == 2}
    for cell in packet['cells']:
        seed = cell['seed']
        nets = {}
        for family in ('pct', 'eqmotion', 'self'):
            row = kinematic[seed] if family == 'self' else arch[family, seed]
            saved = torch.load(run.checked(row['checkpoint']), map_location='cpu', weights_only=False)
            ident = saved['identity']
            if family == 'self':
                if saved['step'] != 2400 or ident != row['identity']:
                    raise ValueError('Wrong self-kinematic checkpoint')
                net = native.KinematicForecaster(False)
            else:
                if (saved['step'], ident['fold'], ident['seed'], ident['family'], ident['baseline_index']) != (
                        4000, 2, seed, family, 4):
                    raise ValueError('Wrong comparator checkpoint')
                net = native.SourceForecaster(ident['architecture'], ident['baseline_index'])
            net.load_state_dict(saved['model'])
            nets[family] = net.eval()
        with np.load(run.checked(cell['cache']), allow_pickle=False) as f:
            out[seed] = (nets, f['transport_coefficients'])
    if tuple(sorted(out)) != SEEDS:
        raise ValueError('Expected the original three seeds')
    return out


def reader(manifest, sequence, inventory, provenance):
    model.require_calibration(manifest, sequence)
    folder = DATA/'dronecrowd_recordings_v1'/sequence
    row = inventory[sequence]
    if run.sha(folder/'metadata.json') != row['metadata_sha256']:
        raise ValueError('Calibration recording metadata changed')
    out = DroneCrowdSceneWindows(folder, PROBES[0], expected_xml_sha256=row['source_xml_sha256'],
        expected_provenance=provenance)
    if len(out) != row['main_query_frames']:
        raise ValueError('Query roster changed')
    return out


def prefixes(source, frames):
    arrays = dict(geometry=[], history=[], agents=[], origin=[], cv=[], ols=[])
    baseline = {k: [] for k in run.NAMES if k not in ('pct', 'eqmotion', 'self')}
    counts = []
    for frame in frames:
        raw = source.get_inputs(int(frame))
        z = native.prepare_prefix(raw)
        h = raw['positions'][raw['history_complete']]
        if len(h) == 0 or not np.array_equal(h[:, -1], z.origin):
            raise ValueError('Past-only target roster changed')
        counts.append(len(h))
        for k, v in zip(arrays, (z.geometry, h, z.agent_id, z.origin, z.baselines[BASE], z.baselines[OLS])):
            arrays[k].append(v)
        for k in baseline:
            baseline[k].append(z.baselines[k])
    return ({k: np.concatenate(v) for k, v in arrays.items()},
            {k: np.concatenate(v) for k, v in baseline.items()}, np.r_[0, np.cumsum(counts)])


def infer(source, frames, loaded, beat):
    arrays, baselines, offsets = prefixes(source, frames)
    common = {k: arrays[k] for k in ('agents', 'origin', 'cv', 'ols')}
    common.update(frames=frames, offsets=offsets)
    for seed in SEEDS:
        nets, coef = loaded[seed]
        paths = {k: native.predict_relative(net, arrays['geometry'], batch_size=256) for k, net in nets.items()}
        paths.update(baselines)
        slots = np.full((len(model.ARMS), len(frames), 2), -1, dtype=np.int64)
        replacements = np.zeros((*slots.shape, 12, 2), dtype=float)
        for q, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            take = slice(lo, hi)
            bank = np.stack([paths[k][take] for k in run.NAMES], 1)
            choices = model.decide(arrays['history'][take], bank, arrays['cv'][take], arrays['ols'][take], coef)
            for j, action in enumerate(choices):
                ix = np.flatnonzero(action >= 0)
                slots[j, q, :len(ix)] = lo+ix
                replacements[j, q, :len(ix)] = bank[ix, action[ix]]
        common[f'index_{seed}'], common[f'paths_{seed}'] = slots, replacements
        beat(state='recording_seed_inferred', seed=seed, rows=len(arrays['agents']))
    return common


def label_arrays(source, frozen):
    target, valid = [], []
    for k, frame in enumerate(frozen['frames']):
        raw = source.get_inputs(int(frame))
        lo, hi = frozen['offsets'][k:k+2]
        z = raw['history_complete']
        np.testing.assert_array_equal(raw['agent_ids'][z], frozen['agents'][lo:hi])
        np.testing.assert_array_equal(raw['positions'][z, -1], frozen['origin'][lo:hi])
        out = source.get_labels(raw)
        target.append(out['future_positions'][z]-frozen['origin'][lo:hi, None])
        valid.append(out['future_mask'][z])
    return np.concatenate(target), np.concatenate(valid)


def score(manifest, cut, clips, inventory, provenance, beat):
    groups = {r: g for r, g in run.read(roles.GROUPS)['recording_group'].items()
              if manifest['recording_role_index'][r] == 'calibration'}
    rows = []
    for cell in clips:
        sequence = cell['sequence']
        source = reader(manifest, sequence, inventory, provenance)
        with np.load(run.checked(cell['artifact']), allow_pickle=False) as f:
            frozen = {k: f[k] for k in f.files}
        np.testing.assert_array_equal(frozen['frames'], source.query_frames)
        target, valid = label_arrays(source, frozen)
        queries = np.repeat(frozen['frames'], np.diff(frozen['offsets']))
        for seed in SEEDS:
            paths = {BASE: frozen['cv'], OLS: frozen['ols']}
            for j, arm in enumerate(model.ARMS):
                i, p = frozen[f'index_{seed}'][j], frozen[f'paths_{seed}'][j]
                paths[arm] = frozen['cv'].copy()
                paths[arm][i[i >= 0]] = p[i >= 0]
            r = run.parent.readout.score_recording(paths, PAIRS, target, valid, queries, sequence,
                                                   easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(r, source=groups[sequence], seed=seed))
        beat(state='calibration_recording_scored', recording=sequence, rows=len(target))
    result = prior.grouped_readout(rows, groups, SEEDS)
    winner = model.winner(result)
    return dict(summary=result, selected_arm=winner, selected_normalized_gain_margin=(
        model.MARGINS[model.GATES.index(winner)] if winner else None),
        calibration_pass=winner is not None, confirmation_outcome_access=False,
        calibration_outcomes_opened=True, independent_confirmation=False,
        original_rules_unchanged=True, submission_ready=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'smoke', 'infer', 'score', 'replay'))
    phase = p.parse_args().phase
    manifest, cut, packet = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit protocol and code before opening calibration inputs')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            print(row, flush=True)

        inventory = run.read(CACHE)
        allowed = manifest['assignments']['calibration']['recordings']
        meta = {r['sequence_id']: r for r in inventory['recordings'] if r['sequence_id'] in allowed}
        if set(meta) != set(allowed) or len(meta) != 40:
            raise ValueError('Require exactly the approved calibration recordings')
        frozen = PUBLIC/'forecast_freeze.json'
        if phase in ('smoke', 'infer'):
            loaded = networks(packet)
            if phase == 'smoke':
                smallest = min(meta, key=lambda k: (meta[k]['agents'], k))
                largest = max(meta, key=lambda k: (meta[k]['agents'], k))
                timings = []
                for seq in (smallest, largest):
                    source = reader(manifest, seq, meta, inventory['provenance'])
                    t = time.monotonic()
                    data = infer(source, source.query_frames[:8], loaded, beat)
                    timings.append(dict(queries=len(data['frames']), rows=len(data['agents']),
                        seconds=time.monotonic()-t))
                dest = PUBLIC/'smoke_execution.json'
                if not dest.exists():
                    run.once(dest, dict(registration=run.sha(REG), measurements=timings,
                        native_arm64=platform.machine() == 'arm64', torch_threads=torch.get_num_threads(),
                        future_labels_read=False, confirmation_access=False))
                return
            if run.read(PUBLIC/'smoke_execution.json')['registration'] != run.sha(REG):
                raise ValueError('Require input-only smoke for current version')
            clips = []
            for seq in allowed:
                path = PRIVATE/f'{seq}.npz'
                mark = path.with_suffix('.json')
                if not mark.exists():
                    source = reader(manifest, seq, meta, inventory['provenance'])
                    arrays = infer(source, source.query_frames, loaded, beat)
                    prior.h.prune.previous.history.feedback.write_array(path, **arrays)
                    run.once(mark, dict(sequence=seq, registration=run.sha(REG),
                        queries=len(arrays['frames']), rows=len(arrays['agents']), artifact=run.artifact(path)))
                cell = run.read(mark)
                if cell['sequence'] != seq or cell['registration'] != run.sha(REG):
                    raise ValueError('Changed calibration recording on resume')
                run.checked(cell['artifact'])
                clips.append(cell)
                beat(state='calibration_recording_frozen', completed=len(clips))
            counts = manifest['input_only_counts']['calibration']
            if (sum(c['rows'] for c in clips), sum(c['queries'] for c in clips)) != (counts['eligible'], counts['queries']):
                raise ValueError('Inferred calibration roster differs from input preflight')
            run.once(frozen, dict(clips=clips, registration=run.sha(REG), future_labels_read=False))
            out = frozen
        else:
            doc = run.read(frozen)
            if doc['registration'] != run.sha(REG) or run.artifact(frozen) != run.read(PUBLIC/'infer_execution.json')['artifact']:
                raise ValueError('Require every calibration prediction frozen before labels')
            if [c['sequence'] for c in doc['clips']] != allowed:
                raise ValueError('Frozen recording list differs from approved calibration')
            out = PUBLIC/'analysis.json'
            if phase == 'replay' and run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(out):
                raise ValueError('Require verified result before replay')
            run.once(out, run.stable(score(manifest, cut, doc['clips'], meta, inventory['provenance'], beat)))
        receipt = PUBLIC/f'{phase}_execution.json'
        value = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out))
        if not receipt.exists():
            run.once(receipt, value)
        elif run.read(receipt)['artifact'] != value['artifact']:
            raise ValueError('Previous phase binds another artifact')


if __name__ == '__main__':
    main()
