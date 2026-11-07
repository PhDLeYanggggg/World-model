"""Lock and score the one-shot DroneCrowd confirmation; no threshold search."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_dronecrowd_calibration as cal
from src.evaluation import m3w_dronecrowd_confirmation as gate
from src.evaluation import m3w_joint_path_audit as pair

run, np, torch, BASE, OLS = cal.run, cal.np, cal.torch, cal.BASE, cal.OLS
PUBLIC = cal.roles.HOME/'dronecrowd_confirmation_readout_v1'
PRIVATE = cal.DATA/'dronecrowd_confirmation_readout_v1'
REG = PUBLIC/'registration.json'
ADMISSION = PUBLIC/'admission.json'
FILES = (Path(__file__), Path(gate.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_dronecrowd_confirmation.py')
MATCH = ('independent', 'scene', 'pair_additive')
ARMS = (BASE, OLS, 'selected', *('matched_'+k for k in MATCH))
PAIRS = dict(selected_vs_cv=(BASE, 'selected', BASE), selected_vs_ols4=(OLS, 'selected', BASE),
             ols4_vs_cv=(BASE, OLS, BASE))
PAIR_CONTRASTS = {f'selected_vs_matched_{k}': ('matched_'+k, 'selected', BASE) for k in MATCH}
PAIRS.update(PAIR_CONTRASTS)


def register():
    manifest, cut, packet = cal.register()
    sources = (cal.REG, cal.roles.DEST, Path(cal.__file__), Path(cal.model.__file__),
               Path(pair.__file__), Path(cal.prior.grouped_readout.__code__.co_filename))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in sources], cut=cut, seeds=cal.SEEDS, arms=ARMS,
        comparisons=PAIRS, risk_reference=BASE, accuracy_comparator=OLS, original_risk_budget=.02,
        threshold_search_on_confirmation=False, confirmation_role_counts=[30, 72, 216],
        bootstrap_draws=3000, bootstrap_seed=73107, one_shot_labels_after_policy_and_forecast_freeze=True)))
    return manifest, cut, packet


def require_committed(paths):
    for path in paths:
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))], cwd=ROOT) != path.read_bytes():
            raise ValueError('Confirm only from committed readout and frozen selection')


def admit():
    analysis, forecast = (cal.PUBLIC/k for k in ('analysis.json', 'forecast_freeze.json'))
    if not analysis.is_file():
        raise ValueError('Calibration has not produced a result')
    for phase, artifact in (('infer', forecast), ('score', analysis), ('replay', analysis)):
        if run.read(cal.PUBLIC/f'{phase}_execution.json')['artifact'] != run.artifact(artifact):
            raise ValueError('Calibration score/replay does not bind the frozen artifacts')
    if run.read(forecast)['registration'] != run.sha(cal.REG):
        raise ValueError('Calibration prediction freeze used a different protocol')
    chosen = gate.selected_gate(run.read(analysis))
    binding = dict(registration=run.sha(REG), calibration=run.artifact(analysis),
        calibration_forecasts=run.artifact(forecast), arm=chosen, original_rules_unchanged=True)
    run.once(ADMISSION, binding)
    return binding


def reader(manifest, sequence, row, provenance):
    gate.require_confirmation(manifest, sequence)
    folder = cal.DATA/'dronecrowd_recordings_v1'/sequence
    if run.sha(folder/'metadata.json') != row['metadata_sha256']:
        raise ValueError('Changed confirmation recording metadata')
    reader = cal.DroneCrowdSceneWindows(folder, cal.PROBES[0],
        expected_xml_sha256=row['source_xml_sha256'], expected_provenance=provenance)
    if len(reader) != row['main_query_frames']:
        raise ValueError('Changed confirmation query schedule')
    return reader


def infer(manifest, info, provenance, packet, admission, beat):
    loaded, clips = cal.networks(packet), []
    for seq in manifest['assignments']['confirmation']['recordings']:
        path = PRIVATE/f'forecast_{seq}.npz'
        mark = path.with_suffix('.json')
        if not mark.exists():
            source = reader(manifest, seq, info[seq], provenance)
            # Uses only the same verified prefix route as calibration.
            data = cal.infer(source, source.query_frames, loaded, beat)
            j = cal.model.ARMS.index(admission['arm'])
            take = [j, *(cal.model.ARMS.index(k) for k in MATCH)]
            for seed in cal.SEEDS:
                for k in ('index', 'paths'):
                    name = f'{k}_{seed}'
                    data[name] = data[name][take]
            cal.prior.h.prune.previous.history.feedback.write_array(path, **data)
            run.once(mark, dict(sequence=seq, admission=run.sha(ADMISSION), queries=len(data['frames']),
                rows=len(data['agents']), artifact=run.artifact(path)))
        cell = run.read(mark)
        if cell['sequence'] != seq or cell['admission'] != run.sha(ADMISSION):
            raise ValueError('Changed confirmation checkpoint')
        run.checked(cell['artifact'])
        clips.append(cell)
        beat(state='confirmation_prefix_frozen', recordings=len(clips))
    expected = manifest['input_only_counts']['confirmation']
    if (len(clips), sum(c['queries'] for c in clips), sum(c['rows'] for c in clips)) != (
            expected['recordings'], expected['queries'], expected['eligible']):
        raise ValueError('Confirmation predictions differ from frozen input roster')
    return dict(admission=run.sha(ADMISSION), clips=clips, future_labels_read=False)


def recording(manifest, seq, cell, row, provenance, cut, group):
    source = reader(manifest, seq, row, provenance)
    with np.load(run.checked(cell['artifact']), allow_pickle=False) as f:
        frozen = {k: f[k] for k in f.files}
    np.testing.assert_array_equal(frozen['frames'], source.query_frames)
    target, valid = cal.label_arrays(source, frozen)
    frames, offset = frozen['frames'], frozen['offsets']
    queries = np.repeat(frames, np.diff(offset))
    histories = []
    for f in frames:
        raw = source.get_inputs(int(f))
        histories.append(raw['positions'][raw['history_complete']])
    rows, moments = [], []
    for seed in cal.SEEDS:
        cv, ols = frozen['cv'], frozen['ols']
        i, p = frozen[f'index_{seed}'], frozen[f'paths_{seed}']
        paths = {BASE: cv, OLS: ols, 'selected': gate.matched_forecast(cv, i[0], p[0], i[0])}
        paths.update({'matched_'+k: gate.matched_forecast(cv, i[j], p[j], i[0])
                      for j, k in enumerate(MATCH, 1)})
        r = run.parent.readout.score_recording(paths, {k: v for k, v in PAIRS.items() if k not in PAIR_CONTRASTS},
                                              target, valid, queries, seq,
                                              easy_cut=cut['easy'], hard_cut=cut['hard'])
        if len({r['metrics'][k]['risk']['selected'] for k in ARMS[2:]}) != 1:
            raise ValueError('Matched control emitted a different count')
        active = (i[0] >= 0).any(1)
        use = np.repeat(active, np.diff(offset))
        weights = np.repeat(1./np.diff(offset)[active], np.diff(offset)[active])
        for name, (a, b, ref) in PAIR_CONTRASTS.items():
            r['comparisons'][name] = (run.parent.readout.comparison_masses(
                paths[a][use], paths[b][use], paths[ref][use], target[use], valid[use], weights)
                if use.any() else dict(reference=0., known_delta=0., unknown_radius=0.))
        rows.append(dict(r, source=group, seed=seed))
        x, y = np.zeros((len(MATCH), len(pair.PAIR_FIELDS))), np.zeros((len(MATCH), len(pair.PAIR_FIELDS)), int)
        for take, h, lo, hi in zip(active, histories, offset[:-1], offset[1:]):
            if not take:
                continue
            ids = slice(lo, hi)
            keys = (*('matched_'+k for k in MATCH), 'selected')
            _, v, _, _ = pair.query(h, np.stack([paths[k][ids] for k in keys]), target[ids], valid[ids],
                                     [(j, len(MATCH)) for j in range(len(MATCH))])
            good = np.isfinite(v)
            x += np.where(good, v, 0.)
            y += good
        moments.append(dict(seed=seed, sums=x.tolist(), counts=y.tolist()))
    return dict(rows=rows, pair_moments=moments)


def aggregate(manifest, documents):
    group_map = run.read(cal.roles.GROUPS)['recording_group']
    roster = {seq: group_map[seq] for seq in manifest['assignments']['confirmation']['recordings']}
    if set(documents) != set(roster):
        raise ValueError('Incomplete confirmation record scores')
    grouped = cal.prior.grouped_readout([r for v in documents.values() for r in v['rows']], roster, cal.SEEDS)
    groups = sorted(set(roster.values()))
    tensors = []
    for g in groups:
        per_seed = []
        for s in cal.SEEDS:
            cells = [r for rec, d in documents.items() if roster[rec] == g
                     for r in d['pair_moments'] if r['seed'] == s]
            x = np.sum([r['sums'] for r in cells], axis=0)
            n = np.sum([r['counts'] for r in cells], axis=0)
            per_seed.append(np.divide(x, n, out=np.full_like(x, np.nan), where=n > 0))
        tensors.append(per_seed)
    v = np.asarray(tensors)
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    relative = {key: {field: pair.bootstrap(v[:, :, i, j], draw)
                      for j, field in enumerate(pair.PAIR_FIELDS)} for i, key in enumerate(PAIR_CONTRASTS)}
    breakdown = {}
    for g in groups:
        rows = [r for rec, d in documents.items() if roster[rec] == g for r in d['rows']]
        arms = {}
        for k in ARMS:
            metrics = [r['metrics'][k] for r in rows]
            known = sum(m['known_rows'] for m in metrics)
            endpoint = sum(m['endpoint_rows'] for m in metrics)
            ade = sum((m['known_ADE_pixel'] or 0.)*m['known_rows'] for m in metrics)
            fde = sum((m['available_endpoint_FDE_diagnostic_pixel'] or 0.)*m['endpoint_rows'] for m in metrics)
            arms[k] = dict(passed_views=sum(m['risk']['supported'] for m in metrics),
                undefined_easy_views=sum(m['risk']['completion_easy_risk_upper'] is None for m in metrics),
                known_rows_seed_repeated=known, endpoint_rows_seed_repeated=endpoint,
                available_label_ADE_pixel=ade/known if known else None,
                available_endpoint_FDE_pixel=fde/endpoint if endpoint else None)
        breakdown[g] = dict(recordings=len(rows)//len(cal.SEEDS), arms=arms)
    return dict(summary=grouped, relative_geometry=relative, per_exclusion_group=breakdown,
        claims=gate.verdict(grouped, relative), independent_confirmation=True, deployment_changed=False,
        threshold_selected_on_confirmation=False, confirmation_labels_opened=True, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'admit', 'infer', 'score', 'replay'))
    phase = p.parse_args().phase
    manifest, cut, packet = register()
    if phase == 'register':
        return
    require_committed((*FILES, REG))
    admission = admit()
    if phase == 'admit':
        return
    require_committed((ADMISSION,))
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            row = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-started, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            print(row, flush=True)

        allowed = manifest['assignments']['confirmation']['recordings']
        inventory = run.read(cal.CACHE)
        info = {r['sequence_id']: r for r in inventory['recordings'] if r['sequence_id'] in allowed}
        provenance = inventory['provenance']
        freeze = PUBLIC/'forecast_freeze.json'
        analysis = PUBLIC/'analysis.json'
        if phase == 'infer':
            run.once(freeze, run.stable(infer(manifest, info, provenance, packet, admission, beat)))
            out = freeze
        else:
            frozen = run.read(freeze)
            if (frozen['admission'] != run.sha(ADMISSION) or [c['sequence'] for c in frozen['clips']] != allowed
                    or run.read(PUBLIC/'infer_execution.json')['artifact'] != run.artifact(freeze)):
                raise ValueError('Require complete sealed confirmation predictions')
            if phase == 'score':
                if analysis.exists():
                    raise ValueError('Already scored; replay aggregate only')
                run.once(PUBLIC/'opening.json', dict(admission=run.sha(ADMISSION), freeze=run.sha(freeze)))
            elif (not analysis.exists() or run.read(PUBLIC/'score_execution.json')['artifact'] != run.artifact(analysis)):
                raise ValueError('Require the original one-shot score before replay')
            mapping = run.read(cal.roles.GROUPS)['recording_group']
            documents = {}
            for cell in frozen['clips']:
                seq = cell['sequence']
                path = PRIVATE/f'score_{seq}.json'
                if not path.exists():
                    if phase == 'replay':
                        raise ValueError('Replay must never reopen labels')
                    result = recording(manifest, seq, cell, info[seq], provenance, cut, mapping[seq])
                    run.once(path, run.stable(dict(forecast=cell['artifact'], admission=run.sha(ADMISSION), result=result)))
                saved = run.read(path)
                if saved['forecast'] != cell['artifact'] or saved['admission'] != run.sha(ADMISSION):
                    raise ValueError('Stale recording score')
                documents[seq] = saved['result']
                beat(state='recording_statistics_checked', recordings=len(documents))
            run.once(analysis, run.stable(aggregate(manifest, documents)))
            out = analysis
        dest = PUBLIC/f'{phase}_execution.json'
        receipt = dict(pid=os.getpid(), seconds=time.monotonic()-started, artifact=run.artifact(out))
        if not dest.exists():
            run.once(dest, receipt)
        elif run.read(dest)['artifact'] != receipt['artifact']:
            raise ValueError('Previous run refers to a different artifact')


if __name__ == '__main__':
    main()
