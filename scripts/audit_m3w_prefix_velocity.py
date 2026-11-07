"""Freeze and score prefix-adaptive forecast directions on exposed development."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_m3w_recording_blend_feasibility as old
from src.world_model import m3w_prefix_velocity as model

run, prev, np, pair = old.run, old.prev, old.np, old.pair
BASE, OLS, GAIN = old.BASE, old.OLS, old.GAIN
FIXED, QUERY, AGENT = model.ARMS
OLS8 = 'history_ols8_velocity'
PUBLIC = old.PUBLIC.parent/'native_prefix_velocity_v1'
PRIVATE = old.PRIVATE.parent/PUBLIC.name
REG = PUBLIC/'registration.json'
FILES = (Path(__file__).resolve(), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_prefix_velocity.py')
ALL = (BASE, OLS, OLS8, FIXED, QUERY, AGENT)
MATCH = (BASE, OLS, GAIN, FIXED, QUERY, AGENT)
CONTRASTS = [(FIXED, AGENT), (QUERY, AGENT), (FIXED, QUERY), (OLS, AGENT), (BASE, AGENT)]


def pairs(names):
    contrasts = CONTRASTS+([(OLS8, AGENT), (OLS, FIXED)] if OLS8 in names else [(GAIN, AGENT), (GAIN, FIXED)])
    return {f'{b}_vs_{a}': (a, b, BASE) for a, b in contrasts}


def register():
    cut, _ = old.register()
    prev.receipt(old.PUBLIC, 'replay')
    bindings = (*FILES, old.REG, old.PUBLIC/'replay_execution.json', Path(pair.__file__),
                Path(run.parent.readout.__file__), prev.PUBLIC/'freeze_execution.json')
    run.once(REG, run.stable(dict(bindings=[run.artifact(f) for f in bindings], cut=cut,
        horizon_steps=12, history_steps=8, seeds=prev.SEEDS, windows=model.WINDOWS, origins=model.ORIGINS,
        prior=model.PRIOR, temperature=model.TEMPERATURE, original_budget=.02,
        primary='matched_agent_prefix_vs_fixed_prefix', secondaries=['whole_roster', 'query_uniform', 'CV', 'OLS4', 'OLS8'],
        no_new_training=True, external_access=False)))
    return cut


def freeze(d):
    paths = model.candidates(d['history'], d['queries'])
    digest = run.kin.prior.source.array_hash
    run.once(PRIVATE/'freeze.json', run.stable(dict(registration=run.sha(REG),
        row_ids=digest(d['global_ids']), paths={k: digest(v) for k, v in paths.items()},
        rows=len(d['global_ids']), past_only=True)))
    return paths


def score(d, proposal, cut, beat):
    refs = run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    target, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    records = prev.old.capacity.cal.prior.h.previous.prior.blocks(d, lab['frames'])
    roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
    groups = sorted(set(roster.values()))
    rows = {'matched': [], 'whole_roster': []}
    geometry, switched = {}, 0
    substituted = dict.fromkeys(model.ARMS, 0)
    comp = pairs(MATCH)
    indices = [(MATCH.index(a), MATCH.index(b)) for a, b, _ in comp.values()]
    for c in prev.receipt(prev.PUBLIC, 'freeze')['cells']:
        seed = c['seed']
        cv, bank = run.candidates(d, refs, seed)
        with np.load(run.checked(c['artifact']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            choice = z['choice'][0]
        cells = {g: [] for g in groups}
        for rec, (ix, queries) in records.items():
            floor = cv[ix]
            ols, ols8 = (bank[ix, run.NAMES.index(k)] for k in (OLS, OLS8))
            gain = run.model.selected_paths(choice[ix], bank[ix], floor)
            active = (gain != floor).any((1, 2))
            switched += int(active.sum())
            local = {k: p[ix] for k, p in proposal.items()}
            # A new exact-CV path does not erase an old intervention.
            matched = {k: np.where((active & (p != floor).any((1, 2)))[:, None, None], p, gain)
                       for k, p in local.items()}
            for k, p in matched.items():
                np.testing.assert_array_equal((p != floor).any((1, 2)), active)
                substituted[k] += int((p != gain).any((1, 2)).sum())
            paths = {BASE: floor, OLS: ols, GAIN: gain, **matched}
            y, mask, h, q = target[ix], valid[ix], d['history'][ix], d['queries'][ix]
            for arm, options in (('matched', paths), ('whole_roster', {BASE: floor, OLS: ols, OLS8: ols8, **local})):
                out = run.parent.readout.score_recording(options, pairs(tuple(options)), y, mask, q, rec,
                                                         easy_cut=cut['easy'], hard_cut=cut['hard'])
                rows[arm].append(dict(out, seed=seed, source=roster[rec]))
            for j in queries:
                cells[roster[rec]].append(pair.query(h[j], np.stack([p[j] for p in paths.values()]),
                                                    y[j], mask[j], indices)[1])
        for g in groups:
            geometry[g, seed] = pair.mean_finite(cells[g])
        beat(state='prefix_readout', seed=seed, matched_switches=switched)
    out = {k: prev.old.prior.grouped_readout(v, roster, prev.SEEDS) for k, v in rows.items()}
    if switched != 9675 or out['matched']['risk'][GAIN] != prev.receipt(prev.PUBLIC, 'replay')['grouped']['risk'][GAIN]:
        raise ValueError('Original controller does not replay')
    draw = np.random.default_rng(73107).integers(len(groups), size=(3000, len(groups)))
    cube = np.asarray([[geometry[g, s] for s in prev.SEEDS] for g in groups])
    out['pair_geometry_matched'] = {k: {m: pair.bootstrap(cube[:, :, i, j], draw)
        for j, m in enumerate(pair.PAIR_FIELDS)} for i, k in enumerate(comp)}
    return dict(out, matched_switches=switched, substitutions=substituted, inference_uses_past_only=True,
        source_groups=len(groups), recordings=len(records), new_training=False, external_access=False,
        independent_confirmation=False, deployment=False, stage5c_executed=False, smc_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'freeze', 'score', 'replay'))
    phase = p.parse_args().phase
    cut = register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', f'HEAD:{f.relative_to(ROOT)}'], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the protocol before real-data inference')
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**v):
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event)
            print(event, flush=True)

        if phase != 'freeze':
            prev.receipt(PUBLIC, 'freeze')
        if phase == 'replay':
            prev.receipt(PUBLIC, 'score')
        d = run.population()
        proposal = freeze(d)
        if phase == 'freeze':
            out = PRIVATE/'freeze.json'
        else:
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(score(d, proposal, cut, beat)))
        mark = PUBLIC/f'{phase}_execution.json'
        if not mark.exists():
            run.once(mark, dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out)))
        elif run.read(mark)['artifact'] != run.artifact(out):
            raise ValueError('Prefix forecast replay differs')


if __name__ == '__main__':
    main()
