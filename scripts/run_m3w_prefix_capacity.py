"""Solve matched bank capacity and locate which nested constraints remove it."""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_directional_utility as directional
from src.evaluation import m3w_prefix_capacity as model

run, np, prior = directional.run, directional.np, directional.prior
PUBLIC, PRIVATE = prior.PUBLIC/'prefix_capacity', prior.PRIVATE/'prefix_capacity'
PARENT_REG = PUBLIC/'registration.json'
REG = PUBLIC/'registration_v2.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_prefix_capacity.py')
TIME_LIMIT = 8.
FEASIBLE = {'optimal', 'feasible_unproven'}


def register():
    parent = directional.PUBLIC
    for phase in ('score', 'replay'):
        if run.read(parent/f'{phase}_execution.json')['artifact']['sha256'] != run.sha(parent/'analysis.json'):
            raise ValueError('Require frozen parent scoring and replay')
    inputs = (PARENT_REG, directional.REG, parent/'freeze.json', parent/'analysis.json',
              parent/'replay_execution.json', directional.size.REG, directional.size.PUBLIC/'freeze.json',
              Path(run.model.__file__), Path(run.parent.readout.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        inputs=[run.artifact(p) for p in inputs], modes=model.MODES, time_limit=TIME_LIMIT,
        expected_views=102, expected_switches=6952, original_budget=.02,
        label_aware_only=True, protected_outcomes_accessed=False,
        lossless_rowwise_dominance=True, prefix_presolve_disabled=True,
        reuse_completed_original_checkpoints=True)))


def summarize(rows):
    if (len(rows) != 102 or len({r['identity']['source'] for r in rows}) != 14
            or sum(r['direct_switches'] for r in rows) != 6952):
        raise ValueError('Require the complete frozen roster and intervention total')
    summary = {}
    for mode in model.MODES:
        arms = [r['modes'][mode] for r in rows]
        valid = [a for a in arms if a['status'] in FEASIBLE]
        summary[mode] = dict(status_counts=dict(Counter(a['status'] for a in arms)),
            infeasible_reasons=dict(Counter(a['reason'] for a in arms if a['status']=='infeasible')),
            unresolved_reasons=dict(Counter(a['reason'] for a in arms if a['status']=='unresolved')),
            positive_paired_witness=sum(a['paired_lower_mass']>1e-9 for a in valid),
            positive_cv_witness=sum(a['vs_cv_lower_mass']>1e-9 for a in valid),
            whole_roster_optimal_paired_lower_pp=None)
        if all(a['status']=='optimal' for a in arms):
            by_source = {s:[] for s in sorted({r['identity']['source'] for r in rows})}
            for s, z in by_source.items():
                for k in run.kin.prior.SEEDS:
                    group = [r for r in rows if r['identity']['source']==s and r['identity']['seed']==k]
                    z.append(100*sum(r['modes'][mode]['paired_lower_mass'] for r in group)
                             /sum(r['reference'] for r in group))
            source = np.asarray(list(by_source.values())).mean(1)
            draws = np.random.default_rng(73107).integers(14, size=(3000,14))
            summary[mode]['whole_roster_optimal_paired_lower_pp'] = dict(
                point=float(source.mean()), interval95=np.quantile(source[draws].mean(1),[.025,.975]).tolist())
    transitions = {}
    for left, right in zip(model.MODES, model.MODES[1:]):
        transitions[f'{left}_to_{right}'] = dict(
            proved_newly_blocked=sum(r['modes'][left]['status'] in FEASIBLE and
                                     r['modes'][right]['status']=='infeasible' for r in rows),
            newly_unresolved=sum(r['modes'][left]['status'] in FEASIBLE and
                                 r['modes'][right]['status']=='unresolved' for r in rows),
            optimal_loss_in_positive_utility=sum(
                r['modes'][left]['status']=='optimal' and r['modes'][right]['status']=='optimal' and
                r['modes'][left]['paired_lower_mass']>1e-9 and
                r['modes'][right]['paired_lower_mass']<=1e-9 for r in rows))
    return run.stable(dict(summary=summary, transitions=transitions, records=rows,
        label_aware_only=True, no_label_aware_choices_exported=True, original_rules_unchanged=True,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))


def execute(phase, beat):
    d, refs = run.population(), run.register()
    labels = run.old.prior.arrays(('target_eval','valid','frames'))
    records = prior.blocks(d,labels['frames'])
    y, valid = labels['target_eval']-d['origin'][:,None], labels['valid']
    cuts = run.cuts_by_source()
    parent = run.read(directional.PUBLIC/'freeze.json')['cells']
    controls = {c['seed']:c['selection'] for c in run.read(directional.size.PUBLIC/'freeze.json')['cells']}
    old = {(r['seed'],r['recording']):r for r in run.read(directional.PUBLIC/'analysis.json')['records']}
    out = []
    for seed_cell in parent:
        seed = seed_cell['seed']
        if phase=='score':
            cv, bank = run.candidates(d,refs,seed)
            dc = np.full(len(cv),-1,np.int16)
            coefs = {}
            original = {c['identity']['held']:c for c in controls[seed]}
            for f in seed_cell['folds']:
                s = f['identity']['held']
                at = d['sites']==s
                with np.load(run.checked(f['cache']),allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],d['global_ids'][at])
                    dc[at] = z['choices'][0]
                with np.load(run.checked(original[s]['cache']),allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['ids'],d['global_ids'][at])
                    np.testing.assert_array_equal(z['choices'][0],dc[at])
                    coefs[s] = z['baseline_coefficients'].copy()
        for recording,(ix,groups) in records.items():
            s = str(d['sites'][ix[0]])
            identity = dict(seed=seed, source=s, recording=recording, registration=run.sha(REG))
            path = PRIVATE/f'{seed}_{recording}.json'
            if not path.exists():
                if phase=='replay':
                    raise ValueError('Replay requires every completed solve')
                floor, b = cv[ix], bank[ix]
                x, scale = run.model.descriptors(d['history'][ix],b,floor,d['crowd'][ix])
                _, a, e, legal = run.model.decision_costs(x,coefs[s],b,floor,scale)
                direct = run.model.selected_paths(dc[ix],b,floor)
                easy, hard = (cuts[s][k] for k in ('easy','hard'))
                p = model.prepare(floor,b,direct,dc[ix],y[ix],valid[ix],groups,easy,np.stack((a,e)),legal)
                n = int(p['fixed'].sum())
                if n != old[seed,recording]['metrics']['direct']['risk']['selected']:
                    raise ValueError('Direct count must agree with its exact original readout')
                modes = {}
                infeasible = None
                for mode in model.MODES:
                    if infeasible:
                        modes[mode] = dict(status='infeasible',reason='inherited_'+infeasible)
                        continue
                    beat(state='solving', seed=seed, complete=len(out), mode=mode)
                    v = model.solve(p,floor,b,direct,y[ix],valid[ix],recording,easy,hard,mode,TIME_LIMIT)
                    modes[mode] = v
                    if v['status']=='infeasible':
                        infeasible = mode
                for left,right in zip(model.MODES,model.MODES[1:]):
                    a,b0 = modes[left],modes[right]
                    if (a['status']=='optimal' and b0['status'] in FEASIBLE and
                            b0['paired_lower_mass']>a['paired_lower_mass']+1e-7):
                        raise ValueError('A tighter problem cannot exceed the looser optimum')
                run.once(path,run.stable(dict(identity=identity, reference=p['reference'],
                                              direct_switches=n, modes=modes)))
            row = run.read(path)
            original_identity = {**identity,'registration':run.sha(PARENT_REG)}
            if row['identity'] not in (identity, original_identity):
                raise ValueError('Cannot resume another registered recording view')
            out.append(row)
            beat(state='recording_done', seed=seed, complete=len(out))
    data = summarize(out)
    data['original_solver_checkpoints_reused'] = sum(
        r['identity']['registration']==run.sha(PARENT_REG) for r in out)
    run.once(PUBLIC/'analysis.json',data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','score','replay'))
    phase = p.parse_args().phase
    register()
    if phase=='register':
        return
    for path in (*FILES,REG):
        if subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],
                                   cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit the diagnostic definition first')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def beat(**fields):
            if time.monotonic()-start>43200 or shutil.disk_usage(ROOT).free<2*1024**3:
                raise OSError('Resume from completed recording checkpoints')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **fields)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json',msg)
            if fields['state']=='recording_done' and fields['complete']%8==0:
                print(msg,flush=True)
        execute(phase,beat)
        run.once(PUBLIC/f'{phase}_execution.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            fresh_solves=phase=='score',aggregate_replay_only=phase=='replay',
            artifact=run.artifact(PUBLIC/'analysis.json')))


if __name__=='__main__':
    main()
