"""Matched visibility-feature ablation on all admitted native recordings."""
import argparse
import csv
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_optional_withdrawal as preceding
from src.world_model import m3w_box_visibility as model

h, run, np, anchor = preceding.h, preceding.run, preceding.np, preceding.anchor
learner = preceding.previous.previous.model
PUBLIC, PRIVATE = preceding.PUBLIC.parent/'box_visibility', preceding.PRIVATE.parent/'box_visibility'
ADMISSION = ROOT/'outputs/publication_readiness_2026_09/moving_development_admission_v1/admission.json'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_box_visibility.py')
ARMS = ('direct', 'trajectory', 'trajectory_boxes')
MODALITIES = ARMS[1:]


def register():
    a = preceding.PUBLIC/'analysis.json'
    if run.read(preceding.PUBLIC/'replay_execution.json')['artifact']['sha256'] != run.sha(a):
        raise ValueError('Require exact preceding replay')
    for artifact in (*run.read(preceding.REG)['parents'], *run.read(preceding.REG)['files']):
        run.checked(artifact)
    paths = (a, preceding.REG, preceding.PUBLIC/'insertion_diagnostic.json', ADMISSION,
             run.old.prior.PARENT/'population.json', Path(learner.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in paths], budget=.02, observation_steps=8,
        steps=learner.STEPS, batch=learner.BATCH, arms=ARMS, cuts=h.model.CUTS,
        independent_outcomes_opened=False)))


def build():
    d = run.population()
    inputs = run.old.prior.arrays(('agents', 'frames'))
    agents, frames = inputs['agents'], inputs['frames']
    meta = run.read(ADMISSION)['recordings']
    names = sorted(meta)
    ids = np.unique(d['recordings'])
    out = np.empty((len(agents), 27), dtype=np.float32)
    for ident in ids:
        at = np.flatnonzero(d['recordings'] == ident)
        entry = meta[names[ident]]
        p = ROOT/entry['source_annotation_path']
        if run.sha(p) != entry['source_annotation_sha256'] or set(d['sites'][at]) != {entry['exclusion_group']}:
            raise ValueError('Box source or exclusion-group mismatch')
        with p.open(encoding='utf-8-sig', newline='') as f:
            rows = np.asarray([[float(v) for v in row[:6]] for row in csv.reader(f)
                               if row and any(x.strip() for x in row)])
        if (rows.ndim != 2 or rows.shape[1] != 6 or not np.isfinite(rows).all()
                or (rows[:, :2] != np.floor(rows[:, :2])).any()):
            raise ValueError('Invalid admitted box rows')
        out[at] = model.aligned(rows, agents[at], frames[at], d['history'][at])
    if not np.isfinite(out).all():
        raise ValueError('Invalid box feature')
    file = PRIVATE/'past_boxes.npz'
    h.prune.previous.history.feedback.write_array(file, ids=d['global_ids'], features=out)
    run.once(PUBLIC/'feature_receipt.json', run.stable(dict(cache=run.artifact(file),
        rows=len(out), recordings=len(ids), dimensions=out.shape[1], observed_steps=8,
        centers_exact=True, future_target_members_opened=False, outcome_labels_in_features=False,
        annotation_geometry_not_occlusion_gold=True)))


def fit(m, controls, motion, boxes, y, valid, beat):
    d, cuts = m.d, run.cuts_by_source()
    moments, stats = {}, {}
    atoms = np.empty((*m.bank.shape[:2], 6))
    for s, ix in m.ix.items():
        x, b, c, scale = m.x[ix], m.bank[ix], m.floor[ix], m.scale[ix]
        events, atoms[ix], _ = anchor.model.labels(c, b, y[ix], valid[ix], cuts[s]['easy'])
        legal = np.linalg.norm(b-c[:, None], axis=-1).max(2) > 0.
        moments[s] = run.model.fitting_moments(x, b, c, y[ix], valid[ix], scale, 1./d['crowd'][ix], cuts[s]['easy'])
        stats[s] = anchor.model.fitting(x, events, legal, d['crowd'][ix])
    folds = []
    width = boxes.shape[1]
    for item in controls:
        r = item['identity']
        site, learn, cal = r['held'], r['fit'], r['cal_a'] + r['cal_b']
        ident = dict(seed=m.seed, held=site, fit=learn, cal=cal, registration=run.sha(REG),
                     feature_receipt=run.sha(PUBLIC/'feature_receipt.json'))
        if (len(learn), len(cal), len(set(learn+cal+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        mark = PRIVATE/f'{m.seed}_{site}.json'
        if not mark.exists():
            blocks = []
            for s in learn:
                co = run.model.coefficients([moments[t] for t in learn if s != t])
                rank = anchor.model.fit([stats[t] for t in learn if s != t])
                local = []
                for ix, groups in m.groups[s]:
                    dc, legal, i, k, x = h.features(m, ix, groups, co, rank, motion)
                    if len(i):
                        t = learner.supervision(atoms[ix], dc)[0][i, k, 1]
                        w = 1./(d['crowd'][ix[i]]*legal.sum(1)[i])
                        local.append((np.column_stack((x, boxes[ix[i]])),
                            np.column_stack((atoms[ix[i], k, 1] > 0., t)), w/w.sum()))
                blocks.extend((x, t, w/len(local)) for x, t, w in local)
            x, target, weight = [np.concatenate(v) for v in zip(*blocks)]
            center = np.average(x, weights=weight, axis=0)
            spread = np.sqrt(np.average((x-center)**2, weights=weight, axis=0)).clip(.01)

            def normal(x):
                return np.clip((x-center)/spread, -8., 8.).astype(np.float32)

            seed = int.from_bytes(hashlib.sha256(f'boxes:{m.seed}:{site}'.encode()).digest()[:4], 'little')
            xx = normal(x)
            nets, checkpoints = [], []
            for arm, x_train in zip(MODALITIES, (xx[:, :-width], xx)):
                ck = mark.with_name(f'{mark.stem}_{arm}.pt')
                nets.append(learner.train(x_train, target, weight, seed, ck,
                    lambda **v: beat(state='training', seed=m.seed, source=site, arm=arm, **v)))
                checkpoints.append(run.artifact(ck))
            co, rank = m.co[site], m.rank(site)

            def options(ix, g):
                dc, legal, i, k, x = h.features(m, ix, g, co, rank, motion)
                x = normal(np.column_stack((x, boxes[ix[i]])))
                result = []
                for net, xx in zip(nets, (x[:, :-width], x)):
                    scores = np.zeros((*legal.shape, 3), np.float32)
                    if len(i):
                        pair = learner.predict(net, xx)
                        scores[i, k, 0] = pair[:, 0]
                        scores[i, k, 1] = pair.prod(1)
                    result.append(list(learner.options(g, legal, scores, 1)))
                return dc, result

            donors = [[], []]
            for s in cal:
                for ix, g in m.groups[s]:
                    dc, candidates = options(ix, g)
                    a = atoms[ix]
                    mass, full = preceding.previous.mass(a, dc)
                    old = anchor.model.risk(mass, full)
                    for dest, opts in zip(donors, candidates):
                        dest.append((opts, a, mass, full, d['crowd'][ix], old))
            thresholds = [h.calibrate(donor) for donor in donors]
            at = m.ix[site]
            choices = np.broadcast_to(m.direct[at], (len(ARMS), len(at))).copy()
            top = np.full((2, len(at)), -1, dtype=np.int16)
            for ix, g in m.groups[site]:
                dc, candidates = options(ix, g)
                np.testing.assert_array_equal(dc, m.direct[ix])
                for j, (opts, cut) in enumerate(zip(candidates, thresholds)):
                    if opts:
                        i, k, *_ = opts[0]
                        top[j, np.searchsorted(at, ix[i])] = k
                    pick = h.model.first(opts, cut)
                    if pick is not None:
                        i, k = pick
                        choices[j+1, np.searchsorted(at, ix[i])] = k
            cache = mark.with_suffix('.npz')
            h.prune.previous.history.feedback.write_array(cache, ids=d['global_ids'][at],
                choices=choices, top=top, center=center, spread=spread)
            run.once(mark, run.stable(dict(identity=ident, cache=run.artifact(cache), checkpoints=checkpoints,
                thresholds=thresholds, fitting_actions=len(target), positive_labels=target.sum(0).tolist())))
        f = run.read(mark)
        if f['identity'] != ident:
            raise ValueError('Incompatible fold checkpoint')
        for artifact in (f['cache'], *f['checkpoints']):
            run.checked(artifact)
        folds.append(f)
        beat(state='fold_frozen', seed=m.seed, complete=len(folds))
    return dict(seed=m.seed, folds=folds)


def score(m, records, frozen, y, valid):
    d, p, cuts = m.d, run.parent, run.cuts_by_source()
    choices = np.full((3, len(m.floor)), -1, np.int16)
    top = np.full((2, len(m.floor)), -1, np.int16)
    for f in frozen['folds']:
        at = m.ix[f['identity']['held']]
        with np.load(run.checked(f['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'][at])
            choices[:, at], top[:, at] = z['choices'], z['top']
    np.testing.assert_array_equal(choices[0], m.direct)
    rows = []
    for rec, (ix, g) in records.items():
        s = str(d['sites'][ix[0]])
        e, hard = cuts[s]['easy'], cuts[s]['hard']
        z, c, b = choices[:, ix], m.floor[ix], m.bank[ix]
        changed = z != z[0]
        if (changed[:, z[0] >= 0].any() or (changed.sum(1) > 1).any()
                or any(((z[:, q] >= 0).sum(1) > min(2, len(q)-1)).any() for q in g)):
            raise ValueError('Expected at most one legal insertion')
        paths = {k: run.model.selected_paths(v, b, c) for k, v in zip(ARMS, z)}
        contrasts = {k: ('direct', k, p.CV) for k in MODALITIES}
        contrasts['boxes_vs_trajectory'] = ('trajectory', 'trajectory_boxes', p.CV)
        out = p.readout.score_recording({p.CV: c, **paths}, contrasts,
            y[ix], valid[ix], d['queries'][ix], rec, easy_cut=e, hard_cut=hard)
        a = anchor.model.labels(c, b, y[ix], valid[ix], e)[1]
        target = learner.supervision(a, z[0])[0]
        probes = []
        for v in top[:, ix]:
            i = np.flatnonzero(v >= 0)
            if len(i) > 1:
                raise ValueError('Require one first-query probe')
            probes.append([bool((a[i, v[i], 1] > 0.).any()), bool(target[i, v[i], 1].any())])
        rows.append(dict(source=s, seed=m.seed, comparisons=out['comparisons'],
            risk={k: out['metrics'][k]['risk'] for k in ARMS}, probes=probes,
            insertions=changed.sum(1).tolist()))
    return rows


def report(rows, freeze):
    sites = sorted({r['source'] for r in rows})
    missing = [r for r in rows if r['risk']['direct']['completion_easy_risk_upper'] is None]
    if (len(sites), len(rows), len(missing)) != (14, 102, 35):
        raise ValueError('Require original cohort')
    groups = [[r for r in rows if r['source'] == s and r['seed'] == k]
              for s in sites for k in run.kin.prior.SEEDS]
    draw = np.random.default_rng(73107).integers(14, size=(3000, 14))

    def summary(x):
        x = np.asarray(x, float).reshape(14, 3).mean(1)
        return dict(point=float(x.mean()), interval95=np.quantile(x[draw].mean(1), [.025, .975]).tolist())

    risk, probe = {}, {}
    for j, a in enumerate(ARMS):
        v = [r['risk'][a] for r in rows]
        risk[a] = dict(passes=sum(x['supported'] for x in v), switches=sum(x['selected'] for x in v),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in v),
            easy_net_over_2=sum(x['known_easy_net_degradation'] is not None
                                and x['known_easy_net_degradation'] > .02 for x in v),
            insertions=sum(r['insertions'][j] for r in rows),
            new_passes=sum(r['risk'][a]['supported'] and not r['risk']['direct']['supported'] for r in rows),
            lost_passes=sum(not r['risk'][a]['supported'] and r['risk']['direct']['supported'] for r in rows))
        if j:
            probe[a] = {f'{scope}_{key}': sum(r['probes'][j-1][k] for r in group)
                for scope, group in (('all', rows), ('missing', missing))
                for k, key in enumerate(('support', 'rescue'))}
    if any(v != run.read(preceding.PUBLIC/'analysis.json')['risk']['direct'][k] for k, v in risk['direct'].items()):
        raise ValueError('Require unchanged direct score')
    folds = [v for c in freeze['cells'] for v in c['folds']]
    utility = {a: summary([run.parent.readout.merge_comparisons([r['comparisons'][a] for r in g])
                          ['completion_lower_pp'] for g in groups])
               for a in (*MODALITIES, 'boxes_vs_trajectory')}
    contrasts = {k: summary([np.mean([int(r['probes'][1][j])-int(r['probes'][0][j]) for r in g])
                             for g in groups]) for j, k in enumerate(('first_support', 'first_rescue'))}
    run.once(PUBLIC/'analysis.json', run.stable(dict(views=102, missing_views=35, source_groups=14,
        risk=risk, probe=probe, paired_lower_pp=utility, contrasts=contrasts, torch_models=2*len(folds),
        calibrated_folds={k: sum(f['thresholds'][j] is not None for f in folds)
                          for j, k in enumerate(MODALITIES)},
        feature_receipt_sha256=run.sha(PUBLIC/'feature_receipt.json'), original_rules_unchanged=True,
        independent_confirmation=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'build', 'fit', 'score', 'replay'))
    phase = parser.parse_args().phase
    register()
    if phase == 'register':
        return
    for path in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(path.relative_to(ROOT))],
                                   cwd=ROOT) != path.read_bytes():
            raise ValueError('Commit registration before execution')
    learner.torch.set_num_threads(4)
    learner.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the last saved update')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            if v.get('state') != 'training' or v.get('step') == learner.STEPS:
                print(msg, flush=True)

        beat(state='start')
        if phase == 'build':
            build()
            return
        d, refs = run.population(), run.register()
        box_receipt = run.read(PUBLIC/'feature_receipt.json')
        with np.load(run.checked(box_receipt['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids'])
            boxes = z['features']
        lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        records = h.previous.prior.blocks(d, lab['frames'])
        y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
        motion = h.previous.model.motion(d['history'], d['groups']) if phase == 'fit' else None
        parent, controls, ranks = h.cost.banks()
        out = PUBLIC/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen decisions')
            frozen = run.read(out)
            saved = {c['seed']: c for c in frozen['cells']}
        cells, rows = [], []
        for cell in parent:
            m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
            if phase == 'fit':
                cells.append(fit(m, controls[m.seed], motion, boxes, y, valid, beat))
            else:
                rows.extend(score(m, records, saved[m.seed], y, valid))
            beat(state='seed_complete', seed=m.seed)
            del m
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=cells, held_excluded=True)))
        else:
            report(rows, frozen)
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            artifact=run.artifact(out if phase == 'fit' else PUBLIC/'analysis.json'),
            python_architecture=platform.machine(), torch_threads=learner.torch.get_num_threads()))


if __name__ == '__main__':
    main()
