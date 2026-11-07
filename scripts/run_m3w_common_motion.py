"""Test a bounded scene-common forecast while holding OLS4 geometry fixed."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_joint_path_audit as audit
from src.evaluation.m3w_grouped_recording_readout import grouped_readout
from src.world_model import m3w_common_motion as model

run, np, h, torch = audit.run, audit.np, audit.h, model.torch
PUBLIC = audit.PUBLIC.parent/'moving_common_motion_v1'
PRIVATE = run.PRIVATE/'common_motion'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_common_motion.py')
FIXED = ('ols4', 'cv_center', 'kinematic_center', 'uniform_center')
ARMS = (*FIXED, 'donor_selected_center', 'learned_center')


def register():
    a, replay = (audit.PUBLIC/n for n in ('analysis.json', 'replay_execution.json'))
    if run.read(replay)['analysis_sha256'] != run.sha(a) or run.read(a)['external_labels_opened']:
        raise ValueError('Require frozen audit results')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES],
        parents=[run.artifact(p) for p in (a, replay, audit.REG, Path(audit.__file__))],
        arms=ARMS, settings=dict(steps=model.STEPS, batch=model.BATCH, validate=model.VALIDATE,
        checkpoint_every=model.CHECKPOINT), primary='learned_center_vs_ols4',
        original_rules_unchanged=True, external_outcomes_opened=False)))


def design(d, records, floor, bank, y, valid):
    ols, kin = (bank[:, run.NAMES.index(a)] for a in (run.parent.OLS, 'self'))
    x, offsets, rows, source, by_record = [], [], [], [], {}
    for rec, (ix, groups) in records.items():
        site = str(d['sites'][ix[0]])
        begin = len(rows)
        for g in groups:
            q = ix[g]
            f, o = model.describe(d['history'][q], np.stack((ols[q], floor[q], kin[q])))
            x.append(f); offsets.append(o); rows.append(q); source.append(site)
        by_record[rec] = np.arange(begin, len(rows))
    return dict(x=np.asarray(x), offsets=np.asarray(offsets), rows=rows, source=np.asarray(source),
        by_record=by_record, ols=ols, y=np.where(valid[..., None], y, 0.), valid=valid)


def score_weights(data, ids, weights):
    """Lower gain against OLS4 with fixed query weights on donor sources."""
    vals = []
    for q, w in zip(ids, weights):
        at = data['rows'][q]
        ols, y, valid = (data[k][at] for k in ('ols', 'y', 'valid'))
        pred = model.shifted(ols, data['offsets'][q], w)
        change = np.linalg.norm(ols-y, axis=-1)-np.linalg.norm(pred-y, axis=-1)
        n = valid.sum(1)
        gain = (change*valid).sum(1)/n.clip(1)
        radius = np.linalg.norm(pred-ols, axis=-1).max(1)
        vals.append(float(np.where(n > 0, gain, -radius).mean()))
    site = data['source'][ids]
    vals = np.asarray(vals)
    return float(np.mean([vals[site == s].mean() for s in np.unique(site)]))


def train_fold(data, learn, donor, identity, ck, seed, beat):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    source = data['source']
    train_ids = np.flatnonzero(np.isin(source, learn))
    val_ids = np.flatnonzero(np.isin(source, donor))
    sizes = dict(zip(*np.unique(source[train_ids], return_counts=True)))
    mass = np.array([1./sizes[s] for s in source[train_ids]])
    x = data['x']
    centre = np.average(x[train_ids], axis=0, weights=mass)
    spread = np.sqrt(np.average((x[train_ids]-centre)**2, axis=0, weights=mass)).clip(.01)
    x = ((x-centre)/spread).clip(-8., 8.).astype(np.float32)
    net = model.make_net(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.001)
    start, best, best_step, best_state = 0, -np.inf, 0, None
    if ck.exists():
        old = torch.load(ck, map_location='cpu', weights_only=True)
        if old['identity'] != identity:
            raise ValueError('Incompatible common-motion checkpoint')
        net.load_state_dict(old['model'])
        opt.load_state_dict(old['optimizer'])
        start, best, best_step, best_state = (old[k] for k in ('step', 'best', 'best_step', 'best_state'))
    plan = rng.choice(train_ids, (model.STEPS, model.BATCH), p=mass/mass.sum())
    to_float = lambda a: torch.as_tensor(a, dtype=torch.float32)
    for t in range(start, model.STEPS):
        batch = plan[t]
        rows = [data['rows'][q] for q in batch]
        at = np.concatenate(rows)
        q = torch.from_numpy(np.repeat(np.arange(len(rows)), [len(i) for i in rows]))
        loss = model.batch_loss(net, to_float(x[batch]), to_float(data['offsets'][batch]),
            to_float(data['ols'][at]), to_float(data['y'][at]), torch.from_numpy(data['valid'][at]), q)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite common-motion objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if t+1 in model.VALIDATE:
            score = score_weights(data, val_ids, model.predict(net, x[val_ids]))
            if score > best:
                best, best_step = score, t+1
                best_state = {name: v.clone() for name, v in net.state_dict().items()}
        if (t+1) % model.CHECKPOINT == 0:
            tmp = ck.with_suffix('.pending')
            torch.save(dict(identity=identity, step=t+1, model=net.state_dict(), optimizer=opt.state_dict(),
                best=best, best_step=best_step, best_state=best_state), tmp)
            tmp.replace(ck)
            beat(step=t+1, loss=float(loss.detach()))
    net.load_state_dict(best_state)
    fixed = [score_weights(data, val_ids, np.broadcast_to(w, (len(val_ids), 3))) for w in model.FIXED]
    return model.predict(net, x), int(np.argmax(fixed)), best_step


def fit(data, seed, controls, beat):
    folds = []
    for item in controls:
        r = item['identity']
        site, learn, donor = r['held'], r['fit'], r['cal_a']+r['cal_b']
        if (len(learn), len(donor), len(set(learn+donor+[site]))) != (7, 6, 14):
            raise ValueError('Require disjoint source roles')
        ident = dict(seed=seed, held=site, fit=learn, donor=donor, registration=run.sha(REG))
        mark = PRIVATE/f'{seed}_{site}.json'
        if not mark.exists():
            key = int.from_bytes(hashlib.sha256(f'common:{seed}:{site}'.encode()).digest()[:4], 'little')
            ck = mark.with_suffix('.pt')
            w, fixed, step = train_fold(data, learn, donor, ident, ck, key,
                lambda **v: beat(state='training', seed=seed, source=site, **v))
            at = np.flatnonzero(data['source'] == site)
            cache = mark.with_suffix('.npz')
            h.prune.previous.history.feedback.write_array(cache, query=at, learned=w[at],
                selected=np.repeat(fixed, len(at)))
            run.once(mark, run.stable(dict(identity=ident, checkpoint=run.artifact(ck),
                cache=run.artifact(cache), selected_fixed=FIXED[fixed], selected_step=step)))
        fold = run.read(mark)
        if fold['identity'] != ident:
            raise ValueError('Incompatible common-motion fold')
        for asset in (fold['checkpoint'], fold['cache']):
            run.checked(asset)
        folds.append(fold)
        beat(state='fold_frozen', seed=seed, complete=len(folds))
    return dict(seed=seed, folds=folds)


def score(data, d, records, floor, seed, frozen, y, valid):
    learned = np.full((len(data['x']), 3), np.nan)
    fixed = np.full(len(learned), -1, int)
    for fold in frozen['folds']:
        at = np.flatnonzero(data['source'] == fold['identity']['held'])
        with np.load(run.checked(fold['cache']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['query'], at)
            learned[at], fixed[at] = z['learned'], z['selected']
    if (fixed < 0).any() or not np.isfinite(learned).all():
        raise ValueError('Incomplete frozen common motion')
    cuts, cv, ols = run.cuts_by_source(), run.parent.CV, data['ols']
    rows = []
    for rec, (ix, groups) in records.items():
        site = str(d['sites'][ix[0]])
        paths = {k: ols[ix].copy() for k in ARMS}
        for q, g in zip(data['by_record'][rec], groups):
            weights = (*model.FIXED, model.FIXED[fixed[q]], learned[q])
            o = ols[ix[g]]
            for name, w in zip(ARMS, weights):
                pred = model.shifted(o, data['offsets'][q], w)
                np.testing.assert_allclose(pred-pred[:1], o-o[:1], atol=1e-10, rtol=1e-10)
                paths[name][g] = pred
        pairs = {f'{k}_vs_cv': (cv, k, cv) for k in ARMS}
        pairs.update({f'{k}_vs_ols4': ('ols4', k, cv) for k in ARMS[1:]})
        pairs['learned_center_vs_donor_selected_center'] = ('donor_selected_center', 'learned_center', cv)
        out = run.parent.readout.score_recording({cv: floor[ix], **paths}, pairs, y[ix], valid[ix],
            d['queries'][ix], rec, easy_cut=cuts[site]['easy'], hard_cut=cuts[site]['hard'])
        rows.append(dict(out, seed=seed, source=site))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the common-motion protocol first')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the last saved common-motion checkpoint')
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            if v.get('state') != 'training' or v.get('step') == model.STEPS:
                print(msg, flush=True)

        d, refs = run.population(), run.register()
        labels = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        y, valid = labels['target_eval']-d['origin'][:, None], labels['valid']
        records = h.previous.prior.blocks(d, labels['frames'])
        roster = {r: str(d['sites'][ix[0]]) for r, (ix, _) in records.items()}
        _, controls, _ = h.cost.banks()
        out = PRIVATE/'freeze.json'
        if phase != 'fit':
            if run.read(PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(out):
                raise ValueError('Require frozen held-source weights')
            frozen = {c['seed']: c for c in run.read(out)['cells']}
        if phase == 'replay' and not (PUBLIC/'score_execution.json').exists():
            raise ValueError('Score before replay')
        fits, scores = [], []
        for seed in run.kin.prior.SEEDS:
            floor, bank = run.candidates(d, refs, seed)
            data = design(d, records, floor, bank, y, valid)
            if phase == 'fit':
                fits.append(fit(data, seed, controls[seed], beat))
            else:
                scores.extend(score(data, d, records, floor, seed, frozen[seed], y, valid))
            beat(state='seed_complete', seed=seed)
        if phase == 'fit':
            run.once(out, run.stable(dict(cells=fits, held_excluded=True)))
        else:
            out = PUBLIC/'analysis.json'
            result = grouped_readout(scores, roster, list(run.kin.prior.SEEDS))
            folds = [f for c in frozen.values() for f in c['folds']]
            from collections import Counter
            run.once(out, run.stable(dict(grouped=result, models=len(folds),
                selected_fixed=dict(Counter(f['selected_fixed'] for f in folds)),
                selected_step=dict(Counter(str(f['selected_step']) for f in folds)),
                pair_geometry_preserved=True, all_visible_diagnostic=True,
                original_rules_unchanged=True, independent_confirmation=False, external_labels_opened=False,
                deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False)))
        receipt = dict(pid=os.getpid(), seconds=time.monotonic()-start, artifact=run.artifact(out),
            python_architecture=platform.machine(), torch_threads=torch.get_num_threads())
        dest = PUBLIC/f'{phase}_execution.json'
        if dest.exists():
            if run.read(dest)['artifact'] != receipt['artifact']:
                raise ValueError('Previous execution binds another artifact')
        else:
            run.once(dest, receipt)


if __name__ == '__main__':
    main()
