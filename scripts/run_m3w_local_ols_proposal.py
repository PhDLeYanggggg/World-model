"""Fit and evaluate OLS4-anchored proposals on separated development sources."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use the native arm64 training environment')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_common_motion as prior
from scripts import audit_m3w_slate_geometry_capacity as geometry
from src.evaluation import m3w_grouped_recording_readout as grouped
from src.world_model import m3w_local_ols_proposal as model

run, np, h, torch = prior.run, prior.np, prior.h, model.torch
PUBLIC = prior.PUBLIC.parent/'native_local_ols_proposal_v1'
PRIVATE = geometry.PRIVATE.parent/'native_local_ols_proposal_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md',
         ROOT/'tests/test_m3w_local_ols_proposal.py')
KINDS = ('self', 'neighbor')
SEEDS = geometry.prior.SEEDS


def register():
    parents = (geometry.REG, geometry.PUBLIC/'development.json', prior.REG,
               Path(h.cost.banks.__code__.co_filename), h.cost.anchor.SIZE.PUBLIC/'freeze.json',
               Path(grouped.__file__),
               Path(run.parent.readout.__file__))
    run.once(REG, run.stable(dict(files=[run.artifact(f) for f in FILES],
        parents=[run.artifact(f) for f in parents], seeds=SEEDS, roles=[7, 6, 1],
        fit_steps=model.STEPS, fit_batch=model.BATCH, amplitudes=model.AMPLITUDES,
        original_recording_rules=True, workers=0, torch_threads=2, external_data_access=False,
        primary=('neighbor_selected_vs_ols4', 'neighbor_selected_vs_self_selected'),
        selection='max_donor_source_mean_completion_lower_vs_ols4_then_smaller_amplitude')))


def tensors(*xs):
    return [torch.as_tensor(x, dtype=torch.float32) for x in xs]


def prediction(net, x, rot, scale):
    with torch.no_grad():
        return torch.cat([model.shift(net, *p) for p in zip(x.split(4096), rot.split(4096),
                                                           scale.split(4096))]).numpy()


def train(train_rows, feature, ref, y, valid, weight, key, identity, ck, beat):
    x, rot, scale = tensors(*feature)
    ols, target = tensors(ref[train_rows], y[train_rows])
    mask = torch.from_numpy(valid[train_rows])
    p = weight[train_rows].copy()
    p /= p.sum()
    tx = x[train_rows]
    mean = (p @ tx.numpy()).astype(np.float32)
    spread = np.sqrt(p @ (tx.numpy()-mean)**2).clip(.1).astype(np.float32)
    x = ((x-torch.from_numpy(mean))/torch.from_numpy(spread)).clamp(-8., 8.)
    torch.manual_seed(key)
    net = model.make_net(x.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.001)
    start = 0
    if ck.exists():
        state = torch.load(ck, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Incompatible proposal checkpoint')
        start = state['step']
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
    plan = np.random.default_rng(key).choice(len(train_rows), (model.STEPS, model.BATCH), p=p)
    for step in range(start, model.STEPS):
        loc = plan[step]
        at = train_rows[loc]
        delta = model.shift(net, x[at], rot[at], scale[at])
        loss = model.loss(delta, ols[loc], target[loc], mask[loc], scale[at])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite proposal objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step+1) % 80 == 0:
            tmp = ck.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step+1, model=net.state_dict(),
                            optimizer=opt.state_dict()), tmp)
            tmp.replace(ck)
            beat(state='training', step=step+1, loss=float(loss.detach()), arm=identity['arm'])
    return net, (x, rot, scale)


def select(causal, delta, y, valid, query_weight, source):
    cv, ols = causal
    sites = [np.flatnonzero(source == s) for s in np.unique(source)]
    scores = []
    for a in model.AMPLITUDES:
        stats = [run.parent.readout.comparison_masses(ols[i], ols[i]+a*delta[i], cv[i],
                    y[i], valid[i], query_weight[i]) for i in sites]
        scores.append(float(np.mean([run.parent.readout.merge_comparisons([s])['completion_lower_pp']
                                     for s in stats])))
    return model.AMPLITUDES[int(np.argmax(scores))], scores


def fit(d, labels, records, refs, controls, beat):
    sources = d['sites']
    groups = [ix[g] for ix, query in records.values() for g in query]
    own, context, rot, scale = model.describe(d['history'], groups)
    qw = np.zeros(len(sources))
    for at in groups:
        qw[at] = 1./len(at)
    sites = sorted(set(sources))
    mass = {s: qw[sources == s].sum() for s in sites}
    w = qw / np.array([mass[s] for s in sources])
    y, valid = labels
    cells = []
    for seed in SEEDS:
        cv, bank = run.candidates(d, refs, seed)
        ols = bank[:, run.NAMES.index(run.parent.OLS)]
        for spec in controls[seed]:
            meta = spec['identity']
            held, learn, donor = meta['held'], meta['fit'], meta['cal_a']+meta['cal_b']
            if (len(learn), len(donor), len(set([held]+learn+donor))) != (7, 6, 14):
                raise ValueError('Roles overlap or are incomplete')
            fit_rows = np.flatnonzero(np.isin(sources, learn))
            val_rows = np.flatnonzero(np.isin(sources, donor))
            test_rows = np.flatnonzero(sources == held)
            for arm in KINDS:
                mark = PRIVATE/f'{seed}_{held}_{arm}.json'
                ident = dict(registration=run.sha(REG), seed=seed, held=held, learn=learn, donor=donor, arm=arm)
                if not mark.exists():
                    x = np.column_stack((own, context if arm == 'neighbor' else np.zeros_like(context)))
                    ck = mark.with_suffix('.pt')
                    key = int.from_bytes(hashlib.sha256(f'localols:{seed}:{held}'.encode()).digest()[:4], 'little')
                    net, feats = train(fit_rows, (x, rot, scale), ols, y, valid, w, key, ident, ck, beat)
                    dv = prediction(net, *(f[val_rows] for f in feats))
                    a, selection = select((cv[val_rows], ols[val_rows]), dv, y[val_rows], valid[val_rows],
                                          qw[val_rows], sources[val_rows])
                    out = mark.with_suffix('.npz')
                    tmp = out.with_suffix('.pending.npz')
                    np.savez(tmp, index=test_rows, delta=prediction(net, *(f[test_rows] for f in feats)))
                    tmp.replace(out)
                    run.once(mark, run.stable(dict(identity=ident, checkpoint=run.artifact(ck),
                        forecast=run.artifact(out), amplitude=a, donor_scores=selection)))
                row = run.read(mark)
                if row['identity'] != ident:
                    raise ValueError('Incompatible frozen fold')
                run.checked(row['forecast'])
                run.checked(row['checkpoint'])
                cells.append(row)
            beat(state='fold_frozen', seed=seed, completed=len(cells))
    if len(cells) != 84 or len({(r['identity']['seed'], r['identity']['held'], r['identity']['arm'])
                               for r in cells}) != 84:
        raise ValueError('Incomplete unique development folds')
    return dict(cells=cells, train_updates_per_fit=model.STEPS, scored_sources_excluded=True)


def score(d, labels, records, refs, frozen):
    y, valid = labels
    rows, roster = [], {}
    cv_name, ols_name = run.parent.CV, run.parent.OLS
    cut = run.read(geometry.REG)['cut']
    for seed in SEEDS:
        cv, bank = run.candidates(d, refs, seed)
        ols = bank[:, run.NAMES.index(ols_name)]
        forecasts = {k: np.full_like(ols, np.nan) for a in KINDS for k in (f'{a}_raw', f'{a}_selected')}
        for cell in frozen['cells']:
            meta = cell['identity']
            if meta['seed'] != seed:
                continue
            a = meta['arm']
            i = np.flatnonzero(d['sites'] == meta['held'])
            with np.load(run.checked(cell['forecast']), allow_pickle=False) as f:
                np.testing.assert_array_equal(f['index'], i)
                forecasts[f'{a}_raw'][i] = ols[i]+f['delta']
                forecasts[f'{a}_selected'][i] = ols[i]+cell['amplitude']*f['delta']
        if not all(np.isfinite(v).all() for v in forecasts.values()):
            raise ValueError('Require complete frozen proposals')
        pairs = {f'{a}_vs_{name}': (ref, a, cv_name) for a in forecasts
                 for name, ref in (('ols4', ols_name), ('cv', cv_name))}
        for suffix in ('raw', 'selected'):
            pairs[f'neighbor_{suffix}_vs_self_{suffix}'] = (f'self_{suffix}', f'neighbor_{suffix}', cv_name)
        for rec, (i, _) in records.items():
            site = str(d['sites'][i[0]])
            roster[rec] = site
            r = run.parent.readout.score_recording({cv_name: cv[i], ols_name: ols[i],
                **{k: v[i] for k, v in forecasts.items()}}, pairs, y[i], valid[i], d['queries'][i], rec,
                easy_cut=cut['easy'], hard_cut=cut['hard'])
            rows.append(dict(r, source=site, seed=seed))
    return grouped.grouped_readout(rows, roster, SEEDS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('register', 'fit', 'score', 'replay'))
    phase = p.parse_args().phase
    register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git', 'show', 'HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit proposal design before fitting')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            msg = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', msg)
            if v.get('state') != 'training':
                print(msg, flush=True)

        d, refs = run.population(), run.register()
        z = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
        labels = (z['target_eval']-d['origin'][:, None], z['valid'])
        records = h.previous.prior.blocks(d, z['frames'])
        private = PRIVATE/'freeze.json'
        if phase == 'fit':
            _, controls, _ = h.cost.banks()
            run.once(private, run.stable(fit(d, labels, records, refs, controls, beat)))
            out = private
        else:
            run.checked(run.read(PUBLIC/'fit_execution.json')['artifact'])
            if phase == 'replay':
                run.checked(run.read(PUBLIC/'score_execution.json')['artifact'])
            frozen = run.read(private)
            counts = {a: {str(k): sum(r['identity']['arm'] == a and r['amplitude'] == k
                                     for r in frozen['cells']) for k in model.AMPLITUDES} for a in KINDS}
            out = PUBLIC/'analysis.json'
            run.once(out, run.stable(dict(grouped=score(d, labels, records, refs, frozen),
                amplitude_counts=counts, fits=len(frozen['cells']), external_data_opened=False,
                result_source='fresh_torch_fits_on_exposed_development', independent_confirmation=False,
                deployment=False, stage5c_executed=False, smc_enabled=False)))
        receipt = PUBLIC/f'{phase}_execution.json'
        if not receipt.exists():
            run.once(receipt, dict(artifact=run.artifact(out), seconds=time.monotonic()-start,
                                    pid=os.getpid(), architecture=platform.machine(), torch_threads=2))
        elif run.read(receipt)['artifact'] != run.artifact(out):
            raise ValueError('Replay differs from the first receipt')


if __name__ == '__main__':
    main()
