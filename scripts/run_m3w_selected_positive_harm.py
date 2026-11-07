"""Train and compare selected-positive-harm heads at matched query counts."""
import argparse
import fcntl
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_nested_action_calibration as nested
from src.world_model import m3w_selected_positive_harm as model

run, np, torch = nested.run, nested.np, model.torch
PUBLIC = run.PUBLIC/'selected_positive_harm'
PRIVATE = run.PRIVATE/'selected_positive_harm'
REG = PUBLIC/'registration_v2.json'
FILES = (Path(__file__), Path(model.__file__), PUBLIC/'protocol.md', ROOT/'tests/test_m3w_selected_positive_harm.py')
STEPS, BATCH = 400, 128
FIT_ARMS = ('direct', 'factorized')


def register():
    paths = [nested.PUBLIC/n for n in ('analysis.json', 'replay_execution.json', 'selected_component_audit.json')]
    if run.read(paths[1])['artifact']['sha256'] != run.sha(paths[0]):
        raise ValueError('Require verified selected cost attribution and calibration')
    run.once(REG, run.stable(dict(files=[run.artifact(p) for p in FILES], parents=[run.artifact(p) for p in
        (*paths, PUBLIC/'registration.json', nested.REG, Path(nested.__file__), Path(nested.cal.__file__), run.REG,
         run.PUBLIC/'freeze.json', nested.base.PUBLIC/'freeze.json', nested.base.PUBLIC/'analysis.json')],
        arms=model.ARMS, steps=STEPS, batch=BATCH, aux_weight=.5, lr=.001, width=32,
        budget=.02, same_candidate_pool_and_query_counts=True, external_outcomes=False)))


def pack(b, c):
    u, a, e, legal = nested.predict(b, c)
    ok = model.admissible(u, a, e, legal)
    i, j = np.where(ok)
    rad = b['radius'][i, j]
    pred = nested.cal.predicted_components(b['x'], c, b['scale'], b['radius'])
    raw = c[:, :, 5]
    missing = (np.einsum('nd,nd->n', b['x'][i, j], raw[j]).clip(0., 1.) * rad)
    features = np.column_stack((b['x'][i, j, 1:], np.eye(ok.shape[1])[j],
        np.arcsinh(u[i, j]/b['scale'][i]), np.arcsinh(a[i, j]/rad), np.arcsinh(e[i, j]/rad),
        np.log1p(pred[i, j, 3]/b['scale'][i]), np.log1p(rad/b['scale'][i]))).astype(np.float32)
    return dict(x=features, index=(i, j), ok=ok, utility=u, all_excess=a, easy_excess=e, radius=rad,
        raw_harm=(pred[i, j, 2]-missing).clip(0.), missing=missing, easy_ref=pred[i, j, 3])


def training(blocks, inner, held, known):
    features, target, sampling, names = [], [], [], []
    cursor = 0
    for s, b in blocks.items():
        if s == held:
            continue
        p = pack(b, inner[tuple(sorted((s, held)))])
        r, a = p['index']
        if not len(r):
            continue
        qcodes = np.empty(len(b['ix']), np.int64)
        for k, g in enumerate(b['groups']):
            qcodes[g] = k
        groups = run.old.prior.groups(qcodes[r])
        z = np.where(known[b['ix'][r]], b['actual'][r, a, 2]/p['radius'], 0.)
        if np.any(z > 1. + 1e-8) or not np.isfinite(z).all():
            raise ValueError('Harm must lie within the causal path radius')
        sampling.append([g+cursor for g in groups]); cursor += len(r)
        features.append(p['x']); target.append(z.clip(0., 1.).astype(np.float32)); names.append(s)
    x, y = np.concatenate(features), np.concatenate(target)
    if not len(names) or not (y > 0.).any():
        raise ValueError('Selected region has no fitting support')
    mean = x.mean(0); sd = x.std(0).clip(1e-3)
    return dict(x=((x-mean)/sd).clip(-8., 8.), y=y, mean=mean, sd=sd, groups=sampling, sources=names)


def train_head(data, draws, key, arm, seed, path, stop, beat):
    x, y = torch.from_numpy(data['x']), torch.from_numpy(data['y'])
    positive = y > 0
    torch.manual_seed(seed)
    net = model.HarmHead(x.shape[1], positive.float().mean(), y[positive].mean())
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
    digest = run.kin.prior.source.array_hash
    identity = dict(cell=key, arm=arm, seed=seed, registration=run.sha(REG), fitting_sources=data['sources'],
        x_sha256=digest(data['x']), y_sha256=digest(data['y']), sampler_sha256=digest(draws))
    step = 0; trace = []
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        saved = state['identity']; old_reg = PUBLIC/'registration.json'
        old_head_bound = run.artifact(Path(model.__file__)) in run.read(old_reg)['files']
        allowed_v1 = (saved['registration'] == run.sha(old_reg) and old_head_bound and state['step'] == STEPS
                      and key == '17_moving_source_component_047')
        if saved != identity and not (allowed_v1 and dict(saved, registration=run.sha(REG)) == identity):
            raise ValueError('Cannot resume changed selection, label or sampling')
        net.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        step, trace = state['step'], state['trace']
    while step < stop:
        j = draws[step]; optimizer.zero_grad(set_to_none=True)
        loss = model.loss(net(x[j]), y[j], arm == 'factorized')
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step(); step += 1
        if step == 1 or step % 100 == 0 or step == stop:
            trace.append(dict(step=step, loss=float(loss.detach())))
        if step % 100 == 0 or step == stop:
            tmp = path.with_suffix('.pending'); torch.save(dict(identity=identity, step=step, trace=trace,
                model=net.state_dict(), optimizer=optimizer.state_dict()), tmp); tmp.replace(path)
    if step != stop:
        raise ValueError('Fixed registered budget required')
    beat(state='trained_head', cell=key, arm=arm, step=step)
    return net.eval(), dict(checkpoint=run.artifact(path), step=step, sampler_sha256=identity['sampler_sha256'],
                            fitting_candidates=len(y), positive_candidates=int(positive.sum()), trace=trace)


def fit_seed(d, refs, parent, label, beat, prefix):
    seed = parent['seed']; cv, paths = run.candidates(d, refs, seed)
    blocks = nested.prepare(d, paths, cv, label, run.cuts_by_source())
    inner = {(h, c): run.model.coefficients([b['moments'] for s, b in blocks.items() if s not in (h, c)])
             for i, h in enumerate(blocks) for c in list(blocks)[i+1:]}
    with np.load(run.checked(parent['cache']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], d['global_ids']); names, outer = z['names'], z['coefficients']
    ref = next(v for v in run.read(nested.base.PUBLIC/'freeze.json')['choices'] if v['seed'] == seed)
    with np.load(run.checked(ref['choices']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], d['global_ids']); raw = z['choice']
    choices = np.broadcast_to(raw, (len(model.ARMS), len(raw))).copy()
    pred_raw = np.zeros(choices.shape, dtype=float); receipts = []
    known = label['valid'].any(1)
    for held, coef in zip(names, outer):
        key = f'{seed}_{held}'; b = blocks[held]; p = pack(b, coef); ix = b['ix']
        fit = training(blocks, inner, held, known)
        if held in fit['sources']:
            raise ValueError('Held source in harm fit')
        draws = model.sampled_batches(fit['groups'], STEPS, BATCH, seed + 7919)
        values = [p['raw_harm'], p['raw_harm']]; stats = {}
        for arm in FIT_ARMS:
            path = PRIVATE/f'{key}_{arm}.pt'
            net, stats[arm] = train_head(fit, draws, key, arm, seed, path, 20 if prefix else STEPS, beat)
            if prefix:
                return dict(cell=key, arm=arm, **stats[arm])
            x = ((p['x']-fit['mean'])/fit['sd']).clip(-8., 8.)
            with torch.no_grad():
                estimate = net(torch.from_numpy(x)).sigmoid().prod(1).numpy().astype(float)
            values.append(estimate*p['radius'])
        if stats['direct']['sampler_sha256'] != stats['factorized']['sampler_sha256']:
            raise ValueError('Matched sampler changed')
        control = nested.base.select(b['groups'], p['utility'], p['all_excess'], p['easy_excess'], b['scale'], p['ok'])
        np.testing.assert_array_equal(control, raw[ix])
        r = np.flatnonzero(control >= 0)
        for k, estimate in enumerate(values):
            score = np.full(p['ok'].shape, -np.inf)
            i, j = p['index']
            adjusted = p['utility'][i, j] - (estimate + p['missing'] - .02*p['easy_ref'])
            score[i, j] = adjusted/b['scale'][i]
            if k:
                choices[k, ix] = model.matched(b['groups'], control, score, p['ok'])
            table = np.zeros(p['ok'].shape); table[i, j] = estimate
            pred_raw[k, ix[r]] = table[r, control[r]]
        receipts.append(dict(held=str(held), sources=fit['sources'], heads=stats))
    dest = PRIVATE/f'{seed}_choices.npz'; tmp = dest.with_suffix('.pending.npz')
    np.savez_compressed(tmp, ids=d['global_ids'], choices=choices, predicted_harm_on_raw=pred_raw); tmp.replace(dest)
    return dict(seed=seed, decisions=run.artifact(dest), fits=receipts)


def fit(d, refs, beat, prefix=False):
    label = run.old.prior.arrays(('target_eval', 'valid')); cells = []
    for v in run.read(run.PUBLIC/'freeze.json')['seeds']:
        p = PRIVATE/f'{v["seed"]}.json'
        if prefix:
            return fit_seed(d, refs, v, label, beat, True)
        if not p.exists():
            run.once(p, dict(registration=run.sha(REG), **fit_seed(d, refs, v, label, beat, False)))
        r = run.read(p)
        if r['registration'] != run.sha(REG):
            raise ValueError('Cannot reuse different fitting split or objective')
        run.checked(r['decisions']); cells.append(r)
    run.once(PUBLIC/'freeze.json', dict(cells=cells, all_choices_frozen=True, matched_sampling=True))
    heads = [a for v in cells for cell in v['fits'] for a in cell['heads'].values()]
    if len(heads) != 84 or any(a['step'] != STEPS for a in heads):
        raise ValueError('All 84 complete registered heads required')
    return dict(cells=len(cells), heads=len(heads), updates=sum(a['step'] for a in heads))


def score(d, refs, beat):
    p = run.parent; freeze = run.read(PUBLIC/'freeze.json')
    if not freeze['all_choices_frozen'] or len(freeze['cells']) != 3:
        raise ValueError('Complete forecast seed roster required')
    lab = run.old.prior.arrays(('target_eval', 'valid')); valid = lab['valid']; y = lab['target_eval']-d['origin'][:, None]
    sites, rec, q = d['sites'], d['recordings'], d['queries']; cuts = run.cuts_by_source(); records = []
    pairs = {f'{k}_vs_cv': (p.CV, k, p.CV) for k in model.ARMS}
    pairs.update(factorized_vs_direct=('direct', 'factorized', p.CV), factorized_vs_ridge=('ridge_risk','factorized',p.CV))
    raw_scores = {(r['seed'], r['recording']): r['metrics']['filter'] for r in
                  run.read(nested.base.PUBLIC/'analysis.json')['records']}
    for f in freeze['cells']:
        seed = f['seed']; floor, paths = run.candidates(d, refs, seed)
        with np.load(run.checked(f['decisions']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], d['global_ids']); ch, pred = z['choices'], z['predicted_harm_on_raw']
        for g in d['groups']:
            n = (ch[:, g] >= 0).sum(1)
            if not (n == n[0]).all():
                raise ValueError('Count mismatch at an original query')
        for r in np.unique(rec):
            ix = np.flatnonzero(rec == r); s = str(sites[ix[0]]); cut = cuts[s]; cv = floor[ix]
            pred_paths = {k: run.model.selected_paths(c[ix], paths[ix], cv) for k, c in zip(model.ARMS, ch)}
            result = p.readout.score_recording({p.CV:cv, **pred_paths}, pairs, y[ix], valid[ix], q[ix], str(r),
                                                easy_cut=cut['easy'], hard_cut=cut['hard'])
            if result['metrics']['raw'] != raw_scores[seed, str(r)]:
                raise ValueError('Original filter-first scores must reproduce')
            ce, _ = nested.cal.errors(cv, y[ix], valid[ix]); known = valid[ix].any(1)
            easy = known & (ce > 0.) & (ce <= cut['easy']); weights = 1./d['crowd'][ix]
            actual = {}; raw_truth = None
            for k, x in pred_paths.items():
                er, _ = nested.cal.errors(x, y[ix], valid[ix])
                h = np.where(easy, np.maximum(er-ce, 0.), 0.)
                actual[k] = float(weights @ h)
                if k == 'raw':
                    raw_truth = h
            use = ch[0, ix] >= 0
            radius = np.linalg.norm(pred_paths['raw']-cv, axis=-1).max(1)
            true_fraction = raw_truth[use]/radius[use]
            mse = {k: float(weights[use] @ np.square(pred[a, ix[use]]/radius[use]-true_fraction)) for a,k in enumerate(model.ARMS)}
            records.append(dict(seed=seed, source=s, known_CV_mass=float(weights[known] @ ce[known]),
                weighted_easy_positive_harm=actual, weighted_selected_harm_fraction_sqerror=mse,
                weighted_raw_selection=float(weights[use].sum()), **result))
        beat(state='scored_seed', seed=seed, views=len(records))
    if len(records) != 102:
        raise ValueError('All repeated recording views required')
    names = np.unique(sites); draw = np.random.default_rng(73107).integers(len(names), size=(3000, len(names)))
    grouped = [[r for r in records if r['source']==s and r['seed']==seed] for s in names for seed in run.kin.prior.SEEDS]
    def ci(values):
        if any(v is None for v in values):
            return None
        v = np.asarray(values).reshape(len(names), 3).mean(1)
        return dict(point=float(v.mean()), interval95=np.quantile(v[draw].mean(1), [.025, .975]).tolist())
    comparisons = {}
    for k in pairs:
        t = [p.readout.merge_comparisons([r['comparisons'][k] for r in g]) for g in grouped]
        comparisons[k] = {key: ci([m[key] for m in t]) for key in ('known_pp', 'completion_lower_pp')}
    risk = {}
    for k in model.ARMS:
        views = [r['metrics'][k]['risk'] for r in records]
        risk[k] = dict(passes=sum(x['supported'] for x in views), switches=sum(x['selected'] for x in views),
            easy_over_2=sum(x['completion_easy_risk_upper'] is not None and x['completion_easy_risk_upper'] > .02 for x in views),
            undefined_easy=sum(x['completion_easy_risk_upper'] is None for x in views))
    errors = {k: sum(r['weighted_selected_harm_fraction_sqerror'][k] for r in records) /
              sum(r['weighted_raw_selection'] for r in records) for k in model.ARMS}
    harm = {k: ci([100*sum(r['weighted_easy_positive_harm']['direct']-r['weighted_easy_positive_harm'][k] for r in g)
                   /sum(r['known_CV_mass'] for r in g) for g in grouped]) for k in model.ARMS}
    run.once(PUBLIC/'analysis.json', run.stable(dict(comparisons=comparisons, risk=risk,
        harm_reduction_vs_direct_pp=harm, fixed_raw_selected_mse=errors, records=records, matched_query_counts=True,
        original_rules_unchanged=True, independent_confirmation=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register','prefix','fit','score','replay'))
    phase = parser.parse_args().phase; register()
    if phase == 'register':
        return
    for f in (*FILES, REG):
        if subprocess.check_output(['git','show','HEAD:'+str(f.relative_to(ROOT))], cwd=ROOT) != f.read_bytes():
            raise ValueError('Commit the experiment before fitting')
    PRIVATE.mkdir(parents=True, exist_ok=True); start = time.monotonic()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def beat(**v):
            if shutil.disk_usage(ROOT).free < 2*1024**3 or time.monotonic()-start > 43200:
                raise OSError('Resume from the atomic head checkpoint')
            event = dict(pid=os.getpid(), phase=phase, seconds=time.monotonic()-start, **v)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', event); print(event, flush=True)
        beat(state='start'); d = run.population(); refs = run.register()
        if phase in ('fit','prefix'):
            result = fit(d, refs, beat, phase == 'prefix')
        else:
            score(d, refs, beat); result = dict(artifact=run.artifact(PUBLIC/'analysis.json'))
        run.once(PUBLIC/f'{phase}_execution.json', dict(pid=os.getpid(), seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **result))


if __name__ == '__main__':
    main()
