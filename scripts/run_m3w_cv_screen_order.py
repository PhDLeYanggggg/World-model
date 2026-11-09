"""Compare screening order and OLS4 addback with frozen admission heads."""
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_cv_admission as base

prior, model = base.prior, base.model
np, torch, read, sha, once, stable, artifact, checked = (
    base.np, base.torch, base.read, base.sha, base.once, base.stable, base.artifact, base.checked)
OUT = base.OUT.parent / 'moving_cv_screen_order_v1'
PRIVATE = base.PRIVATE.parent / OUT.name
RELAXED = (.03, .10)
ARMS = ('before', 'gain_match', 'addback_match', 'screen_match', 'joint_match',
        'screen_full', 'calibrated')


@torch.inference_mode()
def predictions(net, cv, bank, history):
    past, _ = prior.study.motion.features(history, 'prefix')
    scale = prior.model.scale(history)
    n, k = bank.shape[:2]
    gain, risk, easy, legal = np.empty((n,k)), np.empty((n,k,2)), np.empty((n,k)), np.empty((n,k), bool)
    for j in range(k):
        for start in range(0, n, 4096):
            sl = slice(start, start+4096)
            delta = bank[sl,j]-cv[sl]
            radius = np.linalg.norm(delta, axis=-1).max(-1)
            x = model.features(past[sl], delta, scale[sl])
            value = [v.numpy() for v in model.decode(net(torch.from_numpy(x)))]
            gain[sl,j], risk[sl,j], easy[sl,j] = model.predicted(*value, radius/scale[sl])
            legal[sl,j] = radius > 1e-8
    return gain, risk, easy, legal


def nominate(pred, gate=None, keep=None):
    gain, risk, easy, legal = pred
    if gate is not None:
        legal = model.admissible(gain, risk, easy, legal, *gate)
    if keep is not None:
        legal = legal & keep
    top = np.argmax(np.where(legal, gain, -np.inf), axis=1)
    rows = np.arange(len(top))
    return np.where(legal[rows,top], top, -1).astype(np.int8), gain[rows,top], risk[rows,top], easy[rows,top]


def choose(nom, groups, match=None, pair=None):
    action, gain, _, _ = nom
    out = np.full_like(action, -1)
    for i, group in enumerate(groups):
        cap = base.CAP if match is None else int((match[group] >= 0).sum())
        eligible = action[group] >= 0
        if match is not None and eligible.sum() < cap:
            raise ValueError('Matched query has insufficient eligible edits')
        terms = (None,None) if pair is None else (pair[0][i], pair[2][pair[1][i]:pair[1][i+1]])
        rows = group[model.select(gain[group], eligible, cap, *terms)]
        out[rows] = action[rows]
    return out


def calibrate(pred, d, cv, ols, bank, cut):
    groups = prior.query_groups(d['query'])
    ladder = []
    for t, p in base.GRID:
        action = choose(nominate(pred, (t,p)), groups)
        rows = base.metrics(cv, ols, bank, d, {'strict':action}, cut)
        gains = [[prior.prior.parent.readout.merge_comparisons(
            [r['comparisons'][f'strict_vs_{k}'] for r in rows if r['source'] == s]
        )['completion_lower_pp'] for k in ('cv','ols4')] for s in sorted(set(d['sites']))]
        ladder.append(dict(tolerance=t, min_easy=p, edits=int((action >= 0).sum()),
            passes=sum(r['metrics']['strict']['risk']['supported'] for r in rows),
            gain_pp=np.mean(gains, axis=0).tolist()))
    return ladder, model.choose_setting(ladder, len(np.unique(d['recording'])))


def load_head(fit, width, output):
    c = torch.load(checked(fit['artifact']), map_location='cpu', weights_only=True)
    if c['identity'] != fit['identity'] or c['step'] != base.STEPS:
        raise ValueError('Frozen training identity changed')
    net = prior.model.head(width, output)
    net.load_state_dict(c['net'])
    return net.eval()


def freeze_cell(cell, d, donor):
    ident = cell['identity']
    roles = [set(ident[k]) for k in ('train','calibration','held')]
    if len(set.union(*roles)) != 9 or any(len(s) != 3 for s in roles):
        raise ValueError('Expected disjoint fitting-source roles')
    for fit, head in zip(cell['fits'], ('risk','pair'), strict=True):
        if fit['identity'] != dict(ident, head=head):
            raise ValueError('Training role changed')
    fold, seed = ident['fold'], ident['seed']
    _, cal = base.subset(d, ident['calibration'])
    hi, held = base.subset(d, ident['held'])
    width = prior.study.motion.features(cal['history'][:1], 'prefix')[0].shape[1]+25
    risk, pair = [load_head(f, w, o) for f,w,o in zip(cell['fits'], (width,22), (8,1), strict=True)]
    c = prior.study.motion.bank(cal['history'])
    bank = base.expanded(c[:,0], c[:,1:])
    ladder, selected = calibrate(predictions(risk,c[:,0],bank,cal['history']), cal,
                                 c[:,0], c[:,1], bank, cell['cut'])
    cv, _, original = prior.bank(held['history'], {k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
    bank = base.expanded(cv, original)
    ols4 = (len(base.FRACTIONS)-1)*original.shape[1]
    pred = predictions(risk,cv,bank,held['history'])
    best = nominate(pred)
    groups = prior.query_groups(held['query'])
    before = base.choices(best, groups, RELAXED)
    gain_match = choose(best, groups, before)
    with np.load(checked(cell['actions']), allow_pickle=False) as z:
        for actual, key in ((held['ids'],'ids'), (before,'relaxed'), (gain_match,'relaxed_gain_match'),
                            (choose(best,groups),'forced')):
            np.testing.assert_array_equal(actual,z[key])
    keep = (np.arange(bank.shape[1])[None,:] == best[0][:,None])
    keep[:,ols4] = True
    screen = nominate(pred, RELAXED)
    addback = nominate(pred, RELAXED, keep)
    delta = prior.apply(cv, bank, screen[0])-cv
    edges = base.pair_estimates(pair, held['history'], cv, delta, groups)
    actions = dict(before=before, gain_match=gain_match, addback_match=choose(addback,groups,before),
        screen_match=choose(screen,groups,before), joint_match=choose(screen,groups,before,edges),
        screen_full=choose(screen,groups),
        calibrated=(np.full_like(before,-1) if selected is None else choose(nominate(pred,selected),groups)))
    for name in ARMS[:5]:
        for g in groups:
            if (actions[name][g] >= 0).sum() != (before[g] >= 0).sum():
                raise ValueError('Unequal edit count within a matched query')
    path = prior.study.parent.parent.save_actions(PRIVATE/f'choice_{fold}_{seed}.npz', dict(ids=held['ids'],**actions))
    return dict(identity=ident, cut=cell['cut'], actions=path, setting=selected, ladder=ladder,
        calibration_records=len(np.unique(cal['recording'])),
        ols4_choices={k:int((v == ols4).sum()) for k,v in actions.items()},
        action_changes={k:int((v != before).sum()) for k,v in actions.items()},
        legal_screened_rows=int((screen[0] >= 0).sum()))


def score_recordings(cv, ols, bank, d, actions, cut, seed):
    scorer = prior.prior.parent.readout
    pairs = {f'{arm}_vs_{label}': (key,arm) for arm in ARMS for label,key in
             (('cv',base.CV),('ols4',base.OLS))}
    pairs.update({f'{arm}_vs_before': ('before',arm) for arm in ARMS[1:5]})
    pairs.update(screen_vs_gain=('gain_match','screen_match'), joint_vs_screen=('screen_match','joint_match'),
                 screen_vs_addback=('addback_match','screen_match'))
    for rec in np.unique(d['recording']):
        ix = d['recording'] == rec
        q, y, valid = d['query'][ix], d['target'][ix], d['valid'][ix]
        c = cv[ix]
        paths = {base.CV:c,base.OLS:ols[ix],**{k:prior.apply(c,bank[ix],v[ix]) for k,v in actions.items()}}
        r = scorer.score_recording(paths,{},y,valid,q,str(rec),easy_cut=cut[0],hard_cut=cut[1])
        _, inv, n = np.unique(q,return_inverse=True,return_counts=True)
        comparisons = {k:prior.study.motion.full_masses(paths[a],paths[b],c,y,valid,1./n[inv])
                       for k,(a,b) in pairs.items()}
        source, = np.unique(d['sites'][ix])
        yield dict(r, source=str(source), seed=seed, comparisons=comparisons)


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

        def beat(**v):
            info = dict(pid=os.getpid(),seconds=round(time.monotonic()-start,3),**v)
            prior.donor.prior.prior.source.atomic(PRIVATE/'heartbeat.json',info)
            if 'step' not in v:
                print(info,flush=True)

        reg, donor, roles = base.register()
        parent = base.OUT/'freeze.json'
        old = read(parent)
        if old['registration'] != reg or not old['before_held_scoring'] or len(old['cells']) != 9:
            raise ValueError('Missing source-excluded parent freeze')
        registration = OUT/'registration.json'
        once(registration,stable(dict(parent=artifact(parent),files=[artifact(Path(__file__)),artifact(OUT/'protocol.md')],
            grid=base.GRID, relaxed=RELAXED, cap=base.CAP, a_labels_opened=False,
            independent_confirmation=False, original_2pct=True)))
        d = base.fit_population(beat)
        freeze = OUT/'freeze.json'
        if not freeze.exists():
            cells = []
            for cell in old['cells']:
                ident = cell['identity']
                if any(ident[k] != roles[ident['fold']][k] for k in ('train','calibration','held')):
                    raise ValueError('Parent role assignment changed')
                cells.append(freeze_cell(cell,d,donor))
                beat(phase='choice_frozen', fold=ident['fold'], seed=ident['seed'])
            once(freeze,stable(dict(registration=sha(registration),cells=cells,before_held_scoring=True)))
        bundle = read(freeze)
        if bundle['registration'] != sha(registration) or len(bundle['cells']) != 9 or not bundle['before_held_scoring']:
            raise ValueError('All choices must be frozen before scoring')
        rows = []
        for cell in bundle['cells']:
            ident = cell['identity']
            hi, held = base.subset(d, ident['held'])
            seed = ident['seed']
            cv, ols, paths = prior.bank(held['history'],{k:v[hi] for k,v in prior.read_donor(donor,seed,d).items()})
            with np.load(checked(cell['actions']), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'],held['ids'])
                rows.extend(score_recordings(cv,ols,base.expanded(cv,paths),held,{k:z[k] for k in ARMS},cell['cut'],seed))
            beat(phase='held_scored', fold=ident['fold'], seed=seed)
        summary = prior.prefix.grouped.grouped_readout(rows,{r['recording']:r['source'] for r in rows},base.SEEDS)
        if (summary['views'],summary['source_groups'],summary['risk']['before']['switches']) != (72,9,2317):
            raise ValueError('Expected the exact parent cohort and matched count')
        result = dict(freeze=sha(freeze),summary=summary,qualified=sum(c['setting'] is not None for c in bundle['cells']),
            calibration_max_passes=[max(r['passes'] for r in c['ladder']) for c in bundle['cells']],
            ols4_choices={k:sum(c['ols4_choices'][k] for c in bundle['cells']) for k in ARMS},
            action_changes={k:sum(c['action_changes'][k] for c in bundle['cells']) for k in ARMS},
            optimizer_updates=0, original_2pct=True, a_labels_opened=False, independent_confirmation=False)
        once(OUT/'readout.json',stable(result))
        beat(phase='complete')


if __name__ == '__main__':
    main()
