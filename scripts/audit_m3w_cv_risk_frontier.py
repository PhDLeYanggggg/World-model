"""Compare a frozen causal menu expansion with full-row hindsight capacity."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_joint_count_capacity as original
from src.evaluation import m3w_cv_joint_feasibility as full
from src.world_model import m3w_cv_risk_frontier as model

study = original.study
old, base, prior, np, torch = study.old, study.base, study.prior, study.np, study.torch
OUT = study.OUT.parent / 'moving_cv_risk_frontier_v1'
PRIVATE = study.PRIVATE.parent / OUT.name
MODES = ('original', 'expanded', 'full_rows')
SECONDS = 2.


def prepare(d, shape):
    h = d['history']
    cv, ols, bank = study.parent.analytic(h)
    g, r, legal, ref = study.parent.predictions({'shape': shape}, d, cv, ols, bank)['shape']
    groups = prior.query_groups(d['query'])
    anchor = study.parent.model.portfolio(g, r, legal, ref, groups, .50, 0.)
    scale = prior.model.scale(h)
    packet = []
    k = legal.shape[1]
    amplitudes = np.asarray(study.parent.model.AMPLITUDES)
    j, a = np.divmod(np.arange(k), len(amplitudes))
    paths = cv[:, None] + amplitudes[a][None, :, None, None] * (bank[:, j] - cv[:, None])
    agreement = np.linalg.norm(cv - ols, axis=-1).mean(1)[:, None] - np.linalg.norm(paths - ols[:, None], axis=-1).mean(2)
    for at in groups:
        before, _ = study.model.menu(g[at], r[at], legal[at], agreement[at], anchor[at])
        after = model.generate(g[at], r[at], legal[at], ref[at], float(scale[at].mean()))
        merged = {x.tobytes(): x for x in before}
        for x in after:
            merged.setdefault(x.tobytes(), x)
        packet.append(dict(at=at, plans=np.stack(list(merged.values())), original=len(before)))
    return cv, ols, bank, paths, packet


def slate_check(d, cv, ols, bank, p, cut):
    blocks, plans, roster, ref, offset, cap = original.record_problem(d, cv, ols, bank, p, cut[0])
    old_n = [x['original'] for x in p]
    out, prior_proof = {}, None
    for mode, counts in (('original', old_n), ('expanded', [len(x) for x in blocks])):
        b, z = [[x[:n] for x, n in zip(v, counts, strict=True)] for v in (blocks, plans)]
        row, chosen = original.search(b, ref, offset, cap)
        proof = original.check(d, cv, ols, bank, roster, z, chosen, cut) if chosen is not None else None
        if proof is not None:
            row.update(status='verified', witness_pp=proof)
            if mode == 'original':
                prior_proof = proof
        if mode == 'expanded' and prior_proof is not None:
            if row['status'] in ('excluded', 'unsupported', 'lp_infeasible'):
                raise ValueError('Expanded menu rejected an existing witness')
            if proof is None:
                row.update(status='verified', witness_pp=prior_proof, inherited_witness=True)
        out[mode] = row
    return out


def row_check(d, at, cv, ols, bank, paths, cut):
    c, o = cv[at], ols[at]
    y, v, q = (d[k][at] for k in ('target', 'valid', 'query'))
    design = full.coefficients(c, o, paths[at], y, v, q, cut[0])
    z, log = full.solve(design, int(design['counts'].max()), seconds=SECONDS)
    row = dict(status=log['status'], upper=log['upper_pp'])
    if z is None:
        return row
    actions = np.full(len(at), -1, np.int16)
    actions[design['row'][z]] = design['arm'][z]
    if len(np.unique(design['row'][z])) != int(z.sum()):
        raise ValueError('Full-bank solution repeated an agent')
    proof = original.check(d, cv, ols, bank, [at], [actions[None]], [0], cut)
    if proof is not None:
        row.update(status='verified', witness_pp=proof)
    return row


def summarize(rows):
    sites = sorted({r['source'] for r in rows})
    if len(rows) != 72 or len(sites) != 9:
        raise ValueError('Incomplete exposed development roster')
    summary = {}
    for k in MODES:
        categories = sorted({r[k]['status'] for r in rows})
        summary[k] = {a: sum(r[k]['status'] == a for r in rows) for a in categories}
    a = np.asarray([[r[k]['status'] == 'verified' for k in MODES] for r in rows], float)
    site_rows = np.asarray([r['source'] for r in rows])
    m = np.array([a[site_rows == s].mean(0) for s in sites])
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    effects = {}
    for i, j in ((0, 1), (1, 2)):
        delta = 100. * (m[:, j] - m[:, i])
        effects[f'{MODES[j]}_minus_{MODES[i]}'] = dict(point=float(delta.mean()),
            interval95=np.quantile(delta[draw].mean(1), [.025, .975]).tolist())
    return summary, effects


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay', action='store_true', help='recompute all cells and check exact receipts')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            prior.donor.prior.prior.source.atomic(PRIVATE / 'heartbeat.json', z)
            print(z, flush=True)

        refs = [study.OUT / k for k in ('registration.json', 'freeze.json', 'readout.json', 'capacity.json')]
        reg, freeze, result, capacity = map(old.read, refs)
        if (freeze['registration'] != old.sha(refs[0]) or result['freeze'] != old.sha(refs[1])
                or capacity['parent']['sha256'] != old.sha(refs[2]) or not freeze['before_held_scoring']
                or not reg['analytic_only'] or any(x['a_labels_opened'] or x['independent_confirmation']
                    or not x['original_2pct'] for x in (result, capacity))):
            raise ValueError('Require original development-only result')
        shape = old.read(old.checked(reg['parents'][1]))
        rp = OUT / 'registration.json'
        old.once(rp, old.stable(dict(parents=[old.artifact(p) for p in refs],
            files=[old.artifact(p) for p in (Path(__file__), Path(model.__file__), OUT / 'protocol.md')],
            solver_seconds=SECONDS, grid=model.GRID, original_2pct=True)))
        base.register()
        data = base.fit_population(lambda **_: None)
        rows, stats, receipts = [], [], []
        for cell, parent in zip(freeze['cells'], shape['cells'], strict=True):
            ident, cut = cell['identity'], cell['cut']
            if parent['identity'] != ident:
                raise ValueError('Frozen shape fit has a different role')
            stem = f'{ident["fold"]}_{ident["seed"]}'
            cp = PRIVATE / f'cell_{stem}.json'
            binding = dict(registration=old.sha(rp), identity=ident)
            if cp.exists() and not args.replay:
                item = old.read(cp)
                if item['binding'] != binding:
                    raise ValueError('Resume cell binding changed')
            else:
                beat(phase='generate', cell=stem)
                _, h = base.subset(data, ident['held'])
                past, _ = study.parent.parent.context.context(h)
                net = study.loader.load(parent['fits'][1], past.shape[1]+37)
                cv, ols, bank, paths, packet = prepare(h, net)
                n = sum(x['original'] for x in packet)
                if n != cell['held_options']:
                    raise ValueError('Original menu size did not reproduce')
                flat = np.concatenate([p['plans'].ravel() for p in packet])
                choices = prior.study.parent.parent.save_actions(PRIVATE / f'menu_{stem}.npz',
                    dict(ids=h['ids'], plans=flat, lengths=np.array([len(p['plans']) for p in packet]),
                         original=np.array([p['original'] for p in packet])))
                new = sum(len(p['plans']) for p in packet)
                beat(phase='menu_frozen', cell=stem, original=n, expanded=new)
                rr = []
                for rec in np.unique(h['recording']):
                    at = np.flatnonzero(h['recording'] == rec)
                    p = [x for x in packet if h['recording'][x['at'][0]] == rec]
                    row = slate_check(h, cv, ols, bank, p, cut)
                    row['full_rows'] = row_check(h, at, cv, ols, bank, paths, cut)
                    if row['expanded']['status'] == 'verified':
                        if row['full_rows']['status'] in ('undefined_support', 'nonpositive_bound', 'risk_infeasible'):
                            raise ValueError('Full bank excluded an available menu witness')
                        if row['full_rows']['status'] != 'verified':
                            row['full_rows'].update(status='verified', witness_pp=row['expanded']['witness_pp'],
                                inherited_witness=True)
                    source, = np.unique(h['sites'][at])
                    rr.append(dict(row, source=str(source)))
                    beat(phase='record_scored', cell=stem, records=len(rr))
                item = dict(binding=binding, choices=choices, rows=rr, stats=dict(queries=len(packet),
                    original=n, expanded=new, changed=sum(len(x['plans']) > x['original'] for x in packet)))
                old.once(cp, old.stable(item))
            old.checked(item['choices'])
            rows.extend(item['rows'])
            stats.append(item['stats'])
            receipts.append(old.artifact(cp))
        summary, effects = summarize(rows)
        if summary['original'].get('verified', 0) != capacity['summary']['variable_count']['verified']:
            raise ValueError('Prior verified witness count changed')
        stats = {k: sum(x[k] for x in stats) for k in stats[0]}
        old.once(OUT / 'readout.json', old.stable(dict(registration=old.sha(rp), cells=receipts,
            summary=summary, effects=effects, stats=stats, views=len(rows), labels_hindsight_only=True,
            a_labels_opened=False, independent_confirmation=False, original_2pct=True)))
        beat(phase='complete', summary=summary, effects=effects, stats=stats)


if __name__ == '__main__':
    main()
