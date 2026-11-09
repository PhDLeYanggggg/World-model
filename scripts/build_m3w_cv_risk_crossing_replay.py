"""Export anonymous per-recording sufficient statistics for v143."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import train_m3w_cv_risk_crossing as study

old, base, prior, np = study.old, study.base, study.prior, study.np


def collect(cells, data):
    score = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    pairs = {f'{a}_vs_{k}': (b, a) for a in study.ARMS for k, b in (('cv', base.CV), ('ols4', base.OLS))}
    pairs.update(study.PRIMARY)
    rows = []
    for c in cells:
        ident = c['identity']
        _, h = base.subset(data, ident['held'])
        cv, ols, bank = study.study.parent.analytic(h['history'])
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            paths = {base.CV: cv, base.OLS: ols,
                **{k: study.study.parent.model.apply(cv, bank, z[k]) for k in study.ARMS}}
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q = (h[k][at] for k in ('target', 'valid', 'query'))
            s, = np.unique(h['sites'][at])
            z = {k: p[at] for k, p in paths.items()}
            r = score.score_recording(z, {}, y, v, q, str(rec), easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in pairs.items()}
            rows.append(dict(r, source=str(s), seed=ident['seed'], comparisons=terms))
    return rows


def risk_components(rows):
    out = {}
    for k in study.ARMS:
        rs = [r['metrics'][k]['risk'] for r in rows]
        known = [all(x[f] is not None and x[f] <= .02
            for f in ('known_all_risk', 'known_easy_risk', 'known_easy_net_degradation')) for x in rs]
        out[k] = dict(known_only_passes=sum(known),
            lost_after_missing_debit=sum(a and not b['supported'] for a, b in zip(known, rs, strict=True)),
            easy_positive_harm_over=sum(x['completion_easy_risk_upper'] is not None
                and x['completion_easy_risk_upper'] > .02 for x in rs),
            easy_positive_harm_undefined=sum(x['completion_easy_risk_upper'] is None for x in rs),
            wholly_unknown_edits=sum(x['unknown_selected'] for x in rs))
    return out


def build(path):
    f, r = (study.OUT / k for k in ('freeze.json', 'readout.json'))
    freeze, readout = map(old.read, (f, r))
    if (readout['freeze'] != old.sha(f) or not freeze['before_held_scoring'] or readout['a_labels_opened']
            or readout['independent_confirmation'] or not readout['original_2pct']):
        raise ValueError('Require frozen development-only scores')
    base.register()
    data = base.fit_population(lambda **_: None)
    rows = collect(freeze['cells'], data)
    roster = {r['recording']: r['source'] for r in rows}
    if prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS) != readout['summary']:
        raise ValueError('Exact scalar grouping changed')
    rec = {k: f'R{i:02d}' for i, k in enumerate(sorted(roster))}
    src = {k: f'S{i:02d}' for i, k in enumerate(sorted(set(roster.values())))}
    seeds = {k: i for i, k in enumerate(base.SEEDS, 1)}
    packet = dict(schema_version=1, roster={rec[k]: src[v] for k, v in roster.items()},
        seeds=list(seeds.values()), recording_studies={'risk_crossing': pack.trim(rows, rec, src, seeds)},
        joint_cells={k: {str(i): {} for i in seeds.values()} for k in src.values()})
    expected = {'risk_crossing': dict(readout['summary'], seeds=list(seeds.values())), 'joint_pair_contrasts': {}}
    script = Path(pack.replay.__file__).read_text().rsplit("if __name__ == '__main__':", 1)[0]
    if script.count('(34, 14, 3)') != 1:
        raise ValueError('Unexpected anonymous replay template')
    script = script.replace('(34, 14, 3)', '(24, 9, 3)') + '''
root = Path(__file__).resolve().parent
verify_manifest(root)
if recompute(read(root/'statistics.json')) != read(root/'expected.json'):
    raise ValueError('Scalar readout did not replay')
print('Exact anonymous replay: risk crossing, 72 views.')
'''
    files = {k: (pack.ROOT / k).read_bytes() for k in pack.FILES}
    files.update({'README.md': b'Run python replay.py. Scalar risks and utility; not training or pair geometry.\n',
        'replay.py': script.encode(), 'statistics.json': pack.raw_json(packet), 'expected.json': pack.raw_json(expected),
        'requirements.txt': f'numpy=={np.__version__}\n'.encode()})
    files['manifest.json'] = pack.raw_json({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    pack.write_zip(path, files)
    old.once(study.OUT / 'replay_receipt.json', old.stable(dict(parent=old.artifact(r), risk_components=risk_components(rows),
        code=old.artifact(Path(__file__)), anonymous_sha256=old.sha(path), views=len(rows),
        original_2pct=True, independent_confirmation=False)))
    print(old.sha(path), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=study.PRIVATE / 'anonymous_risk_crossing.zip')
    build(p.parse_args().out)
