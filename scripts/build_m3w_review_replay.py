"""Export a self-contained review replay ZIP, not a training or dataset release."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_common_motion as common
from src.reproduction import m3w_review_replay as replay

audit, model = common.audit, common.audit.model
sparse, run, np, h = audit.sparse, common.run, common.np, common.h
PUBLIC = audit.PUBLIC.parent/'review_replay_v1'
DEFAULT = ROOT/'data/stage_cvpr2027_experiments/review_replay_v1/review_replay.zip'
FILES = ('src/evaluation/m3w_grouped_recording_readout.py',
         'src/evaluation/m3w_recording_forecast_comparison.py',
         'src/evaluation/m3w_selection_forecast_readout.py',
         'src/evaluation/m3w_native_metrics.py',
         'src/evaluation/m3w_joint_path_audit.py',
         'src/world_model/m3w_record_joint_actions.py')
RISK = ('rows', 'selected', 'completion_all_risk_upper', 'completion_easy_risk_upper',
        'known_easy_net_degradation')


def raw_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def read_cells(experiment):
    p = experiment.PRIVATE/'freeze.json'
    if run.read(experiment.PUBLIC/'fit_execution.json')['artifact']['sha256'] != run.sha(p):
        raise ValueError('Frozen decisions do not match the fit receipt')
    return {r['seed']: r for r in run.read(p)['cells']}


def expected():
    studies = dict(sparse=sparse, common=common, joint_pair_contrasts=audit)
    out = {}
    for name, study in studies.items():
        p = study.PUBLIC/'analysis.json'
        for phase in ('score', 'replay'):
            r = run.read(study.PUBLIC/f'{phase}_execution.json')
            digest = r.get('analysis_sha256') or r['artifact']['sha256']
            if digest != run.sha(p):
                raise ValueError('Published score and replay disagree')
        value = run.read(p)
        if value['external_labels_opened'] or value['independent_confirmation']:
            raise ValueError('Only exposed development scores can be exported')
        out[name] = value['comparisons' if study is audit else 'grouped']
        if study is not audit:
            out[name]['seeds'] = list(range(1, len(out[name]['seeds'])+1))
    return out


def trim(rows, recordings, sources, seeds):
    return [dict(recording=recordings[r['recording']], source=sources[r['source']],
        seed=seeds[r['seed']], metrics={name: dict(rows=a['rows'],
            p95_known_ADE_pixel=a['p95_known_ADE_pixel'], risk={k: a['risk'][k] for k in RISK})
            for name, a in r['metrics'].items()}, comparisons=r['comparisons']) for r in rows]


def collect():
    d, refs = run.population(), run.register()
    lab = run.old.prior.arrays(('target_eval', 'valid', 'frames'))
    records = h.previous.prior.blocks(d, lab['frames'])
    y, valid = lab['target_eval']-d['origin'][:, None], lab['valid']
    roster = {rec: str(d['sites'][ix[0]]) for rec, (ix, _) in records.items()}
    aliases = {name: f'S{i:02d}' for i, name in enumerate(sorted(set(roster.values())), 1)}
    rec_alias = {rec: f'V{i:03d}' for i, rec in enumerate(sorted(roster), 1)}
    seed_alias = {seed: i for i, seed in enumerate(run.kin.prior.SEEDS, 1)}
    cells, controls, ranks = h.cost.banks()
    sf, df, cf = map(read_cells, (sparse, sparse.parent, common))
    jf = {r['seed']: r for r in run.read(run.PUBLIC/'freeze.json')['seeds']}
    stats = {a: {} for a in aliases.values()}
    studies = dict(sparse=[], common=[])
    pairs = [tuple(audit.ARMS.index(k) for k in p) for p in audit.PAIRS]
    for cell in cells:
        m = h.prune.previous.portfolio.Seed(d, refs, records, cell, controls, ranks)
        seed = m.seed
        sparse_rows = sparse.score(m, records, sf[seed], df[seed], y, valid)
        data = common.design(d, records, m.floor, m.bank, y, valid)
        common_rows = common.score(data, d, records, m.floor, seed, cf[seed], y, valid)
        for name, rows in (('sparse', sparse_rows), ('common', common_rows)):
            studies[name].extend(trim(rows, rec_alias, aliases, seed_alias))
        joint, other = audit.decision_arrays(m, sf[seed], df[seed], jf[seed])
        blocks = {s: [] for s in aliases}
        for rec, (ix, groups) in records.items():
            floor, bank = m.floor[ix], m.bank[ix]
            full = {k: run.model.selected_paths(v[ix], bank, floor) for k, v in joint.items()}
            quarter = sparse.parent.paths(m, ix, {k: v[ix] for k, v in other.items()})
            by_arm = dict(cv=floor, ols4=bank[:, run.NAMES.index(run.parent.OLS)], **full, **quarter)
            paths = np.stack([by_arm[k] for k in audit.ARMS])
            block = []
            for g in groups:
                at = ix[g]
                _, v, _, _ = model.query(d['history'][at], paths[:, g], y[at], valid[at], pairs)
                block.append(v)
            blocks[roster[rec]].append(np.asarray(block))
        for site in aliases:
            values = model.mean_finite(np.concatenate(blocks[site]))
            stats[aliases[site]][str(seed_alias[seed])] = {
                f'{b}_vs_{a}': {name: float(values[i, j]) if np.isfinite(values[i, j]) else None
                               for j, name in enumerate(model.PAIR_FIELDS)}
                for i, (a, b) in enumerate(audit.PAIRS)}
        print(dict(state='aggregate_complete', seed=seed_alias[seed]), flush=True)
        del m
    return dict(schema_version=1, seeds=list(seed_alias.values()),
        roster={rec_alias[r]: aliases[s] for r, s in roster.items()},
        recording_studies=studies, joint_cells=stats)


def write_zip(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending')
    with ZipFile(tmp, 'w', compression=ZIP_DEFLATED, compresslevel=9) as z:
        for name, blob in sorted(files.items()):
            entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            z.writestr(entry, blob, compress_type=ZIP_DEFLATED, compresslevel=9)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT)
    args = p.parse_args()
    packet, target = collect(), expected()
    statistics = raw_json(packet)
    if replay.recompute(packet) != target:
        raise ValueError('Exported statistics do not reproduce the frozen tables')
    files = {name: (ROOT/name).read_bytes() for name in FILES}
    files.update({'replay.py': Path(replay.__file__).read_bytes(),
        'README.md': (PUBLIC/'reviewer_README.md').read_bytes(),
        'requirements.txt': f'numpy=={np.__version__}\n'.encode(),
        'statistics.json': statistics, 'expected.json': raw_json(target)})
    manifest = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    files['manifest.json'] = raw_json(manifest)
    write_zip(args.out, files)
    print(dict(files=len(files), bytes=args.out.stat().st_size,
               sha256=run.sha(args.out)), flush=True)


if __name__ == '__main__':
    main()
