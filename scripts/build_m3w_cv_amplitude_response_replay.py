"""Export the frozen amplitude study's anonymous scalar readout."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import train_m3w_cv_amplitude_response as study


def build(path):
    old, base = study.old, study.base
    f, r = (study.OUT / name for name in ('freeze.json', 'readout.json'))
    freeze, readout = map(old.read, (f, r))
    if (readout['freeze'] != old.sha(f) or not freeze['before_held_scoring']
            or readout['a_labels_opened'] or readout['independent_confirmation']
            or not readout['original_2pct']):
        raise ValueError('Require frozen development-only results')
    _, donor, _ = base.register()
    data = base.fit_population(lambda **_: None)
    rows = study.evaluate(freeze['cells'], data, donor, lambda **_: None, export_rows=True)
    roster = {x['recording']: x['source'] for x in rows}
    grouped = study.prior.prefix.grouped.grouped_readout
    if grouped(rows, roster, base.SEEDS) != readout['summary']:
        raise ValueError('Policy readout did not reproduce')
    rec = {k: f'R{i:02d}' for i, k in enumerate(sorted(roster))}
    src = {k: f'S{i:02d}' for i, k in enumerate(sorted(set(roster.values())))}
    seeds = {k: i for i, k in enumerate(base.SEEDS, 1)}
    packet = dict(schema_version=1, roster={rec[k]: src[v] for k, v in roster.items()},
        seeds=list(seeds.values()), recording_studies={'amplitude': pack.trim(rows, rec, src, seeds)},
        joint_cells={k: {str(i): {} for i in seeds.values()} for k in src.values()})
    expected = {'amplitude': dict(readout['summary'], seeds=list(seeds.values())), 'joint_pair_contrasts': {}}
    script = Path(pack.replay.__file__).read_text().rsplit("if __name__ == '__main__':", 1)[0]
    if script.count('(34, 14, 3)') != 1:
        raise ValueError('Unexpected reviewer replay template')
    script = script.replace('(34, 14, 3)', '(24, 9, 3)') + '''
root = Path(__file__).resolve().parent
verify_manifest(root)
if recompute(read(root/'statistics.json')) != read(root/'expected.json'):
    raise ValueError('Scalar readout did not replay')
print('Exact anonymous replay: amplitude study, 72 views.')
'''
    files = {name: (pack.ROOT / name).read_bytes() for name in pack.FILES}
    files.update({'README.md': b'Run python replay.py. Replays scalar risks and utility, not training or geometry.\n',
        'replay.py': script.encode(), 'statistics.json': pack.raw_json(packet),
        'expected.json': pack.raw_json(expected), 'requirements.txt': f'numpy=={study.np.__version__}\n'.encode()})
    files['manifest.json'] = pack.raw_json({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    pack.write_zip(path, files)
    old.once(study.OUT / 'replay_receipt.json', old.stable(dict(parent=old.artifact(r),
        code=old.artifact(Path(__file__)), anonymous_sha256=old.sha(path), views=72,
        original_2pct=True, independent_confirmation=False)))
    print(old.sha(path), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=study.PRIVATE / 'anonymous_amplitude_response.zip')
    build(p.parse_args().out)
