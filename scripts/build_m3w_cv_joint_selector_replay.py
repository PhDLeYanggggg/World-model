"""Export a NumPy-only, anonymous replay of agent and pair statistics."""

import argparse
import hashlib
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import run_m3w_cv_joint_selector as study


def build(out):
    result = study.read(study.OUT / 'readout.json')
    raw = study.read(study.PRIVATE / 'recording_scores.json')
    if (result['freeze'] != study.sha(study.OUT / 'freeze.json') or result['independent_confirmation']
            or raw['readout'] != study.sha(study.OUT / 'readout.json')):
        raise ValueError('Require checked development-only selector readout')
    rows = raw['rows']
    roster = {r['recording']: r['source'] for r in rows}
    src = {s: f'S{i:02d}' for i, s in enumerate(sorted(set(roster.values())))}
    rec = {s: f'R{i:02d}' for i, s in enumerate(sorted(roster))}
    seeds = {k: i for i, k in enumerate(study.SEEDS, 1)}
    alias = {rec[k]: src[v] for k, v in roster.items()}
    payload = dict(schema_version=1, seeds=list(seeds.values()), roster=alias,
                   recording_studies={'selector': pack.trim(rows, rec, src, seeds)},
                   joint_cells={k: {str(s): {} for s in seeds.values()} for k in src.values()})
    edge = [dict(source=src[r['source']], recording=rec[r['recording']], seed=seeds[r['seed']],
                 geometry=r['geometry']) for r in rows]
    expected_agent = {**result['agent'], 'seeds': list(seeds.values())}
    if pack.replay.recompute(payload)['selector'] != expected_agent:
        raise ValueError('Anonymous agent stats do not match')
    script = Path(pack.replay.__file__).read_text().rsplit("if __name__ == '__main__':", 1)[0]
    # Use the normal manifest checker but reconstruct both sets of summaries.
    script = script.replace('from pathlib import Path', 'from pathlib import Path\nfrom types import SimpleNamespace', 1)
    script += (f'\nSEEDS = {tuple(seeds.values())!r}\nGEOM = {study.GEOM!r}\n'
               'geometry = SimpleNamespace(bootstrap=bootstrap)\n\n' + inspect.getsource(study.pair_summary) + '\n'
               'root = Path(__file__).resolve().parent\nverify_manifest(root)\n'
               'value = dict(agent=recompute(read(root/"statistics.json"))["selector"], '
               'pair=pair_summary(read(root/"pair_statistics.json")))\n'
               'assert value == read(root/"expected.json")\nprint(json.dumps(value, sort_keys=True))\n')
    files = {name: (pack.ROOT/name).read_bytes() for name in pack.FILES}
    files.update({'README.md': b'Run python replay.py with NumPy. Anonymous statistics only, no model fitting.\n',
        'replay.py': script.encode(), 'statistics.json': pack.raw_json(payload), 'pair_statistics.json': pack.raw_json(edge),
        'expected.json': pack.raw_json(dict(agent=expected_agent, pair=result['pair'])),
        'requirements.txt': f'numpy=={study.np.__version__}\n'.encode()})
    files['manifest.json'] = pack.raw_json({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    pack.write_zip(out, files)
    print(study.sha(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=study.PRIVATE / 'anonymous_cv_joint_selector.zip')
    build(p.parse_args().out)
