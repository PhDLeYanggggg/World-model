"""Export anonymous recording masses for the cost-versus-order comparison."""

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_review_replay as pack
from scripts import run_m3w_motion_choice_cost as study


def build(out):
    readout = study.OUT / 'readout.json'
    score = study.read(readout)
    local = study.read(study.PRIVATE / 'recording_scores.json')
    if (local['readout'] != study.sha(readout) or score['independent_confirmation']
            or score['registration'] != study.sha(study.OUT / 'registration.json')
            or score['choice_freeze'] != study.sha(study.OUT / 'freeze.json')):
        raise ValueError('Require bound development-only results')
    roster = {r['recording']: r['source'] for r in local['rows']}
    sources = {k: f'S{i:02d}' for i, k in enumerate(sorted(set(roster.values())))}
    recordings = {k: f'V{i:02d}' for i, k in enumerate(sorted(roster))}
    seeds = {k: i for i, k in enumerate(study.SEEDS, 1)}
    packet = dict(schema_version=1, seeds=list(seeds.values()),
                  roster={recordings[k]: sources[v] for k, v in roster.items()},
                  recording_studies={'motion_choice_cost': pack.trim(local['rows'], recordings, sources, seeds)},
                  joint_cells={k: {str(s): {} for s in seeds.values()} for k in sources.values()})
    expected = dict(motion_choice_cost={**score['summary'], 'seeds': list(seeds.values())}, joint_pair_contrasts={})
    if pack.replay.recompute(packet) != expected:
        raise ValueError('Anonymous masses must exactly reproduce the grouped result')
    files = {name: (pack.ROOT / name).read_bytes() for name in pack.FILES}
    files.update({'README.md': b'Run python replay.py --show. Anonymous scalar replay, not training or raw data.\n',
                  'replay.py': Path(pack.replay.__file__).read_bytes(),
                  'requirements.txt': f'numpy=={study.np.__version__}\n'.encode(),
                  'statistics.json': pack.raw_json(packet), 'expected.json': pack.raw_json(expected)})
    files['manifest.json'] = pack.raw_json({k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    pack.write_zip(out, files)
    print(study.sha(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=study.PRIVATE / 'anonymous_motion_choice_cost.zip')
    build(p.parse_args().out)
