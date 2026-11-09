"""Diagnose the frozen rank scores without changing release thresholds."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.vehicle_recording_replay import verify

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'outputs/publication_readiness_2026_09/vehicle_recording_replay_v1'
DEFAULT = ROOT/'data/stage_cvpr2027_experiments/submission_admin/vehicle_recording_replay.zip'
STAGES = ('easy_reference', 'risk', 'dual', 'full')


def compare(rows, arm):
    eligible = [(r['arms'][arm]['score'], verify.qualification(r['arms'][arm])[:4]) for r in rows
                if r['arms'][arm]['score'] is not None]
    if not eligible:
        raise ValueError('A direction lacks finite candidate scores')
    result = {}
    for j, stage in enumerate(STAGES):
        labels = [(s, not q[3] if j == 0 else q[j-1]) for s, q in eligible]
        hits, n = verify.top_mass(labels, 5)
        random_rate = sum(v for _, v in labels)/len(labels)
        result[stage] = (hits/n, random_rate)
    return result


def diagnose(data):
    roster = {(s, c, seed) for s in range(7) for c in range(7) if s != c for seed in verify.SEEDS}
    have = [(x['site'], x['calibration'], x['seed']) for x in data['directions']]
    if len(have) != len(roster) or set(have) != roster:
        raise ValueError('Evaluation roster changed')
    report = {}
    for a in verify.FITTED:
        groups = {k: defaultdict(list) for k in STAGES}
        for c in data['directions']:
            for stage, rates in compare(c['recordings'], a).items():
                groups[stage][c['site']].append(rates)
        report[a] = {}
        for k, group in groups.items():
            draws = data['source_resamples']
            top = verify.interval({s: [100*x for x, _ in r] for s, r in group.items()}, draws)
            random = verify.interval({s: [100*y for _, y in r] for s, r in group.items()}, draws)
            lift = verify.interval({s: [100*(x-y) for x, y in r] for s, r in group.items()}, draws)
            report[a][k] = dict(top5_percent=top, same_direction_uniform_percent=random, lift_pp=lift)
    return report


def main(path):
    r = json.loads((REPORT/'receipt.json').read_text())
    if sha256(path.read_bytes()).hexdigest() != r['archive_sha256']:
        raise ValueError('Replay archive is not the verified export')
    with ZipFile(path) as z:
        manifest = json.loads(z.read('manifest.json'))
        raw = z.read('evidence.json')
        if sha256(raw).hexdigest() != manifest['evidence.json']:
            raise ValueError('Scalar payload differs from its member hash')
    result = dict(archive_sha256=r['archive_sha256'], matched_top_k=5, original_2pct_unchanged=True,
                  external_outcomes_read=False, post_readout_diagnostic=True, threshold_changed=False,
                  comparisons=diagnose(json.loads(raw)))
    output = REPORT/'ranking_diagnostic.json'
    blob = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if output.exists() and output.read_text() != blob:
        raise ValueError('Recorded diagnostic differs')
    if not output.exists():
        with output.open('x') as f:
            f.write(blob)
    print(json.dumps({a: {k: v['lift_pp']['mean'] for k, v in arms.items()}
                      for a, arms in result['comparisons'].items()}, indent=2))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--archive', type=Path, default=DEFAULT)
    main(cli.parse_args().archive)
