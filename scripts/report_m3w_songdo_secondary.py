"""Build a secondary Songdo report without reading archives or future rows."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import m3w_songdo_protocol as primary
from src.evaluation import m3w_songdo_secondary as secondary


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def read_finished(run):
    registration, registration_hash = load(run/'registration.json')
    freeze, freeze_hash = load(run/'freeze.json')
    scored, scored_hash = load(run/'confirmation.json')
    if (freeze['registration_sha256'] != registration_hash or scored['registration_sha256'] != registration_hash
            or scored['release_freeze_sha256'] != freeze_hash or scored['role'] != 'confirmation'):
        raise ValueError('Completed scores must bind their original registration and release freeze')
    mode = primary.matched_only(registration['spec'])
    if mode:
        if freeze.get('evaluation') != 'matched_only' or 'thresholds' in freeze:
            raise ValueError('Matched-only run cannot carry release thresholds')
    else:
        primary.check_thresholds(freeze['thresholds'])
    cuts = {k: scored[k] for k in ('easy_cut', 'hard_cut', 'calibration_cutoffs_sha256') if k in scored}
    if any(freeze[k] != v for k, v in cuts.items()):
        raise ValueError('Native cutoffs differ from the frozen values')
    roles = registration['flight_roles']
    components = {n: g for n, (role, g) in roles.items() if role == 'confirmation'}
    if any(g in set(components.values()) for role, g in roles.values() if role == 'calibration'):
        raise ValueError('Linked sources overlap calibration and confirmation')
    records, checksums = {}, {}
    if any(Path(n).name != n for n in components) or len({Path(n).stem for n in components}) != len(components):
        raise ValueError('Unsafe or ambiguous registered flight name')
    for n in sorted(components):
        r, h = load(run/'confirmation'/(Path(n).stem+'.json'))
        if any(r[k] != v for k, v in dict(flight=n, registration_sha256=registration_hash,
                release_freeze_sha256=freeze_hash, role='confirmation', **cuts).items()):
            raise ValueError('A flight score belongs to another run')
        records[n], checksums[n] = r['scores'], h
    core = primary.summarize(records, components, only=mode)
    if any(core[k] != scored[k] for k in core):
        raise ValueError('Per-flight scores do not reproduce the primary report')
    if not mode:
        for v in records.values():
            for s, r in v.items():
                for a in primary.ARMS:
                    t = freeze['thresholds'][s][a]
                    expected = a in r['scores'] and t is not None and (t == 'all' or r['scores'][a] >= t)
                    if r['released'][a]['edited'] != expected:
                        raise ValueError('Release does not match the frozen threshold')
    report = secondary.summarize(records, components, matched_only=mode)
    return dict(**report, source_kind='score_receipt_replay', primary_unchanged=True,
                original_2pct_unchanged=True, no_new_label_access=True,
                binding=dict(registration_sha256=registration_hash, release_freeze_sha256=freeze_hash,
                             primary_score_sha256=scored_hash, flight_score_sha256=checksums))


def write(path, result):
    blob = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if path.exists() and path.read_bytes() != blob:
        raise FileExistsError('Refusing to replace a different report')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('xb') as f:
            f.write(blob)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    path = args.output or args.run/'secondary_report.json'
    r = read_finished(args.run)
    write(path, r)
    print(json.dumps({k: r[k] for k in ('flights', 'components', 'flight_seed_views', 'matched_seed_views')}, indent=2))
