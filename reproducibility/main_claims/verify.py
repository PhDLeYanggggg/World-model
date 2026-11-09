"""Replay the working paper's headline counts from minimized frozen inputs."""
import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

SEEDS = {17, 29, 43}
EXPECTED = {
    'opendd': {'locations': 7, 'recordings': 202, 'views': 606,
               'hindsight_passes': 330, 'past_only_passes': 57, 'unresolved': 1},
    'moving': {'recordings': 24, 'views': 72, 'ols4_excluded': 67},
    'wepdtof': {'sources': 13, 'recordings': 15, 'views': 45, 'fixed_joint_dual_passes': 0},
}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def summarize(data):
    counts = dict(locations=len(data['opendd']), recordings=0, views=0, hindsight_passes=0,
                  past_only_passes=0, unresolved=0)
    for group in data['opendd']:
        h = group['hindsight_status']
        past = group['past_only']
        views = sum(h.values())
        require({r[0] for r in past} == SEEDS and len(past) == 3 and views % 3 == 0
                and all(r[1] == views // 3 and 0 <= r[2] <= r[1] for r in past)
                and all(type(v) is int and v >= 0 for v in h.values()), 'OpenDD site totals do not align')
        counts['recordings'] += views // 3
        counts['views'] += views
        counts['hindsight_passes'] += h['verified_positive_both']
        counts['past_only_passes'] += sum(r[2] for r in past)
        counts['unresolved'] += h.get('unresolved', 0)

    def seeds(rows, seed_col):
        require({r[seed_col] for r in rows} == SEEDS and len(rows) == 3, 'Expected one row per frozen seed')

    moving = data['moving']
    for rows in moving:
        seeds(rows, 0)
        require(all(isfinite(r[1]) for r in rows), 'Nonfinite Moving upper bound')
    ext = data['wepdtof']
    records, passes = 0, 0
    for source in ext:
        records += len(source)
        for rows in source:
            seeds(rows, 0)
            for _, supported, mass, cv, ols4, recorded in rows:
                require(type(supported) is bool and type(recorded) is bool and isfinite(mass) and mass >= 0,
                        'Invalid external risk or denominator')
                require((mass == 0 and cv is None and ols4 is None)
                        or (mass > 0 and isfinite(cv) and isfinite(ols4)), 'Undefined external utility mismatch')
                derived = supported and mass > 0 and min(cv, ols4) > 0
                require(derived == recorded, 'External dual-pass field does not replay')
                passes += int(derived)
    return dict(opendd=counts, moving=dict(recordings=len(moving), views=sum(map(len, moving)),
                    ols4_excluded=sum(r[1] <= 0 for rows in moving for r in rows)),
                wepdtof=dict(sources=len(ext), recordings=records, views=3*records, fixed_joint_dual_passes=passes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parent
    binding = json.loads((root/'manifest.json').read_text())
    for name in ('README.md', 'verify.py', 'evidence.json'):
        require(sha256((root/name).read_bytes()).hexdigest() == binding['members'][name],
                'Changed replay member: '+name)
    values = summarize(json.loads((root/'evidence.json').read_text()))
    require(values == EXPECTED, 'Headline counts do not match the frozen paper')
    print(json.dumps(values, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
