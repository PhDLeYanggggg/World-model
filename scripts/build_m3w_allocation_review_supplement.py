"""Extend the verified anonymous bundle with both OpenDD allocation controls."""

import json
from math import isclose
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_allocation_supplement import verify
from scripts import package_m3w_combined_review_supplement as bundle
from reproducibility.focused_review_supplement.verify import run_replay, unpack


main, archive = bundle.main, bundle.archive
OUT = archive.PUBLIC / 'allocation_review_supplement_v1'
DEST = bundle.LOCAL / 'review_supplement_v4.zip'
TEMPLATE = archive.ROOT / 'reproducibility/opendd_allocation_supplement'


def payload(movement, fitted):
    groups = []
    if set(movement['by_site']) != set(fitted['by_site']):
        raise ValueError('Scored locations differ')
    for site in sorted(movement['by_site']):
        group = []
        for m, p in zip(movement['by_site'][site], fitted['by_site'][site], strict=True):
            if (m['seed'], m['views'], m['online'], m['score_plan']) != (p['seed'], p['online']['scores']['frozen']['views'],
                    p['online']['scores']['frozen']['dual_positive'], p['matched']['scores']['frozen']['dual_positive']):
                raise ValueError('Scored site and seed rows differ')
            row = {k: m[k] for k in ('views', 'edits', 'count_motion', 'count_only')} | {'online_pass': m['online']}
            for setting in ('online', 'matched'):
                row[setting] = {a: {k: v[k] for k in ('edits', 'dual_positive', 'risk_pass', 'undefined')}
                                | {'paired_lower_pp': p[setting]['paired_lower_pp'].get(a, 0.)}
                                for a, v in p[setting]['scores'].items()}
            group.append(row)
        groups.append(group)
    draws = np.random.default_rng(817).integers(7, size=(3000, 7)).tolist()
    return dict(locations=groups, bootstrap_location_indices=draws, matched_unresolved=fitted['summary']['matched_unresolved'])


def check_replay(result, motion, fitted):
    for k in ('count_motion', 'count_only'):
        if result['motion'][k] != motion['summary'][k]:
            raise ValueError('Hindsight counts changed')
    m, r = motion['contrasts']['count_motion_minus_online'], result['motion']['count_motion_minus_online']
    if not all(isclose(a, b, abs_tol=1e-11) for a, b in zip(
            [r['point'], *r['interval95']], [m['point_pp'], *m['ci95_pp']], strict=True)):
        raise ValueError('Hindsight interval changed')
    for mode in ('online', 'matched'):
        for arm, r in result[mode].items():
            f = fitted['summary'][mode][arm]
            if any(r[k] != f[k] for k in ('edits', 'dual_positive', 'risk_pass', 'undefined')):
                raise ValueError('Arm counts changed')
            for key in ('paired_pass_rate_pp', 'paired_conservative_gain_pp'):
                if key in r and not all(isclose(a, b, abs_tol=1e-11) for a, b in zip(
                        [r[key]['point'], *r[key]['interval95']], [f[key]['point'], *f[key]['interval95']], strict=True)):
                    raise ValueError('Paired interval changed')


def package():
    old = bundle.read_receipt('focused_review_supplement_v1')
    draft = bundle.read_receipt('evidence_manuscript_v176_count_motion')
    v3 = bundle.checked(bundle.LOCAL / 'review_supplement_v3.zip', old['archive_sha256'])
    motion_raw = bundle.checked(archive.PUBLIC / 'opendd_count_motion_v1/readout.json', draft['count_motion_diagnostic_sha256'])
    fit_dir = archive.PUBLIC / 'opendd_prefix_value_v1'
    fit_raw = (fit_dir / 'readout.json').read_bytes()
    motion, fitted = json.loads(motion_raw), json.loads(fit_raw)
    for value, path in ((motion, archive.PUBLIC / 'opendd_count_motion_v1'), (fitted, fit_dir)):
        if (value['registration'] != main.digest((path / 'registration.json').read_bytes())
                or value['confirmation_access'] or not value['original_2pct']):
            raise ValueError('A development result is not bound to its original protocol')
    if fitted['checkpoint_manifest'] != main.digest((fit_dir / 'head_freeze.json').read_bytes()):
        raise ValueError('Fitted readout is not bound to the checkpoint manifest')
    items = {k: (TEMPLATE / k).read_bytes() for k in ('README.md', 'verify.py')}
    items |= {'focused_evidence.zip': v3, 'allocation.json': main.as_json(payload(motion, fitted))}
    if set(items) != verify.MEMBERS or any(archive.PRIVATE_MARKERS.search(raw) for name, raw in items.items()
                                           if name != 'focused_evidence.zip'):
        raise ValueError('Identifying or unexpected content in the anonymous supplement')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    with TemporaryDirectory(prefix='allocation-review-') as temp:
        temp = Path(temp)
        for p in (temp / 'first.zip', temp / 'second.zip'):
            archive.encode(p, items)
        raw = (temp / 'first.zip').read_bytes()
        if raw != (temp / 'second.zip').read_bytes():
            raise ValueError('The reviewer archive is not deterministic')
        unpack(temp / 'first.zip', temp / 'unpacked')
        result = run_replay(temp / 'unpacked')
        check_replay(result['allocation'], motion, fitted)
        if result['inherited']['original']['headline'] != old['headline']:
            raise ValueError('Prior evidence changed')
    receipt = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), prior_bundle_sha256=old['archive_sha256'],
        companion_pdf_sha256=draft['artifacts']['manuscript.pdf'], motion_readout_sha256=main.digest(motion_raw),
        fitted_readout_sha256=main.digest(fit_raw), standard_library_replay=True, deterministic=True,
        includes_checkpoints=False, includes_raw_labels=False, uploaded=False, submission_ready=False)
    for p, v in {DEST: raw, OUT / 'receipt.json': main.as_json(receipt)}.items():
        if p.exists() and p.read_bytes() != v:
            raise FileExistsError(f'Refusing to replace {p.name}')
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    package()
