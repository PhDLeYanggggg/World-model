"""Package the verified v178 exchange frontier alongside the earlier evidence."""

import json
from math import isclose
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.opendd_frontier_supplement import verify
from scripts import package_m3w_combined_review_supplement as bundle


main, archive = bundle.main, bundle.archive
OUT = archive.PUBLIC/'frontier_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v6.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_frontier_supplement'


def export(row):
    keys = ('views', 'edits', 'anchor_dual', 'bounds')
    locations = [[{k: r[k] for k in keys} for r in row['by_site'][s]] for s in sorted(row['by_site'])]
    draws = np.random.default_rng(817).integers(7, size=(3000, 7)).tolist()
    return dict(locations=locations, draws=draws)


def check(replayed, s):
    if any(replayed[k] != s[k] for k in ('views', 'edits', 'anchor_dual', 'bounds')):
        raise ValueError('Scalar replay changed the feasibility range')
    a, b = replayed['verified_one_minus_zero_pp'], s['verified_one_minus_zero_pp']
    if (a['locations_positive'] != b['locations_positive'] or not all(isclose(x, y, abs_tol=1e-11) for x, y in zip(
            [a['point_pp'], *a['ci95_pp']], [b['point_pp'], *b['ci95_pp']], strict=True))):
        raise ValueError('Scalar replay changed the location interval')


def package():
    prior = bundle.read_receipt('decision_review_supplement_v1')
    paper = bundle.read_receipt('evidence_manuscript_v178_exchange_frontier')
    v5 = bundle.checked(bundle.LOCAL/'review_supplement_v5.zip', prior['archive_sha256'])
    h = paper['exchange_frontier_readout_sha256']
    row = json.loads(bundle.checked(archive.PUBLIC/'opendd_exchange_frontier_v1/readout.json', h))
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items |= {'decision_evidence.zip': v5, 'frontier.json': main.as_json(export(row))}
    if set(items) != verify.MEMBERS or any(archive.PRIVATE_MARKERS.search(v) for k, v in items.items() if k != 'decision_evidence.zip'):
        raise ValueError('Identifying or unexpected frontier evidence')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    with TemporaryDirectory(prefix='frontier-review-') as temp:
        root = Path(temp)
        for p in (root/'first.zip', root/'second.zip'):
            archive.encode(p, items)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('The reviewer archive is not deterministic')
        unpack(root/'first.zip', root/'unpacked')
        replay = run_replay(root/'unpacked')
        check(replay['exchange_frontier'], row['summary'])
    receipt = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), prior_bundle_sha256=prior['archive_sha256'],
        companion_pdf_sha256=paper['artifacts']['manuscript.pdf'], frontier_readout_sha256=h, standard_library_replay=True,
        deterministic=True, includes_checkpoints=False, includes_raw_labels=False, uploaded=False, submission_ready=False)
    for p, v in {DEST: raw, OUT/'receipt.json': main.as_json(receipt)}.items():
        if p.exists() and p.read_bytes() != v:
            raise FileExistsError(f'Refusing to replace {p.name}')
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    package()
