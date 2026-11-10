"""Bind the equal-replacement replay and inherited evidence to the v191 draft."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.interval_match_supplement import verify
from scripts import build_m3w_coherent_pair_bounds_supplement as prior
from scripts import build_m3w_interval_match_candidate as paper
from scripts.build_m3w_exact_exchange_review_supplement import check

bundle, archive, main = prior.bundle, prior.archive, prior.main
OUT = archive.PUBLIC/'interval_match_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v22.zip'
TEMPLATE = archive.ROOT/'reproducibility/interval_match_supplement'


def package():
    (d, _), hashes = paper.inputs()
    parent = bundle.read_receipt(prior.OUT.name)
    draft = bundle.read_receipt(paper.OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    if draft['source_readout_sha256'] != hashes or draft['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Draft and previous archive do not share the same lineage')
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    panel = dict(location_counts=d['anonymous_site_counts'],
                 site_means={q: {k: v['site_means'] for k, v in row.items()}
                             for q, row in d['interval_minus_direct'].items()},
                 draws=np.random.default_rng(2917).integers(7, size=(3000, 7)).tolist())
    expected = dict(totals=d['totals'], effects={q: {k: {i: v[i] for i in ('point', 'ci95')} for k, v in row.items()}
                                              for q, row in d['interval_minus_direct'].items()})
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items['matched.json'] = main.as_json(panel)
    items['protocol.md'] = bundle.checked(paper.STUDY/'protocol.md', hashes['protocol'])
    if any(archive.PRIVATE_MARKERS.search(v) for v in items.values()):
        raise ValueError('Identifying material in the new panel')
    items['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(items) != verify.MEMBERS:
        raise ValueError('Unexpected archive roster')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    with TemporaryDirectory(prefix='equal-replacements-review-') as tmp:
        p = Path(tmp)
        for name in ('first.zip', 'second.zip'):
            archive.encode(p/name, items)
        data = (p/'first.zip').read_bytes()
        if data != (p/'second.zip').read_bytes():
            raise ValueError('Non-deterministic archive')
        unpack(p/'first.zip', p/'unpacked')
        if not check(run_replay(p/'unpacked')['equal_replacements'], expected):
            raise ValueError('Matched comparison does not replay')
    row = dict(archive_sha256=main.digest(data), archive_bytes=len(data), prior_bundle_sha256=parent['archive_sha256'],
        companion_pdf_sha256=pdf, source_readout_sha256=hashes['matched'], standard_library_replay=True,
        includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for path, content in {DEST: data, OUT/'receipt.json': main.as_json(row)}.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError('Refusing to replace a different artifact')
        if not path.exists():
            with path.open('xb') as f:
                f.write(content)
    print(row)


if __name__ == '__main__':
    package()
