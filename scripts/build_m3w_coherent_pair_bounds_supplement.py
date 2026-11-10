"""Add the new development diagnostic to the v190 anonymous companion."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.coherent_pair_bounds_supplement import verify
from reproducibility.focused_review_supplement.verify import unpack, run_replay
from scripts.build_m3w_exact_exchange_review_supplement import check
from scripts import build_m3w_external_completion_supplement as prior

bundle, archive, main = prior.bundle, prior.archive, prior.main
STUDY = archive.PUBLIC/'opendd_pair_coherent_bounds_v1'
OUT = archive.PUBLIC/'coherent_pair_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v21.zip'
TEMPLATE = archive.ROOT/'reproducibility/coherent_pair_bounds_supplement'


def package():
    source = (STUDY/'readout.json').read_bytes()
    d = json.loads(source)
    if (not d['original_bound_replayed'] or not d['witnesses_query_specific_only'] or d['external_outcomes_read']
            or not d['original_2pct'] or d['deployment_changed'] or d['directions'] != 126):
        raise ValueError('Require unchanged development comparisons')
    parent = bundle.read_receipt(prior.OUT.name)
    draft = bundle.read_receipt(prior.paper.OUT.name)
    pdf = draft['artifacts']['manuscript.pdf']
    if parent['companion_pdf_sha256'] != pdf:
        raise ValueError('Review archive does not bind the current main manuscript')
    bundle.checked(prior.paper.OUT/'manuscript.pdf', pdf)
    e = d['estimates']
    sites = {q: {a: {k: m['site_means'] for k, m in row['metrics'].items()} for a, row in v.items()} for q, v in e.items()}
    expected = dict(counts=d['counts'], effects={q: {a: {k: {f: m[f] for f in ('point', 'ci95')}
                    for k, m in row['metrics'].items()} for a, row in v.items()} for q, v in e.items()})
    panel = dict(counts=d['counts'], site_means=sites,
                 draws=np.random.default_rng(2917).integers(7, size=(3000, 7)).tolist())
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items['method.md'] = (STUDY/'method.md').read_bytes()
    items['coherent_pairs.json'] = main.as_json(panel)
    if any(archive.PRIVATE_MARKERS.search(v) for v in items.values()):
        raise ValueError('Identifying material in the new panel')
    items['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(items) != verify.MEMBERS:
        raise ValueError('Unexpected archive roster')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    with TemporaryDirectory(prefix='coherent-pair-review-') as folder:
        p = Path(folder)
        for name in ('first.zip', 'second.zip'):
            archive.encode(p/name, items)
        result = (p/'first.zip').read_bytes()
        if result != (p/'second.zip').read_bytes():
            raise ValueError('Non-deterministic review archive')
        unpack(p/'first.zip', p/'unpacked')
        if not check(run_replay(p/'unpacked')['coherent_pairs'], expected):
            raise ValueError('Shared-position intervals did not replay')
    row = dict(archive_sha256=main.digest(result), archive_bytes=len(result), prior_bundle_sha256=parent['archive_sha256'],
        companion_pdf_sha256=pdf, source_readout_sha256=main.digest(source), standard_library_replay=True,
        includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for path, content in {DEST: result, OUT/'receipt.json': main.as_json(row)}.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError('Refusing to replace a different verified artifact')
        if not path.exists():
            with path.open('xb') as f:
                f.write(content)
    print(row)


if __name__ == '__main__':
    package()
