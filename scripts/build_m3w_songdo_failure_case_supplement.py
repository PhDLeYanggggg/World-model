"""Package the anonymous case study with the unchanged v201 manuscript replay."""

import json
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.songdo_failure_cases_supplement import verify
from scripts import build_m3w_songdo_step_budget_candidate as prior

archive, bundle, main = prior.archive, prior.bundle, prior.main
STUDY = main.PUBLIC/'songdo_failure_cases_v1'
TEMPLATE = archive.ROOT/'reproducibility/songdo_failure_cases_supplement'
DEST = bundle.LOCAL/'review_supplement_v34.zip'


def package():
    parent = bundle.read_receipt(prior.PACK.name)
    pdf = parent['companion_pdf_sha256']
    bundle.checked(prior.OUT/'manuscript.pdf', pdf)
    raw = (STUDY/'readout.json').read_bytes()
    row = json.loads(raw)
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts['cases.json'] = raw
    parts['protocol.md'] = bundle.checked(STUDY/'protocol.md', row['identity']['protocol_sha256'])
    parts['failure_cases.pdf'] = bundle.checked(prior.OUT.parent/STUDY.name/'failure_cases.pdf',
                                               row['identity']['figure_sha256'])
    pdf_doc = PdfReader(BytesIO(parts['failure_cases.pdf']))
    extracted = ' '.join(p.extract_text() for p in pdf_doc.pages)
    if (len(pdf_doc.pages) != 1 or 'Incomplete future labels' not in extracted
            or any(archive.PRIVATE_MARKERS.search(p) for p in [*parts.values(), extracted.encode(),
                                                              str(pdf_doc.metadata).encode()])):
        raise ValueError('Identifying material in case addendum')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected archive contents')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='songdo-cases-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        payload = (root/'first.zip').read_bytes()
        if payload != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if run_replay(root/'unpacked')['songdo_failure_cases'] != verify.replay(row):
            raise ValueError('Anonymous case replay disagrees with results')
    receipt = dict(archive_sha256=main.digest(payload), archive_bytes=len(payload),
        prior_bundle_sha256=parent['archive_sha256'], companion_pdf_sha256=pdf,
        source_readout_sha256=main.digest(raw), figure_sha256=row['identity']['figure_sha256'],
        standard_library_replay=True, includes_raw_labels=False, includes_checkpoints=False,
        uploaded=False, submission_ready=False)
    for path, content in ((DEST, payload), (STUDY/'review_receipt.json', main.as_json(receipt))):
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f'Refusing to replace frozen review artifact: {path.name}')
        if not path.exists():
            with path.open('xb') as f:
                f.write(content)
    print(receipt)


if __name__ == '__main__':
    package()
