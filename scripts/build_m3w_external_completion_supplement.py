"""Package the updated external uncertainty panel without any outcome coordinates."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.external_completion_supplement import verify
from reproducibility.focused_review_supplement.verify import unpack, run_replay
from scripts.build_m3w_exact_exchange_review_supplement import check
from scripts import build_m3w_external_completion_candidate as paper
from scripts import build_m3w_vehicle_pair_angle_supplement as prior

bundle, archive, main = prior.bundle, prior.archive, prior.main
OUT = archive.PUBLIC/'external_completion_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v20.zip'
TEMPLATE = archive.ROOT/'reproducibility/external_completion_supplement'


def package():
    prior_receipt = bundle.read_receipt(prior.OUT.name)
    draft = bundle.read_receipt(paper.OUT.name)
    d, hashes = paper.inputs()
    if (draft['source_readout_sha256'] != hashes
            or draft['prior_manuscript_sha256'] != prior_receipt['companion_pdf_sha256']):
        raise ValueError('Unbound external uncertainty draft')
    pdf = draft['artifacts']['manuscript.pdf']
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    bundle.checked(paper.OUT/paper.FIGURE, draft['added_archive_assets_sha256'][paper.FIGURE])
    receipt, h = paper.evidence.storage.load(paper.evidence.SONGDO/'confirmation_receipt.json')
    if h != hashes['songdo']:
        raise ValueError('Changed source components')
    rows = {a: dict(observed=r['known_pp']['component_points'], lower=r['lower_pp']['component_points'])
            for a, r in receipt['matched_joint_minus'].items()}
    w = d['wepdtof']['curve']
    exp = dict(songdo=d['songdo']['controls'], wepdtof_bounds=[w['lower'], w['upper']])
    draws = np.random.default_rng(paper.evidence.DRAW_SEED).integers(14, size=(paper.evidence.DRAWS, 14)).tolist()
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items['external_completion.json'] = main.as_json(dict(songdo_components=rows, draws=draws, wepdtof_bounds=exp['wepdtof_bounds']))
    if any(archive.PRIVATE_MARKERS.search(v) for v in items.values()):
        raise ValueError('Identifying material in the new panel')
    items['prior_evidence.zip'] = bundle.checked(prior.DEST, prior_receipt['archive_sha256'])
    if set(items) != verify.MEMBERS:
        raise ValueError('Unexpected archive members')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    with TemporaryDirectory(prefix='external-completion-review-') as folder:
        root = Path(folder)
        for name in ('first.zip', 'second.zip'):
            archive.encode(root/name, items)
        result = (root/'first.zip').read_bytes()
        if result != (root/'second.zip').read_bytes():
            raise ValueError('Non-deterministic review archive')
        unpack(root/'first.zip', root/'unpacked')
        if not check(run_replay(root/'unpacked')['external_completion'], exp):
            raise ValueError('External uncertainty estimates did not replay')
    row = dict(archive_sha256=main.digest(result), archive_bytes=len(result), prior_bundle_sha256=prior_receipt['archive_sha256'],
        companion_pdf_sha256=pdf, source_readout_sha256=hashes, standard_library_replay=True,
        includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for p, b in {DEST: result, OUT/'receipt.json': main.as_json(row)}.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            if p.read_bytes() != b:
                raise FileExistsError(f'Refusing to replace {p.name}')
        else:
            with p.open('xb') as f:
                f.write(b)
    print(row)


if __name__ == '__main__':
    package()
