"""Build the v187 anonymous aggregate replay package."""

from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_vehicle_shortlist_supplement import verify
from scripts import build_m3w_vehicle_policy_anchor_supplement as previous
from scripts import build_m3w_vehicle_shortlist_candidate as paper

bundle, archive, main = previous.bundle, previous.archive, previous.main
OUT = archive.PUBLIC/'vehicle_shortlist_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v17.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_vehicle_shortlist_supplement'


def package():
    parent = bundle.read_receipt(previous.OUT.name)
    candidate = bundle.read_receipt(paper.OUT.name)
    d, hashes = paper.inputs()
    if (candidate['prior_manuscript_sha256'] != parent['companion_pdf_sha256']
            or candidate['source_readout_sha256'] != hashes):
        raise ValueError('Shortlist readout does not bind the manuscript')
    pdf = candidate['artifacts']['manuscript.pdf']
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    data, expected = previous.previous.panel(d, paper.STUDY, counts_key='counts')
    data['fixed_mask_capacity'] = expected['fixed_mask_capacity'] = d['fixed_mask_capacity']
    with ZipFile(BytesIO(bundle.checked(previous.DEST, parent['archive_sha256']))) as z:
        items = {k: z.read(k) for k in verify.OLD-{'README.md'}}
        items['verify_policy_v16.py'] = z.read('verify.py')
    items.update({k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')})
    items['vehicle_shortlist.json'] = main.as_json(data)
    if set(items) != verify.MEMBERS or any(
            archive.PRIVATE_MARKERS.search(b) for k, b in items.items() if k != 'decision_evidence.zip'):
        raise ValueError('Identifying or unexpected material in replay')
    items['manifest.json'] = main.as_json({k: main.digest(b) for k, b in items.items()})
    builder = previous.previous.previous.previous.previous.previous.previous.prior.previous.prior_builder
    util = builder.previous
    with TemporaryDirectory(prefix='vehicle-shortlist-review-') as directory:
        p = Path(directory)
        for name in ('first.zip', 'second.zip'):
            archive.encode(p/name, items)
        result = (p/'first.zip').read_bytes()
        if result != (p/'second.zip').read_bytes():
            raise ValueError('Non-deterministic package')
        util.unpack(p/'first.zip', p/'unpacked')
        if not builder.check(util.run_replay(p/'unpacked')['vehicle_shortlist'], expected):
            raise ValueError('Anonymous shortlist comparison changed')
    receipt = dict(archive_sha256=main.digest(result), archive_bytes=len(result), prior_bundle_sha256=parent['archive_sha256'],
        companion_pdf_sha256=pdf, source_readout_sha256=hashes, deterministic=True, standard_library_replay=True,
        includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for p, b in {DEST: result, OUT/'receipt.json': main.as_json(receipt)}.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            if p.read_bytes() != b:
                raise FileExistsError(f'Refusing to replace {p.name}')
        else:
            with p.open('xb') as f:
                f.write(b)
    print(receipt)


if __name__ == '__main__':
    package()
