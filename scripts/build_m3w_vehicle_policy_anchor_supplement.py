"""Build the v186 anonymous reviewer package."""

from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_vehicle_policy_anchor_supplement import verify
from scripts import build_m3w_vehicle_family_supplement as previous
from scripts import build_m3w_vehicle_policy_anchor_candidate as paper

bundle, archive, main = previous.bundle, previous.archive, previous.main
OUT = archive.PUBLIC/'vehicle_policy_anchor_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v16.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_vehicle_policy_anchor_supplement'


def capacity_counts(r):
    return {**{f'shortlist.hit_{k}': v for k, v in zip((1, 3, 6, 24), r['hits'], strict=True)},
            **{f'shortlist.{q}_cap_{k}': v for q, values in zip(('ego_count', 'fixed_15pct'), r['caps'], strict=True)
               for k, v in zip((1, 3, 6, 24), values, strict=True)}}


def package():
    parent = bundle.read_receipt(previous.OUT.name)
    candidate = bundle.read_receipt(paper.OUT.name)
    d, hashes = paper.inputs()
    if (candidate['prior_manuscript_sha256'] != parent['companion_pdf_sha256']
            or candidate['source_readout_sha256'] != hashes):
        raise ValueError('Recipient readout does not bind the manuscript')
    pdf = candidate['artifacts']['manuscript.pdf']
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    data, expected = previous.panel(d, paper.STUDY)
    top = json.loads((paper.STUDY/'topk_capacity.json').read_text())
    for site, row in zip(sorted(top['by_site']), data['locations'], strict=True):
        r = top['by_site'][site]
        if (r['views'], r['hits'][0]) != (row['counts']['views'], row['counts']['successful_nominees']):
            raise ValueError('Shortlist lost the frozen top-one roster')
        row['counts'].update(capacity_counts(r))
    expected['counts'].update(capacity_counts(top['summary']))
    with ZipFile(BytesIO(bundle.checked(previous.DEST, parent['archive_sha256']))) as z:
        items = {k: z.read(k) for k in verify.OLD-{'README.md'}}
        items['verify_family_v15.py'] = z.read('verify.py')
    items.update({k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')})
    items['vehicle_policy_anchor.json'] = main.as_json(data)
    if set(items) != verify.MEMBERS or any(
            archive.PRIVATE_MARKERS.search(b) for k, b in items.items() if k != 'decision_evidence.zip'):
        raise ValueError('Identifying or unexpected material in replay')
    items['manifest.json'] = main.as_json({k: main.digest(b) for k, b in items.items()})
    builder = previous.previous.previous.previous.previous.previous.prior.previous.prior_builder
    util = builder.previous
    with TemporaryDirectory(prefix='vehicle-policy-review-') as directory:
        p = Path(directory)
        for name in ('first.zip', 'second.zip'):
            archive.encode(p/name, items)
        result = (p/'first.zip').read_bytes()
        if result != (p/'second.zip').read_bytes():
            raise ValueError('Non-deterministic package')
        util.unpack(p/'first.zip', p/'unpacked')
        if not builder.check(util.run_replay(p/'unpacked')['vehicle_policy_anchor'], expected):
            raise ValueError('Anonymous recipient comparison changed')
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
