"""Package the exact v179 audit without another nested evidence archive."""

from io import BytesIO
import json
from math import isclose
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_exact_exchange_supplement import verify
from scripts import build_m3w_frontier_review_supplement as previous


bundle, main, archive = previous.bundle, previous.main, previous.archive
OUT = archive.PUBLIC/'exact_exchange_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v7.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_exact_exchange_supplement'


def export(row):
    keys = ('views', 'edits', 'candidates', 'anchor_dual', 'exact_frontier', 'comparisons')
    return dict(locations=[[{k: r[k] for k in keys} for r in row['by_site'][s]] for s in sorted(row['by_site'])])


def check(a, b):
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return all(check(v, b[k]) for k, v in a.items())
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(check(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return isclose(a, b, abs_tol=1e-11)
    return a == b


def package():
    prior = bundle.read_receipt('frontier_review_supplement_v1')
    paper = bundle.read_receipt('evidence_manuscript_v179_exact_exchange')
    raw = bundle.checked(previous.DEST, prior['archive_sha256'])
    h = paper['exact_exchange_readout_sha256']
    row = json.loads(bundle.checked(archive.PUBLIC/'opendd_one_exchange_ranker_v1/ranking_diagnostic.json', h))
    with ZipFile(BytesIO(raw)) as z:
        old = {k: z.read(k) for k in ('frontier.json', 'decision_evidence.zip', 'verify.py')}
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items |= {'frontier.json': old['frontier.json'], 'decision_evidence.zip': old['decision_evidence.zip'],
              'verify_frontier.py': old['verify.py'], 'ranking.json': main.as_json(export(row))}
    if set(items) != verify.MEMBERS or any(archive.PRIVATE_MARKERS.search(v) for k, v in items.items() if k != 'decision_evidence.zip'):
        raise ValueError('Identifying or unexpected exchange evidence')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    with TemporaryDirectory(prefix='exact-exchange-review-') as directory:
        root = Path(directory)
        for p in (root/'first.zip', root/'second.zip'):
            archive.encode(p, items)
        output = (root/'first.zip').read_bytes()
        if output != (root/'second.zip').read_bytes():
            raise ValueError('The reviewer archive is not deterministic')
        previous.unpack(root/'first.zip', root/'unpacked')
        replay = previous.run_replay(root/'unpacked')
        if not check(replay['exact_exchange'], row['summary']):
            raise ValueError('Anonymous replay changed the exhaustive result')
    receipt = dict(archive_sha256=main.digest(output), archive_bytes=len(output), prior_bundle_sha256=prior['archive_sha256'],
        companion_pdf_sha256=paper['artifacts']['manuscript.pdf'], exact_exchange_readout_sha256=h,
        standard_library_replay=True, deterministic=True, includes_checkpoints=False, includes_raw_labels=False,
        uploaded=False, submission_ready=False)
    for p, v in {DEST: output, OUT/'receipt.json': main.as_json(receipt)}.items():
        if p.exists() and p.read_bytes() != v:
            raise FileExistsError(f'Refusing to replace {p.name}')
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    package()
