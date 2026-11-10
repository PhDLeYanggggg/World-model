"""Build an anonymous review package from the v185 aggregate evidence."""

from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_vehicle_family_supplement import verify
from scripts import build_m3w_vehicle_nominee_supplement as previous
from scripts import build_m3w_vehicle_family_candidate as paper

bundle, archive, main = previous.bundle, previous.archive, previous.main
OUT = archive.PUBLIC/'vehicle_family_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v15.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_vehicle_family_supplement'


def panel(d, study=paper.STUDY):
    freeze = json.loads((study/'freeze.json').read_text())
    effects = {f'{k}.{m}': v for k, r in d['paired'].items() for m, v in r.items()}
    effects.update({f'{k}.{m}': v for k, r in d['ranking'].items() for m, v in r.items() if 'by_site' in v})
    local = paper.OUT.parent/study.name
    loc = []
    for s in sorted(next(iter(effects.values()))['by_site']):
        rows = []
        for name, c in freeze['cells'].items():
            if s not in c['packets']:
                continue
            cal, = set(c['packets'])-{s}
            r = json.loads((local/f'{name}_{s}_score.json').read_text())
            if r['identity'] != dict(freeze=d['freeze'], site=s, seed=int(name.rsplit('_', 1)[-1]),
                                     calibration=cal, packet=c['packets'][s]['sha256']):
                raise ValueError('Unbound family-aligned direction')
            rows.append(r)
        if len(rows) != 18:
            raise ValueError('Expected 18 directions per location')
        counts = {f'{a}.{q}.{k}': sum(r['scores'][a][q][k] for r in rows) for a, v in d['counts'].items()
                  for q, m in v.items() for k in m}
        counts.update({k: sum(r[k] for r in rows) for k in d['totals']})
        loc.append(dict(counts=counts, effects={k: r['by_site'][s] for k, r in effects.items()}))
    want = dict(counts={**d['totals'], **{f'{a}.{q}.{k}': n for a, v in d['counts'].items()
                for q, m in v.items() for k, n in m.items()}},
                effects={k: {m: r[m] for m in ('point', 'ci95')} for k, r in effects.items()})
    return dict(locations=loc, draws=np.random.default_rng(2917).integers(7, size=(3000, 7)).tolist()), want


def package():
    parent = bundle.read_receipt(previous.OUT.name)
    candidate = bundle.read_receipt(paper.OUT.name)
    d, hashes = paper.inputs()
    if (candidate['prior_manuscript_sha256'] != parent['companion_pdf_sha256']
            or candidate['source_readout_sha256'] != hashes):
        raise ValueError('Family readout does not bind the manuscript')
    pdf = candidate['artifacts']['manuscript.pdf']
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    data, expected = panel(d)
    with ZipFile(BytesIO(bundle.checked(previous.DEST, parent['archive_sha256']))) as z:
        items = {k: z.read(k) for k in verify.OLD-{'README.md'}}
        items['verify_nominee_v14.py'] = z.read('verify.py')
    items.update({k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')})
    items['vehicle_family.json'] = main.as_json(data)
    if set(items) != verify.MEMBERS or any(
            archive.PRIVATE_MARKERS.search(b) for k, b in items.items() if k != 'decision_evidence.zip'):
        raise ValueError('Identifying or unexpected replay material')
    items['manifest.json'] = main.as_json({k: main.digest(b) for k, b in items.items()})
    builder = previous.previous.previous.previous.previous.prior.previous.prior_builder
    util = builder.previous
    with TemporaryDirectory(prefix='vehicle-family-review-') as directory:
        p = Path(directory)
        for name in ('first.zip', 'second.zip'):
            archive.encode(p/name, items)
        result = (p/'first.zip').read_bytes()
        if result != (p/'second.zip').read_bytes():
            raise ValueError('Non-deterministic package')
        util.unpack(p/'first.zip', p/'unpacked')
        if not builder.check(util.run_replay(p/'unpacked')['vehicle_family'], expected):
            raise ValueError('Anonymous family-aligned scores changed')
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
