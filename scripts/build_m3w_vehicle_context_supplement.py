"""Bind the anonymous geometry/context replay to the v181 candidate."""

from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_vehicle_context_supplement import verify
from scripts import build_m3w_vehicle_context_candidate as paper
from scripts import build_m3w_vehicle_direction_supplement as previous


bundle, archive, main = previous.bundle, previous.archive, previous.main
OUT = archive.PUBLIC/'vehicle_context_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v11.zip'
LOCAL = paper.OUT.parent/paper.STUDY.name
TEMPLATE = archive.ROOT/'reproducibility/opendd_vehicle_context_supplement'


def panel(d):
    f = json.loads((paper.STUDY/'freeze.json').read_text())
    effects = {f'{k}.{m}': v for k, r in d['paired'].items() for m, v in r.items()}
    effects.update({f'ade.{k}': v for k, v in d['available_ade'].items()})
    locations = []
    for site in sorted(d['available_ade']['cv']['by_site']):
        rows = []
        for name, cell in f['cells'].items():
            if site in cell['sites']:
                score = json.loads((LOCAL/f'{name}_{site}_score.json').read_text())
                if score['identity'] != dict(freeze=d['freeze'], packet=cell['sites'][site]['sha256'],
                                             site=site, seed=int(name.rsplit('_', 1)[-1])):
                    raise ValueError('Changed grouped score')
                rows.append(score)
        if len(rows) != 18:
            raise ValueError('Each location must have 18 grouped receipts')
        counts = {f'{a}.{k}': sum(r['arms'][a][k] for r in rows) for a, v in d['counts'].items() for k in v}
        counts['views'] = sum(r['counts']['views'] for r in rows)
        locations.append(dict(counts=counts, effects={k: r['by_site'][site] for k, r in effects.items()}))
    want = dict(counts={'views': d['views'], **{f'{a}.{k}': w for a, v in d['counts'].items() for k, w in v.items()}},
                effects={k: {m: r[m] for m in ('point', 'ci95')} for k, r in effects.items()})
    return dict(locations=locations, draws=np.random.default_rng(2917).integers(7, size=(3000, 7)).tolist()), want


def package():
    parent = bundle.read_receipt(previous.OUT.name)
    candidate = bundle.read_receipt(paper.OUT.name)
    d, hashes = paper.inputs()
    if (candidate['prior_manuscript_sha256'] != parent['companion_pdf_sha256']
            or candidate['source_readout_sha256'] != hashes):
        raise ValueError('Companion manuscript or source readouts differ')
    pdf = candidate['artifacts']['manuscript.pdf']
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    data, expected = panel(d)
    raw = bundle.checked(previous.DEST, parent['archive_sha256'])
    with ZipFile(BytesIO(raw)) as z:
        items = {k: z.read(k) for k in verify.OLD-{'README.md'}}
        items['verify_vehicle.py'] = z.read('verify.py')
    items.update({k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')})
    items['vehicle_context.json'] = main.as_json(data)
    if set(items) != verify.MEMBERS or any(
            archive.PRIVATE_MARKERS.search(b) for k, b in items.items() if k != 'decision_evidence.zip'):
        raise ValueError('Identifying or unexpected material in the replay')
    items['manifest.json'] = main.as_json({k: main.digest(b) for k, b in items.items()})
    util = previous.prior.previous.prior_builder.previous
    with TemporaryDirectory(prefix='vehicle-context-review-') as tmp:
        p = Path(tmp)
        for filename in ('first.zip', 'second.zip'):
            archive.encode(p/filename, items)
        output = (p/'first.zip').read_bytes()
        if output != (p/'second.zip').read_bytes():
            raise ValueError('Non-deterministic review archive')
        util.unpack(p/'first.zip', p/'unpacked')
        replay = util.run_replay(p/'unpacked')['vehicle_context']
        if not previous.prior.previous.prior_builder.check(replay, expected):
            raise ValueError('Anonymous replay changed the results')
    receipt = dict(archive_sha256=main.digest(output), archive_bytes=len(output), prior_bundle_sha256=parent['archive_sha256'],
        companion_pdf_sha256=pdf, source_readout_sha256=hashes, deterministic=True, standard_library_replay=True,
        includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for p, b in {DEST: output, OUT/'receipt.json': main.as_json(receipt)}.items():
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
