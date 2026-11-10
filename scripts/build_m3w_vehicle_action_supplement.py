"""Bind anonymous action-value replay to the v182 manuscript."""

from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_vehicle_action_supplement import verify
from scripts import build_m3w_vehicle_action_value_candidate as paper
from scripts import build_m3w_vehicle_context_supplement as previous

bundle, archive, main = previous.bundle, previous.archive, previous.main
OUT = archive.PUBLIC/'vehicle_action_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v12.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_vehicle_action_supplement'


def export(d):
    f = json.loads((paper.STUDY/'freeze.json').read_text())
    effects = {f'{k}.{m}': v for k, r in d['paired'].items() for m, v in r.items()}
    loc = []
    local = paper.OUT.parent/paper.STUDY.name
    for s in sorted(next(iter(effects.values()))['by_site']):
        rows = []
        for name, c in f['cells'].items():
            if s not in c['packets']:
                continue
            cal, = set(c['packets'])-{s}
            score = json.loads((local/f'{name}_{s}_score.json').read_text())
            if score['identity'] != dict(freeze=d['freeze'], packet=c['packets'][s]['sha256'], site=s,
                                         calibration=cal, seed=int(name.rsplit('_', 1)[-1])):
                raise ValueError('Unbound grouped score')
            rows.append(score)
        if len(rows) != 18:
            raise ValueError('Require 18 paired direction scores per location')
        counts = {f'{a}.{k}': sum(r['arms'][a][k] for r in rows) for a, v in d['counts'].items() for k in v}
        counts['views'] = sum(r['views'] for r in rows)
        loc.append(dict(counts=counts, effects={k: r['by_site'][s] for k, r in effects.items()}))
    want = dict(counts={'views': d['views'], **{f'{a}.{k}': w for a, v in d['counts'].items() for k, w in v.items()}},
                effects={k: {m: r[m] for m in ('point', 'ci95')} for k, r in effects.items()})
    return dict(locations=loc, draws=np.random.default_rng(2917).integers(7, size=(3000, 7)).tolist()), want


def package():
    prior = bundle.read_receipt(previous.OUT.name)
    candidate = bundle.read_receipt(paper.OUT.name)
    d, hashes = paper.inputs()
    if (candidate['prior_manuscript_sha256'] != prior['companion_pdf_sha256']
            or candidate['source_readout_sha256'] != hashes):
        raise ValueError('Action comparison does not bind the compiled manuscript')
    pdf = candidate['artifacts']['manuscript.pdf']
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    data, expected = export(d)
    with ZipFile(BytesIO(bundle.checked(previous.DEST, prior['archive_sha256']))) as z:
        items = {k: z.read(k) for k in verify.OLD-{'README.md'}}
        items['verify_context_v11.py'] = z.read('verify.py')
    items.update({k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')})
    items['vehicle_action.json'] = main.as_json(data)
    if set(items) != verify.MEMBERS or any(
            archive.PRIVATE_MARKERS.search(b) for k, b in items.items() if k != 'decision_evidence.zip'):
        raise ValueError('Identifying or unexpected replay material')
    items['manifest.json'] = main.as_json({k: main.digest(b) for k, b in items.items()})
    builder = previous.previous.prior.previous.prior_builder
    util = builder.previous
    with TemporaryDirectory(prefix='action-review-') as directory:
        root = Path(directory)
        for filename in ('first.zip', 'second.zip'):
            archive.encode(root/filename, items)
        output = (root/'first.zip').read_bytes()
        if output != (root/'second.zip').read_bytes():
            raise ValueError('Non-deterministic supplement')
        util.unpack(root/'first.zip', root/'unpacked')
        replay = util.run_replay(root/'unpacked')['vehicle_action']
        if not builder.check(replay, expected):
            raise ValueError('Offline action statistics changed')
    receipt = dict(archive_sha256=main.digest(output), archive_bytes=len(output), prior_bundle_sha256=prior['archive_sha256'],
        companion_pdf_sha256=pdf, source_readout_sha256=hashes, deterministic=True, standard_library_replay=True,
        includes_raw_labels=False, includes_checkpoints=False, uploaded=False, submission_ready=False)
    for path, value in {DEST: output, OUT/'receipt.json': main.as_json(receipt)}.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != value:
                raise FileExistsError(f'Refusing to replace {path.name}')
        else:
            with path.open('xb') as f:
                f.write(value)
    print(receipt)


if __name__ == '__main__':
    package()
