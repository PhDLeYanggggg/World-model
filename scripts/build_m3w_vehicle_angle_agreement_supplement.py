"""Build the v188 anonymous companion, preserving the prior evidence byte for byte."""

from collections import defaultdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_vehicle_angle_agreement_supplement import verify
from scripts import build_m3w_vehicle_angle_agreement_candidate as paper
from scripts import build_m3w_vehicle_shortlist_supplement as prior

bundle, archive, main = prior.bundle, prior.archive, prior.main
OUT = archive.PUBLIC/'vehicle_angle_agreement_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v18.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_vehicle_angle_agreement_supplement'


def flatten(d):
    return {**{f'{q}.{a}.{k}': v for q, c in d['counts'].items() for a, r in c.items() for k, v in r.items()},
            **{f'{q}.capacity.{k}': v for q, c in d['capacity'].items() for k, v in c.items()}}


def panel(d):
    freeze = json.loads((paper.STUDY/'freeze.json').read_text())
    local = paper.OUT.parent/paper.STUDY.name
    grouped = defaultdict(list)
    for k, v in freeze['directions'].items():
        ref = v['identity']
        s, name = ref['site'], ref['direction']
        cal, = set(name.split('_')[:-1])-{s}
        row = json.loads((local/f'{k}_score.json').read_text())
        if row['identity'] != dict(freeze=d['freeze'], site=s, seed=int(name.rsplit('_', 1)[-1]),
                                  calibration=cal, packet=v['sha256']):
            raise ValueError('Unbound direction in angle readout')
        grouped[s].append(flatten(row))
    effects = {f'{q}.{a}.{k}': v for q, c in d['effects'].items() for a, r in c.items() for k, v in r.items()}
    counts = flatten(d)
    totals = {k: counts.get(k, 0) for k in next(iter(grouped.values()))[0]}
    if counts.keys()-totals.keys():
        raise ValueError('Unexpected totals')
    groups = []
    for s, rows in sorted(grouped.items()):
        if len(rows) != 18:
            raise ValueError('Missing angle directions')
        groups.append(dict(counts={k: sum(r[k] for r in rows) for k in totals},
                           effects={k: v['by_site'][s] for k, v in effects.items()}))
    return dict(locations=groups, draws=np.random.default_rng(2917).integers(7, size=(3000, 7)).tolist()), dict(
        counts=totals, effects={k: {m: r[m] for m in ('point', 'ci95')} for k, r in effects.items()})


def package():
    parent = bundle.read_receipt(prior.OUT.name)
    candidate = bundle.read_receipt(paper.OUT.name)
    d, hashes = paper.inputs()
    if (candidate['prior_manuscript_sha256'] != parent['companion_pdf_sha256']
            or candidate['source_readout_sha256'] != hashes):
        raise ValueError('Angle readout does not bind the manuscript')
    pdf = candidate['artifacts']['manuscript.pdf']
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    data, expected = panel(d)
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items['angle_agreement.json'] = main.as_json(data)
    if any(archive.PRIVATE_MARKERS.search(b) for b in items.values()):
        raise ValueError('Identifying material in new replay')
    items['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(items) != verify.MEMBERS:
        raise ValueError('Unexpected evidence members')
    items['manifest.json'] = main.as_json({k: main.digest(b) for k, b in items.items()})
    builder = prior.previous.previous.previous.previous.previous.previous.previous.prior.previous.prior_builder
    util = builder.previous
    with TemporaryDirectory(prefix='vehicle-angle-review-') as folder:
        p = Path(folder)
        for name in ('first.zip', 'second.zip'):
            archive.encode(p/name, items)
        result = (p/'first.zip').read_bytes()
        if result != (p/'second.zip').read_bytes():
            raise ValueError('Non-deterministic review archive')
        util.unpack(p/'first.zip', p/'unpacked')
        if not builder.check(util.run_replay(p/'unpacked')['angle_agreement'], expected):
            raise ValueError('Anonymous agreement readout changed')
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
