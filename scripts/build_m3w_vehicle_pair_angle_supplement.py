"""Build and independently replay the anonymous v189 companion."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_vehicle_pair_angle_supplement import verify
from reproducibility.focused_review_supplement.verify import unpack, run_replay
from scripts.build_m3w_exact_exchange_review_supplement import check
from scripts import build_m3w_vehicle_angle_agreement_supplement as prior
from scripts import build_m3w_vehicle_pair_angle_candidate as paper

bundle, archive, main = prior.bundle, prior.archive, prior.main
OUT = archive.PUBLIC/'vehicle_pair_angle_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v19.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_vehicle_pair_angle_supplement'


def flatten(d):
    return {f'{tag}.{q}.{a}.{k}': v for tag, c in d.items() for q, z in c.items()
            for a, r in z.items() for k, v in r.items()}


def panel(d):
    freeze = json.loads((paper.STUDY/'freeze.json').read_text())
    local = paper.OUT.parent/paper.STUDY.name
    grouped = defaultdict(list)
    training = dict(sum((Counter(c['training']) for c in freeze['cells'].values()), Counter()))
    if training != d['training']:
        raise ValueError('Unbound training counts')
    keys = None
    for name, c in freeze['cells'].items():
        for s, v in c['packets'].items():
            cal, = set(c['packets'])-{s}
            row = json.loads((local/f'{name}_{s}_score.json').read_text())
            if row['identity'] != dict(freeze=d['freeze'], site=s, seed=int(name.rsplit('_', 1)[-1]),
                                      calibration=cal, packet=v['sha256']):
                raise ValueError('Unbound score receipt')
            fields = {tag: row['results'][tag]['counts'] for tag in d['counts']}
            fields = flatten({tag: {q: {a: fields[tag][q][a] for a in m} for q, m in r.items()}
                              for tag, r in d['counts'].items()})
            if keys is not None and fields.keys() != keys:
                raise ValueError('Mismatched scoring fields')
            keys = fields.keys()
            grouped[s].append(fields)
    totals = flatten(d['counts'])
    if totals.keys()-keys:
        raise ValueError('Unexpected counts')
    effects = {**{f'incumbent.{k}': v for k, v in flatten(d['effects']).items()},
               **{f'direct_minus.{k}': v for k, v in flatten(d['direct_minus']).items()}}
    groups = []
    for s, rows in sorted(grouped.items()):
        if len(rows) != 18:
            raise ValueError('Missing score directions')
        groups.append(dict(counts={k: sum(r[k] for r in rows) for k in keys},
                           effects={k: v['by_site'][s] for k, v in effects.items()}))
    return dict(locations=groups, training=training, draws=np.random.default_rng(2917).integers(7, size=(3000, 7)).tolist()), dict(
        counts={k: totals.get(k, 0) for k in keys}, training=training,
        effects={k: {m: r[m] for m in ('point', 'ci95')} for k, r in effects.items()})


def package():
    parent = bundle.read_receipt(prior.OUT.name)
    candidate = bundle.read_receipt(paper.OUT.name)
    d, hashes = paper.inputs()
    if (candidate['prior_manuscript_sha256'] != parent['companion_pdf_sha256']
            or candidate['source_readout_sha256'] != hashes):
        raise ValueError('Pair-angle readout does not bind the manuscript')
    pdf = candidate['artifacts']['manuscript.pdf']
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    data, expected = panel(d)
    data['missing_label_bounds'] = expected['missing_label_bounds'] = d['bounds']['effects']
    items = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    items['pair_angle.json'] = main.as_json(data)
    if any(archive.PRIVATE_MARKERS.search(b) for b in items.values()):
        raise ValueError('Identifying material in the new panel')
    items['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(items) != verify.MEMBERS:
        raise ValueError('Unexpected evidence members')
    items['manifest.json'] = main.as_json({k: main.digest(b) for k, b in items.items()})
    with TemporaryDirectory(prefix='vehicle-pair-review-') as folder:
        p = Path(folder)
        for name in ('first.zip', 'second.zip'):
            archive.encode(p/name, items)
        result = (p/'first.zip').read_bytes()
        if result != (p/'second.zip').read_bytes():
            raise ValueError('Non-deterministic review archive')
        unpack(p/'first.zip', p/'unpacked')
        if not check(run_replay(p/'unpacked')['pair_angle'], expected):
            raise ValueError('Anonymous angle-target readout changed')
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
