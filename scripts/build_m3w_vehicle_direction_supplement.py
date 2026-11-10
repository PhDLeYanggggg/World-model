"""Package fixed-recipient vehicle direction comparisons for anonymous replay."""

from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_vehicle_direction_supplement import verify
from scripts import build_m3w_angular_review_supplement as prior
from scripts import build_m3w_vehicle_direction_candidate as paper


bundle, archive, main = prior.bundle, prior.archive, prior.main
OUT = archive.PUBLIC/'vehicle_direction_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v10.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_vehicle_direction_supplement'
FORCED = ('coupled_minus_independent', 'coupled_minus_original')
SINGLE = ('interaction_minus_independent_at_interaction', 'interaction_minus_original')


def export(row, forced):
    names = sorted({r['identity']['site'] for r in row['by_direction']})
    counters = (('any_relative_win', 'single_rotation_relative_win', 'shared_relative_win', 'angular_dual')
                if forced else ())
    arms = ({'coupled_dual': ('coupled', 'dual'), 'independent_dual': ('independent', 'dual')} if forced else
            {'interaction_dual': ('interaction', 'dual'), 'matched_dual': ('independent_at_interaction', 'dual'),
             'rotations': ('interaction', 'rotated_views'), 'matched_rotations': ('independent_at_interaction', 'rotated_views')})

    def counts(r):
        return dict(views=r['counts']['views'], original_dual=r['arms']['original']['dual'],
                    **{k: r['counts'][k] for k in counters}, **{k: r['arms'][a][b] for k, (a, b) in arms.items()})

    pairs = FORCED if forced else SINGLE
    stats = ('lower_pp', 'pair_lower')
    data = [[[dict(counts=counts(r), effect={f'{a}_{b}': r['contrasts'][a][b] for a in pairs for b in stats})
              for r in row['by_direction'] if r['identity']['site'] == s and r['identity']['seed'] == seed]
             for seed in (17, 29, 43)] for s in names]
    summary = row['summary']
    want = dict(counts=counts(summary), effects={f'{a}_{b}': summary['paired'][a][b] for a in pairs for b in stats})
    return data, want


def package():
    parent = bundle.read_receipt('angular_review_supplement_v1')
    candidate = bundle.read_receipt(paper.OUT.name)
    if candidate['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Companion manuscript is not based on the verified prior')
    pdf = candidate['artifacts']['manuscript.pdf']
    bundle.checked(paper.OUT/'manuscript.pdf', pdf)
    rows, hashes = paper.inputs()
    if hashes != candidate['source_readout_sha256']:
        raise ValueError('Vehicle results differ from the compiled manuscript')
    panels, expected = {}, {}
    for label, r in zip(('forced', 'one_rotation'), rows, strict=True):
        panels[label], expected[label] = export(r, label == 'forced')
    data = dict(panels=panels, draws=np.random.default_rng(2917).integers(7, size=(3000, 7)).tolist())
    raw = bundle.checked(prior.DEST, parent['archive_sha256'])
    with ZipFile(BytesIO(raw)) as z:
        items = {k: z.read(k) for k in verify.OLD-{'README.md'}}
        items['verify_angular.py'] = z.read('verify.py')
    items.update({k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')})
    items['vehicle.json'] = main.as_json(data)
    if set(items) != verify.MEMBERS or any(
            archive.PRIVATE_MARKERS.search(b) for k, b in items.items() if k != 'decision_evidence.zip'):
        raise ValueError('Identifying or unexpected vehicle evidence')
    items['manifest.json'] = main.as_json({k: main.digest(b) for k, b in items.items()})
    utility = prior.previous.prior_builder.previous
    with TemporaryDirectory(prefix='vehicle-direction-review-') as directory:
        root = Path(directory)
        for filename in ('first.zip', 'second.zip'):
            archive.encode(root/filename, items)
        output = (root/'first.zip').read_bytes()
        if output != (root/'second.zip').read_bytes():
            raise ValueError('Non-deterministic anonymous archive')
        utility.unpack(root/'first.zip', root/'unpacked')
        result = utility.run_replay(root/'unpacked')
        if not prior.previous.prior_builder.check(result['vehicle_direction'], expected):
            raise ValueError('Anonymous replay changed a vehicle result')
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
