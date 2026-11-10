"""Package anonymized direction and threshold evidence beside the frozen paper."""

from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_angular_supplement import verify
from scripts import build_m3w_context_review_supplement as previous


bundle, main, archive = previous.bundle, previous.main, previous.archive
OUT = archive.PUBLIC/'angular_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v9.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_angular_supplement'

# Aggregate paths differ from grouped paths only for interval summaries.
PANELS = {
    'direction_menu': ('opendd_exchange_directions_v1',
        {'original_dual': 'oracle.original.positive', 'expanded_dual': 'oracle.expanded.positive',
         'expanded_material': 'oracle.expanded.gain_gt_0_10pp'}, {}),
    'same_recipients': ('opendd_exchange_angular_same_recipients_v1',
        {'original_dual': 'arms.original.dual', 'relative_dual': 'arms.relative.dual'},
        {'relative_vs_original': ('effect.relative.lower_pp', 'paired.relative.lower_pp')}),
    'angle_oracle': ('opendd_angular_incumbent_oracle_v1',
        {'material_views': 'arms.constant_angle.above_0_10pp', 'oracle_dual': 'arms.constant_angle.dual'},
        {'constant_angle': ('arms.constant_angle.lower_pp', 'arms.constant_angle')}),
    'angle_choice': ('opendd_angular_choice_v1',
        {'direct_rotations': 'arms.direct.rotated_edits', 'factor_rotations': 'arms.factor.rotated_edits',
         'factor_dual': 'arms.factor.dual'},
        {'factor_vs_original': ('lower_pp.factor_minus_original', 'paired.factor_minus_original.lower')}),
    'calibration': ('opendd_angular_calibration_v1',
        {'zero_rotations': 'arms.zero.rotated_edits', 'calibrated_rotations': 'arms.mean.rotated_edits',
         'stable_rotations': 'arms.stable.rotated_edits', 'constant_rotations': 'arms.constant_at_mean.rotated_edits',
         'calibrated_dual': 'arms.mean.dual', 'constant_dual': 'arms.constant_at_mean.dual'},
        {'constant_vs_incumbent': ('arms.constant_at_mean.gain_pp', 'arms.constant_at_mean.vs_incumbent'),
         'model_vs_incumbent': ('arms.mean.gain_pp', 'arms.mean.vs_incumbent')}),
    'matched_mask': ('opendd_angular_calibration_mask_v1',
        {'rotations': 'counts.rotations', 'rotation_views': 'counts.rotation_views', 'flexible_views': 'counts.flexible_views'},
        {k: ('lower.'+k, 'paired.'+k) for k in (
            'constant_minus_model', 'constant_minus_random_mask', 'random_mask_vs_incumbent')}),
}


def get(row, path, missing=None):
    try:
        for key in path.split('.'):
            row = row[key]
    except KeyError:
        if missing is None:
            raise
        return missing
    return row


def export(label, d):
    _, sums, effects = PANELS[label]
    sums = dict(views='counts.views', edits='counts.edits', **sums)
    sites = d['by_site']
    if len(sites) != 7 or any(len(g) != 3 or sorted(r['seed'] for r in g) != [17, 29, 43] for g in sites.values()):
        raise ValueError('Changed group or seed roster')
    rows = [[dict(counts={k: get(r, p, 0) for k, p in sums.items()},
                  effects={k: get(r, p[0]) for k, p in effects.items()})
             for r in sorted(sites[s], key=lambda r: r['seed'])] for s in sorted(sites)]
    summary = d['summary']
    expected = dict(counts={k: get(summary, p) for k, p in sums.items()}, effects={})
    for k, (_, p) in effects.items():
        v = get(summary, p)
        expected['effects'][k] = dict(point_pp=v.get('point_pp', v.get('lower_pp')), ci95_pp=v['ci95_pp'])
    return rows, expected


def package():
    prior = bundle.read_receipt('context_review_supplement_v1')
    paper = bundle.read_receipt('evidence_manuscript_v179_exact_exchange')
    pdf = paper['artifacts']['manuscript.pdf']
    if pdf != prior['companion_pdf_sha256']:
        raise ValueError('Changed companion manuscript')
    raw = bundle.checked(previous.DEST, prior['archive_sha256'])
    data, expected, hashes = {}, {}, {}
    for label, (folder, _, _) in PANELS.items():
        root = archive.PUBLIC/folder
        raw_readout = (root/'readout.json').read_bytes()
        r = json.loads(raw_readout)
        bundle.checked(root/'registration.json', r['registration'])
        if r['external_outcome_access'] or not r['original_2pct']:
            raise ValueError('Angular results do not have the registered development roles')
        if 'checkpoint_manifest' in r:
            bundle.checked(root/'head_freeze.json', r['checkpoint_manifest'])
        if 'calibration' in r:
            bundle.checked(root/'calibration_freeze.json', r['calibration'])
        data[label], expected[label] = export(label, r)
        hashes[label] = main.digest(raw_readout)
    with ZipFile(BytesIO(raw)) as z:
        items = {k: z.read(k) for k in verify.OLD-{'README.md'}}
        items['verify_context.py'] = z.read('verify.py')
    items.update({k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')})
    items['angular.json'] = main.as_json(data)
    if set(items) != verify.MEMBERS or any(
        archive.PRIVATE_MARKERS.search(v) for k, v in items.items() if k != 'decision_evidence.zip'
    ):
        raise ValueError('Identifying or unexpected angular evidence')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    utility = previous.prior_builder.previous
    with TemporaryDirectory(prefix='angular-review-') as directory:
        root = Path(directory)
        for name in ('first.zip', 'second.zip'):
            archive.encode(root/name, items)
        output = (root/'first.zip').read_bytes()
        if output != (root/'second.zip').read_bytes():
            raise ValueError('Reviewer archive is not deterministic')
        utility.unpack(root/'first.zip', root/'unpacked')
        replayed = utility.run_replay(root/'unpacked')
        if not previous.prior_builder.check(replayed['angular'], expected):
            raise ValueError('Anonymous replay changed an angular result')
    receipt = dict(archive_sha256=main.digest(output), archive_bytes=len(output),
        prior_bundle_sha256=prior['archive_sha256'], companion_pdf_sha256=pdf, source_sha256=hashes,
        standard_library_replay=True, deterministic=True, includes_raw_labels=False, includes_checkpoints=False,
        uploaded=False, submission_ready=False)
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
