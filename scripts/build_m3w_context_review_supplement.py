"""Add the matched OpenDD context experiment to the anonymous replay."""

from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.opendd_context_supplement import verify
from scripts import build_m3w_exact_exchange_review_supplement as prior_builder


bundle, main, archive = prior_builder.bundle, prior_builder.main, prior_builder.archive
REPORT = archive.PUBLIC/'opendd_exchange_context_v1'
OUT = archive.PUBLIC/'context_review_supplement_v1'
DEST = bundle.LOCAL/'review_supplement_v8.zip'
TEMPLATE = archive.ROOT/'reproducibility/opendd_context_supplement'


def export(own, matched):
    def counts(arms):
        return {a: {k: s[k] for k in verify.FIELDS} for a, s in arms.items()}

    if own['by_site'].keys() != matched['by_site'].keys():
        raise ValueError('Context cohorts do not match')
    locations = []
    for site in sorted(own['by_site']):
        rows = []
        for a, b in zip(own['by_site'][site], matched['by_site'][site], strict=True):
            if a['seed'] != b['seed'] or any(
                s['views'] != b['views'] or s['edits'] != b['edits'] for s in a['scores'].values()
            ) or a['transitions']['aligned']['exchanged'] != b['exchanged']:
                raise ValueError('Context matched-swap counts changed')
            rows.append(dict(views=b['views'], edits=b['edits'], exchanged=b['exchanged'],
                             own_mask=counts(a['scores']), matched=counts(b['arms']), effect=b['effect']))
        locations.append(rows)
    s = matched['summary']
    expected = {k: s[k] for k in ('views', 'edits', 'exchanged', 'aligned_minus')}
    expected.update(own_mask=counts(own['summary']['arms']), matched=counts(s['arms']))
    return dict(locations=locations), expected


def package():
    prior = bundle.read_receipt('exact_exchange_review_supplement_v1')
    paper = bundle.read_receipt('evidence_manuscript_v179_exact_exchange')
    pdf = paper['artifacts']['manuscript.pdf']
    if pdf != prior['companion_pdf_sha256']:
        raise ValueError('Companion manuscript changed')
    raw = bundle.checked(prior_builder.DEST, prior['archive_sha256'])
    own_raw = (REPORT/'readout.json').read_bytes()
    matched_raw = (REPORT/'matched_swap_readout.json').read_bytes()
    own, matched = json.loads(own_raw), json.loads(matched_raw)
    bundle.checked(REPORT/'registration.json', own['registration'])
    bundle.checked(REPORT/'head_freeze.json', own['checkpoint_manifest'])
    if (matched['rank_readout_sha256'] != main.digest(own_raw)
        or any(r['external_outcome_access'] is not False or r['original_2pct'] is not True for r in (own, matched))
        or matched['no_training'] is not True
        or matched['common_intervention_mask'] != 'guarded_aligned_history_features'):
        raise ValueError('Context readouts do not have the expected provenance')
    context, expected = export(own, matched)
    with ZipFile(BytesIO(raw)) as z:
        items = {k: z.read(k) for k in verify.OLD - {'README.md'}}
        items['verify_exchange.py'] = z.read('verify.py')
    items.update({k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')})
    items['context.json'] = main.as_json(context)
    if set(items) != verify.MEMBERS or any(
        archive.PRIVATE_MARKERS.search(v) for k, v in items.items() if k != 'decision_evidence.zip'
    ):
        raise ValueError('Identifying or unexpected context evidence')
    items['manifest.json'] = main.as_json({k: main.digest(v) for k, v in items.items()})
    with TemporaryDirectory(prefix='context-review-') as directory:
        root = Path(directory)
        for p in (root/'first.zip', root/'second.zip'):
            archive.encode(p, items)
        output = (root/'first.zip').read_bytes()
        if output != (root/'second.zip').read_bytes():
            raise ValueError('The reviewer archive is not deterministic')
        prior_builder.previous.unpack(root/'first.zip', root/'unpacked')
        replayed = prior_builder.previous.run_replay(root/'unpacked')
        if not prior_builder.check(replayed['exchange_context'], expected):
            raise ValueError('Anonymous replay changed the matched context result')
    receipt = dict(archive_sha256=main.digest(output), archive_bytes=len(output),
                   prior_bundle_sha256=prior['archive_sha256'], companion_pdf_sha256=pdf,
                   own_mask_readout_sha256=main.digest(own_raw), matched_readout_sha256=main.digest(matched_raw),
                   standard_library_replay=True, deterministic=True, includes_checkpoints=False,
                   includes_raw_labels=False, uploaded=False, submission_ready=False)
    for p, v in {DEST: output, OUT/'receipt.json': main.as_json(receipt)}.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            if p.read_bytes() != v:
                raise FileExistsError(f'Refusing to replace {p.name}')
        else:
            with p.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    package()
