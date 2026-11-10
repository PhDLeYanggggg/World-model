"""Bind the WEPDTOF equal-step control to the anonymous v217 draft."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_integrated_reader_candidate as prior
from scripts import build_m3w_reader_supplement as reader
from scripts.build_m3w_full_grid_candidate import builder
from reproducibility.m3w_wepdtof_equal_step_replay.verify import MEMBERS, replay

OUT = prior.OUT.with_name('evidence_manuscript_v217_wepdtof_equal_steps')
PUBLIC = prior.PUBLIC.with_name(OUT.name)
READOUT = PUBLIC.parent/'wepdtof_equal_step_v1/readout.json'
TEMPLATE = reader.TEMPLATE.with_name('m3w_wepdtof_equal_step_replay')
HANDOFF = reader.ADMIN/'local_handoff_v217'


def inputs():
    raw = READOUT.read_bytes()
    d = json.loads(raw)
    if (not d['original_2pct'] or not d['previously_scored_outcomes_reused'] or d['new_fit']
            or d['new_external_targets_opened'] or d['original_release_changed'] or d['independent_confirmation']
            or (d['views'], d['total_sources']) != (45, 13)):
        raise ValueError('Expected only a sensitivity of the original WEPDTOF outcome')
    if d['freeze_sha256'] != reader.digest(READOUT.with_name('geometry_freeze.json').read_bytes()):
        raise ValueError('The matched geometry is not bound to the readout')
    data = {k: d[k] for k in ('counts', 'retention', 'source_values', 'by_comparator', 'equal_steps_minus_original')}
    rng = np.random.default_rng(82317)
    data['draws'] = rng.integers(10, size=(3000, 10)).tolist()
    replay(data)
    return data, dict(wepdtof_equal_steps=reader.digest(raw))


def revise(text, data):
    anchor = 'excludes beating both references in 3 of those views.'
    if text.count(anchor) != 1:
        raise ValueError('Expected the frozen WEPDTOF result')
    row = replay(data)
    lo, hi = (f'{v:+.5f}' for v in row['envelope_pp'])
    updated = (f"\nMatching movement at every step leaves the joint-minus-independent\n"
               f"envelope at [{lo}, {hi}] pp. Joint and additive forecasts\n"
               "are identical in all 36 edited views, so this test cannot attribute\n"
               "an effect to the cross term. Neither sensitivity changes release.")
    return text.replace(anchor, anchor+updated, 1), (f'[{lo}, {hi}]', 'identical in all 36 edited views')


def extend_pdf(previous, data):
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(612, 792), invariant=1)
    pdf.setAuthor('Anonymous')
    y = reader.layout.header(pdf, 'FROZEN EXTERNAL SENSITIVITY', 'WEPDTOF: match movement at each step')
    para, table = reader.layout.para, reader.layout.table
    y = para(pdf, 'The original forecasts already match edit counts and peak displacement. '
             'The stepwise control holds each edited recipient and waypoint direction fixed. '
             'At each step it assigns the minimum original departure from CV across both '
             'arms to every edited path. It retains all 198 edited queries and 366 edits '
             'per arm. These are query-seed counts, not independent samples.', y)
    rows = []
    for key, name in (('fixed_independent', 'Independent'), ('fixed_uniform', 'Same family'),
                      ('fixed_no_pair', 'Additive')):
        a = data['by_comparator'][key]['matched_steps']
        b = data['equal_steps_minus_original'][key]['observed']
        rows.append([name, f"{a['observed']['point']:+.5f}",
                     f"[{a['lower']['point']:+.5f}, {a['upper']['point']:+.5f}]",
                     f"{b['point']:+.5f} [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}]"])
    y = table(pdf, [['Joint minus', 'Observed gain', 'Full-grid envelope', 'Change vs original [95% interval]'], *rows],
              [93, 92, 133, 182], y)
    y = para(pdf, 'Gains are in percentage points of the common observed CV error; positive favors '
             'joint. The interval resamples the ten estimable original sources, after averaging clips '
             'and seeds. Three of the thirteen sources do not support the original paired comparison. '
             'The full-grid envelope reflects missing targets, not source sampling uncertainty.', y)
    r = data['by_comparator']['fixed_independent']['matched_steps']
    y = para(pdf, f"Relative-vector gain over independent is {r['relative_observed']['point']:+.5f} "
             f"pixels on common labeled pair-steps, with a full-grid envelope of "
             f"[{r['relative_lower']['point']:+.5f}, {r['relative_upper']['point']:+.5f}]. "
             'The equal-step paths retain the original peak and about 97% of integrated '
             'movement. A uniformly scaled smaller-peak control therefore reproduces the '
             'original results to rounding precision.', y)
    y = para(pdf, 'Joint and additive return the same trajectories in all 36 edited views '
             'before step matching. Their zero contrast is an identity on this sample, not '
             'evidence that the cross term has no effect in other settings. The new variants '
             'are not release candidates; the original risk and release results are unchanged.', y)
    reader.finish(pdf, 9, y)
    pdf.save()
    writer = PdfWriter()
    for raw in (previous, buf.getvalue()):
        for p in PdfReader(BytesIO(raw)).pages:
            if p.get('/Annots'):
                raise ValueError('Unexpected annotation in the supplementary material')
            writer.add_page(p)
    writer.add_metadata({'/Title': 'Supplementary material', '/Author': 'Anonymous'})
    out = BytesIO()
    writer.write(out)
    pages = PdfReader(BytesIO(out.getvalue())).pages
    text = ' '.join(' '.join(p.extract_text().split()) for p in pages)
    if len(pages) != 9 or reader.PRIVATE.search(text) or '198 edited queries' not in text:
        raise ValueError('Unexpected anonymous appendix text or layout')
    return out.getvalue()


def package():
    data, hashes = inputs()
    parent = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    main = json.loads((PUBLIC/'receipt.json').read_bytes())
    if main['prior_manuscript_sha256'] != parent['companion_pdf_sha256'] or main['source_readout_sha256'] != hashes:
        raise ValueError('Main and supplementary evidence differ')
    paper = reader.checked(OUT/'manuscript.pdf', main['artifacts']['manuscript.pdf'])
    old = reader.checked(prior.HANDOFF/'replay.zip', parent['archive_sha256'])
    supplement = extend_pdf(reader.checked(prior.HANDOFF/'supplement.pdf', parent['reader_pdf_sha256']), data)
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': supplement, 'equal_steps.json': reader.as_json(data), 'prior_evidence.zip': old})
    if set(parts) != MEMBERS or reader.PRIVATE.search(parts['equal_steps.json'].decode()):
        raise ValueError('Unexpected replay contents')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    payload = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='v217-replay-') as tmp:
        p = Path(tmp)/'replay.zip'
        p.write_bytes(payload)
        reader.unpack(p, Path(tmp)/'extracted')
        value = reader.run_replay(Path(tmp)/'extracted')['wepdtof_equal_steps']
        if value != replay(data):
            raise ValueError('Anonymous replay did not reproduce WEPDTOF')
    files = {'paper.pdf': paper, 'supplement.pdf': supplement, 'replay.zip': payload}
    manifest = {k: reader.digest(v) for k, v in files.items()}
    receipt = dict(companion_pdf_sha256=manifest['paper.pdf'], reader_pdf_sha256=manifest['supplement.pdf'],
                   reader_pages=9, archive_sha256=manifest['replay.zip'], archive_bytes=len(payload),
                   prior_archive_sha256=parent['archive_sha256'], source_readout_sha256=hashes,
                   replay=value, original_2pct=True, uploaded=False, submission_ready=False)
    targets = {**{HANDOFF/k: v for k, v in files.items()}, HANDOFF/'manifest.json': reader.as_json(manifest),
               PUBLIC/'review_receipt.json': reader.as_json(receipt)}
    if any(k.exists() and k.read_bytes() != v for k, v in targets.items()):
        raise FileExistsError('Refusing to overwrite a different reviewer artifact')
    for k, v in targets.items():
        k.parent.mkdir(parents=True, exist_ok=True)
        if not k.exists():
            with k.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    args = parser.parse_args()
    builder.build(OUT, args.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                  read_inputs=inputs, revise_source=revise)
    package()
