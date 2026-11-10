"""Correct the WEPDTOF comparator and report its frozen-score crossover."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_wepdtof_equal_step_candidate as prior
from scripts import build_m3w_reader_supplement as reader
from scripts.build_m3w_full_grid_candidate import builder
from reproducibility.m3w_wepdtof_penalty_replay.verify import MEMBERS, replay

OUT = prior.OUT.with_name('evidence_manuscript_v218_wepdtof_penalty')
PUBLIC = prior.PUBLIC.with_name(OUT.name)
TEMPLATE = reader.TEMPLATE.with_name('m3w_wepdtof_penalty_replay')
HANDOFF = reader.ADMIN/'local_handoff_v218'
BREAK = PUBLIC.parent/'wepdtof_penalty_breakpoint_v1/readout.json'


def inputs():
    steps, digests = prior.inputs()
    raw = BREAK.read_bytes()
    d = json.loads(raw)
    if (not d['frozen_paths_exactly_reproduced'] or not d['original_2pct']
            or any(d[k] for k in ('outcomes_opened', 'new_fit', 'weight_changed', 'release_changed'))
            or d['edited_identity_views'] != steps['counts']['fixed_no_pair']['identical_edited_views']):
        raise ValueError('Expected the forecast-only diagnosis on the same frozen bank')
    replay(d)
    return dict(steps=steps, breaks=d), digests | dict(wepdtof_penalty_breakpoints=reader.digest(raw))


def revise(text, data):
    method = 'the Songdo comparison also fixes the peak displacement of both\nedits within the matched query.'
    moving = 'The two\ngraph controls also score changes in the relative paths of causal'
    related = ('Here the matched comparison isolates the choice of recipients\n'
               'and the contribution of a relative-path cross term; it does not\n'
               'evaluate a downstream planner.')
    old = ('pp. Joint and additive forecasts\n'
           'are identical in all 36 edited views, so this test cannot attribute\n'
           'an effect to the cross term. Neither sensitivity changes release.')
    if any(text.count(anchor) != 1 for anchor in (method, old, moving, related)):
        raise ValueError('Expected the v217 comparator wording')
    spec = ('the Songdo and WEPDTOF comparisons also fix the peak\n'
            'displacement of each edit within a matched query. WEPDTOF\n'
            'uses a proximity-weighted relative-displacement penalty; its\n'
            'control removes the whole penalty. Songdo compares separately\n'
            'trained donor heads: one scores a pair using its relative\n'
            'features and the other combines two actor scores.')
    r = replay(data['breaks'])
    minimum = f"{r['minimum_crossover']:.5f}"
    result = (f'pp. Turning off the entire WEPDTOF pair penalty\n'
              f'leaves all 36 edited views identical: no query changes\n'
              f'at weight 0.10, and the first possible crossover is {minimum}.\n'
              'These are frozen scores, not estimates of accuracy at another weight.')
    text = text.replace(moving, 'On Moving, the two\ngraph controls also score changes in the relative paths of causal', 1)
    text = text.replace(related, 'The Moving ablation isolates the relative-path cross term.\n'
                        'The other matched comparisons test their respective\n'
                        'selection rules, not a downstream planner.', 1)
    return text.replace(method, spec, 1).replace(old, result, 1), (minimum, 'whole penalty', '36 edited views')


def overview():
    raw = BytesIO()
    pdf = canvas.Canvas(raw, pagesize=(612, 792), invariant=1)
    layout = reader.layout
    y = layout.header(pdf, 'SUPPLEMENTARY MATERIAL', 'Reading the replacement experiments')
    y = layout.para(pdf, 'Each query has eight observed and twelve forecast steps. The matched proposal '
                    'comparison and the actual release test answer different questions. CV is the no-edit '
                    'reference, not a passed risk check.', y)
    y = layout.table(pdf, [['Dataset', 'Role', 'Source grouping'],
        ['MovingDroneCrowd++', 'Development and capacity', '14 sources on main roster'],
        ['OpenDD', 'Development and matched controls', '7 held locations'],
        ['DroneCrowd', 'Calibration only', '15 exclusion groups'],
        ['WEPDTOF', 'Frozen evaluation', '13 sources, 10 estimable for paired utility'],
        ['Songdo Traffic', '40 calibration / 160 confirmation flights', '14 confirmation source components']],
        [124, 185, 191], y)
    y = layout.para(pdf, '<b>What the controls remove.</b> Moving compares the squared-relative score with '
                    'and without its cross term. WEPDTOF removes the entire proximity-weighted mean pair '
                    'penalty. Songdo compares separate donor-trained pair and actor heads, not two '
                    'settings of a common analytic penalty. Same-family controls restrict edits to one '
                    'forecast family. Match counts throughout and movement where specified.', y)
    y = layout.para(pdf, '<b>Missing labels.</b> The observed paired gain plus or minus the weighted '
                    'forecast disagreement on missing steps gives the sample envelope. This does not '
                    'impute targets. The bootstrap resamples sources, not missing coordinates, seeds '
                    'or overlapping windows.', y)
    reader.finish(pdf, 1, y)
    pdf.save()
    return raw.getvalue()


def append(previous, d):
    raw = BytesIO()
    pdf = canvas.Canvas(raw, pagesize=(612, 792), invariant=1)
    pdf.setAuthor('Anonymous')
    para, table, head = reader.layout.para, reader.layout.table, reader.layout.header
    y = head(pdf, 'FROZEN EXTERNAL SENSITIVITY', 'WEPDTOF: movement and pair penalty')
    y = para(pdf, 'The fixed-count WEPDTOF score is the two-reference utility minus a proximity-weighted '
             'mean relative departure from CV. The no-pair-penalty arm sets that entire penalty to zero; '
             'it is not the cross-term-only ablation of the squared-relative score used elsewhere.', y)
    y = para(pdf, 'Step matching keeps the edited recipients and directions, replacing each step magnitude '
             'with the smallest one from either arm. All 198 edited query-seed views and all 366 '
             'edits per arm are retained. The common peak stays unchanged.', y)
    rows = []
    for arm, name in (('fixed_independent', 'Independent'), ('fixed_uniform', 'Same family'),
                      ('fixed_no_pair', 'No pair penalty')):
        a = d['steps']['by_comparator'][arm]['matched_steps']
        b = d['steps']['equal_steps_minus_original'][arm]['observed']
        rows.append([name, f"{a['observed']['point']:+.5f}", f"[{a['lower']['point']:+.5f}, {a['upper']['point']:+.5f}]",
                     f"{b['point']:+.5f} [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}]"])
    y = table(pdf, [['Joint minus', 'Observed gain', 'Full-grid envelope', 'Change vs original [95% interval]'], *rows],
              [96, 89, 133, 182], y)
    y = para(pdf, 'Gain is in percentage points of observed CV error; positive favors joint. The interval '
             'resamples ten estimable original sources after averaging clips and seeds. The envelope '
             'instead varies missing targets. The original 2% rule and release decisions do not change.', y)
    r = replay(d['breaks'])
    y = para(pdf, '<b>Why the penalty changes no choice.</b> The unpenalized winner keeps its place until a '
             'lower-penalty candidate can overcome its utility gap. Exact enumeration of the frozen '
             'menu reproduces the original trajectories without opening any target labels.', y)
    y = table(pdf, [['Reason the winner cannot change at weight 0.10', 'Queries'],
                    ['One edit, so pair penalty is zero', r['one_edit']],
                    ['Two edits; winner already minimizes the penalty', r['no_lower_penalty']],
                    ['Two edits; first crossover lies above 0.10', r['finite_crossovers']]],
              [391, 109], y)
    y = para(pdf, f"The earliest possible crossover is {r['minimum_crossover']:.5f}. Joint and no-pair-penalty "
             'forecasts are identical in all 36 edited views. These breakpoints describe the frozen '
             'score and provide no evidence about accuracy or release at a different coefficient.', y)
    reader.finish(pdf, 9, y)
    pdf.save()
    merged = PdfWriter()
    for pages in (PdfReader(BytesIO(overview())).pages, PdfReader(BytesIO(previous)).pages[1:],
                  PdfReader(BytesIO(raw.getvalue())).pages):
        for p in pages:
            if p.get('/Annots'):
                raise ValueError('Unexpected annotation in the reader appendix')
            merged.add_page(p)
    merged.add_metadata({'/Title': 'Supplementary material', '/Author': 'Anonymous'})
    b = BytesIO()
    merged.write(b)
    content = b.getvalue()
    pages = PdfReader(BytesIO(content)).pages
    text = ' '.join(' '.join(p.extract_text().split()) for p in pages)
    if len(pages) != 9 or reader.PRIVATE.search(text) or 'No pair penalty' not in text or '0.11013' not in text:
        raise ValueError('Corrected appendix did not render as expected')
    return content


def package():
    data, hashes = inputs()
    prev = prior.prior
    parent = json.loads((prev.PUBLIC/'review_receipt.json').read_bytes())
    main = json.loads((PUBLIC/'receipt.json').read_bytes())
    replaced = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    if (main['prior_manuscript_sha256'] != replaced['companion_pdf_sha256']
            or main['source_readout_sha256'] != hashes):
        raise ValueError('Wrong manuscript for the corrected appendix')
    paper = reader.checked(OUT/'manuscript.pdf', main['artifacts']['manuscript.pdf'])
    old = reader.checked(prev.HANDOFF/'replay.zip', parent['archive_sha256'])
    appendix = append(reader.checked(prev.HANDOFF/'supplement.pdf', parent['reader_pdf_sha256']), data)
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': appendix, 'prior_evidence.zip': old,
        'equal_step_verify.py': (prior.TEMPLATE/'verify.py').read_bytes(),
        'equal_steps.json': reader.as_json(data['steps']), 'breakpoints.json': reader.as_json(data['breaks'])})
    if set(parts) != MEMBERS or any(reader.PRIVATE.search(parts[k].decode()) for k in ('equal_steps.json', 'breakpoints.json')):
        raise ValueError('Unexpected reviewer archive members')
    parts['manifest.json'] = reader.as_json({k: reader.digest(v) for k, v in parts.items()})
    payload = reader.archive_bytes(parts)
    with TemporaryDirectory(prefix='v218-replay-') as tmp:
        p = Path(tmp)/'replay.zip'
        p.write_bytes(payload)
        reader.unpack(p, Path(tmp)/'extracted')
        r = reader.run_replay(Path(tmp)/'extracted')['wepdtof_penalty_breakpoints']
        if r != replay(data['breaks']):
            raise ValueError('Anonymous breakpoint replay differs')
    files = {'paper.pdf': paper, 'supplement.pdf': appendix, 'replay.zip': payload}
    manifest = {k: reader.digest(v) for k, v in files.items()}
    receipt = dict(companion_pdf_sha256=manifest['paper.pdf'], reader_pdf_sha256=manifest['supplement.pdf'],
                   reader_pages=9, archive_sha256=manifest['replay.zip'], archive_bytes=len(payload),
                   prior_archive_sha256=parent['archive_sha256'], supersedes_reader_v217=True,
                   source_readout_sha256=hashes, replay=r, original_2pct=True, uploaded=False, submission_ready=False)
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
