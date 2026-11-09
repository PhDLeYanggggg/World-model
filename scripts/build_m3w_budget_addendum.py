"""Append verified budget diagnostics to a new local supplementary PDF."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

from build_m3w_supplement_guide import INK, MARGIN, RULE, WIDTH, footer, header, paragraph
from verify_m3w_budget_addendum_scalars import validate

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'outputs/publication_readiness_2026_09'
ADMIN = ROOT/'data/stage_cvpr2027_experiments/submission_admin'
SOURCE = ADMIN/'supplement_with_evidence_guide.pdf'
SOURCE_SHA = 'b36fede903ed779fb49ac23138bb5940331fcba3ecffa24a44ef2dc621ddae40'
RECEIPTS = {
    'exact': ('opendd_exact_slot_headroom_v2', '1241b77020c23a62e8f724408e64f8575e3ac1f63b00d5067f6e943e8e9a8f6b'),
    'expanded': ('opendd_self_menu_capacity_v1', '80639e6ccf481682c6d5bedde12c9df724dbe443195ee1ba338c7cc7f09b5b3f'),
    'trained': ('opendd_expanded_costs_v1', 'c2757b41dc19f32d90afd48df89bc3664ec4cc17e5b7362b587183d9ff65e7eb'),
    'timing': ('opendd_paired_window_budget_v1', 'bcd430a28947536ee2f588aac74fb7047fc37e34a232e534c11f948967584fb7'),
    'risk': ('opendd_paired_window_budget_v1', '4d68126cbd42e676bc8c9439ee8060fd221b9a2611331a1481b22a5bd07bd088'),
    'guard': ('opendd_easy_slot_support_v1', 'e594da1321dddffe35399cc4fdd6923ba27c5c0e6d7ac4f6452246e04af49a84'),
}
SITES = tuple(f'rdb{i}' for i in range(1, 8))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load():
    evidence = {}
    for key, (folder, expected) in RECEIPTS.items():
        p = REPORTS/folder/('risk_attribution.json' if key == 'risk' else 'readout.json')
        if digest(p) != expected:
            raise ValueError(f'Unrecognized source evidence: {key}')
        evidence[key] = json.loads(p.read_text())
    return evidence


def scalar_snapshot(e):
    data = dict(counts={}, contrasts={}, expected={}, resamples=np.random.default_rng(3183)
                .choice(6, size=(3000, 6)).tolist())

    def add_counts(label, sites, keys, total):
        data['counts'][label] = [[sites[s][k] for k in keys] for s in SITES]
        data['expected'][label] = [total[k] for k in keys]

    x = e['exact']
    arm = 'two_edits_only'
    add_counts('original_hindsight', {s: x['groups'][s][arm] for s in SITES},
               ('estimable', 'verified_dual', 'verified_direct_positive', 'verified_direct_positive_recordings'), x['totals'][arm])
    x = e['expanded']
    add_counts('expanded_hindsight', x['sites'],
               ('estimable', 'expanded_dual', 'expanded_positive', 'expanded_positive_recordings'), x['totals'])
    for study, group, totals, arms in (
            ('trained', 'by_site', 'counts', ('frozen_additive', 'frozen_joint', 'trained_additive', 'trained_joint')),
            ('timing', 'sites', 'counts', ('fixed_joint', 'pooled_joint')),
            ('guard', 'sites', 'totals', ('pooled_joint', 'guarded_joint'))):
        x = e[study]
        for arm in arms:
            keys = ('edits', 'risk', 'dual', 'defined_selected_easy') if study == 'guard' else ('edits', 'risk', 'dual')
            add_counts(f'{study}_{arm}', {s: x[group][s]['counts'][arm] for s in SITES}, keys, x[totals][arm])

    for label, study, site_group, nested, key, totals in (
            ('trained_joint_minus_additive', 'trained', 'by_site', 'contrasts_pp', 'trained_joint_vs_trained_additive', 'contrasts'),
            ('pooled_joint_minus_fixed', 'timing', 'sites', 'contrasts', 'pooled_joint_vs_fixed_joint', 'contrasts'),
            ('guarded_joint_minus_pooled', 'guard', 'sites', 'contrast', 'joint', 'contrast')):
        x = e[study]
        data['contrasts'][label] = [x[site_group][s][nested][key] for s in SITES]
        summary = x[totals][key]
        data['expected'][label] = [summary['mean_pp'], *summary['ci95_pp']]
    validate(data)
    return data


def table(c, rows, widths, y):
    if sum(widths) != WIDTH:
        raise ValueError('Table must fit the text area')
    t = Table(rows, colWidths=widths, rowHeights=[24]*len(rows))
    t.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEABOVE', (0, 0), (-1, 0), .75, INK), ('LINEBELOW', (0, 0), (-1, 0), .5, RULE),
        ('LINEBELOW', (0, -1), (-1, -1), .75, INK)]))
    _, h = t.wrap(WIDTH, 450)
    t.drawOn(c, MARGIN, y-h)
    return y-h-17


def interval(values):
    point, lo, hi = values
    return f'{point:+.5f} pp (95% interval [{lo:+.5f}, {hi:+.5f}])'


def pages(s, e, first):
    v = validate(s)
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter, invariant=1)
    c.setTitle('Supplementary matched-budget diagnostics')
    c.setAuthor('')
    header(c, 'OPEN DEVELOPMENT DIAGNOSTIC', 'What can the same edit budget buy?')
    y = paragraph(c, 'We compared ways to choose which forecasts replace constant velocity (CV). '
                  'All arms retain the original 3,690 edit slots and their peak displacements. Each model '
                  'observes eight positions and predicts twelve steps. We charge for missing future '
                  'positions and keep the original recording-level 2% risk rule.', 642)
    rows = [['Hindsight menu', 'Pass risk, beat CV and OLS4', 'Also beat base additive']]
    for label, k in (('Original', 'original_hindsight'), ('Add a self-history forecast', 'expanded_hindsight')):
        n, dual, win, _ = v[k]
        rows.append([label, f'{dual}/{n}', f'{win}/{n}'])
    y = table(c, rows, [198, 146, 144], y)
    y = paragraph(c, 'The oracle uses future labels and keeps every single-edit choice fixed. '
                  f'The {v["expanded_hindsight"][2]} expanded-menu wins span '
                  f'{v["expanded_hindsight"][3]} recordings at three locations. '
                  'They show that better choices exist, not that an online selector can recognize them.', y)
    rows = [['Expanded-menu selector', 'Risk passes', 'Risk and both references']]
    for label, arm in (('Frozen additive', 'frozen_additive'), ('Frozen joint', 'frozen_joint'),
                       ('Retrained additive', 'trained_additive'), ('Retrained joint', 'trained_joint')):
        _, risk, dual = v[f'trained_{arm}']
        rows.append([label, f'{risk}/606', f'{dual}/606'])
    y = table(c, rows, [217, 110, 161], y)
    y = paragraph(c, 'The retrained arms use the same gain and harm estimates. Additive scores each '
                  'edit separately; joint combines the estimated costs before scoring. The conservative '
                  f'joint-minus-additive difference is {interval(v["trained_joint_minus_additive"])}. '
                  'The expanded menu creates some opportunity, but the learned joint rule does not '
                  'outperform the matched additive rule.', y)
    y = paragraph(c, 'The 606 views are 202 exposed OpenDD recordings evaluated at three seeds. The '
                  'paired interval resamples six estimable locations. The seventh location has no '
                  'matched edit comparison. These are development results, not independent confirmation.', y)
    if y < 65:
        raise ValueError('First addendum page overflowed')
    footer(c, first)

    header(c, 'BUDGET TIMING AND REFERENCE SUPPORT', 'Why moving edits can hurt')
    y = paragraph(c, 'We allowed edits to move between two consecutive nonoverlapping windows from the '
                  'same recording. Both decoders could move the same slots and retain the exact count '
                  'and displacement. The scheduler sees both observed prefixes, so this is an offline '
                  'timing diagnostic. No future labels select its allocations.', 642)
    rows = [['Joint policy', 'Defined easy reference', 'Risk passes', 'Risk + both refs.']]
    for label, k in (('Fixed timing', 'timing_fixed_joint'), ('Pooled timing', 'guard_pooled_joint'),
                     ('Pooled, support guarded', 'guard_guarded_joint')):
        _, risk, dual, *easy = v[k]
        rows.append([label, f'{easy[0]}/606' if easy else 'not recomputed', f'{risk}/606', f'{dual}/606'])
    y = table(c, rows, [181, 121, 76, 110], y)
    risk = e['risk']['transitions']['joint']
    undefined = e['risk']['lost_pass_reasons']['joint']['completion_easy_risk_upper:undefined']
    y = paragraph(c, f'Pooling loses {risk["1_to_0"]} risk passes and gains {risk["0_to_1"]}. '
                  f'In {undefined} of the lost passes, the selected easy reference becomes undefined. '
                  'This does not mean the risk is below 2%; the denominator is absent. The pooled-minus-fixed '
                  f'utility difference is {interval(v["pooled_joint_minus_fixed"])}.', y)
    y = paragraph(c, 'A donor-trained model estimates easy reference mass from past motion. The guard '
                  'accepts a pooled plan only when its estimated mass is at least that of the fixed plan. '
                  'More views now have a measurable easy reference, but the guard loses utility relative to '
                  f'unguarded pooling: {interval(v["guarded_joint_minus_pooled"])}.', y)
    y = paragraph(c, 'These controls distinguish three questions: whether a better edit exists, whether '
                  'its risk can be estimated, and whether a learned rule improves forecasts. The '
                  'expanded menu helps the first; the support guard helps the second. Neither establishes '
                  'the third under the matched budget. The independent source roles and risk rule are unchanged.', y)
    if y < 65:
        raise ValueError('Second addendum page overflowed')
    footer(c, first+1)
    c.save()
    buf.seek(0)
    return buf


def verify(original, final):
    n = len(original.pages)
    if len(final.pages) != n+2:
        raise ValueError('The supplement must contain exactly two added pages')
    def annotations(page):
        return page['/Annots'] if '/Annots' in page else []

    for a, b in zip(original.pages, final.pages[:n], strict=True):
        if (a.extract_text() != b.extract_text() or a.get_contents().get_data() != b.get_contents().get_data()
                or len(annotations(a)) != len(annotations(b)) or a.mediabox != b.mediabox):
            raise ValueError('An original page changed')
    def dest(r):
        return {k: r.get_destination_page_number(v) for k, v in r.named_destinations.items()}
    if dest(original) != dest(final):
        raise ValueError('Named destinations changed')
    for a, b in zip(annotations(original.pages[0]), annotations(final.pages[0]), strict=True):
        x, y = a.get_object()['/Dest'], b.get_object()['/Dest']
        old_page = next(i for i, page in enumerate(original.pages) if page.indirect_reference == x[0])
        if final.pages[old_page].indirect_reference != y[0] or x[1:] != y[1:]:
            raise ValueError('An original evidence-guide link changed')
    for i in range(n, n+2):
        if annotations(final.pages[i]):
            raise ValueError('The new pages must be self-contained')
    outline = {v.title: final.get_destination_page_number(v) for v in final.outline if not isinstance(v, list)}
    if outline.get('Matched-budget diagnostics') != n:
        raise ValueError('New outline entry does not open the addendum')
    return dict(pages=n+2, original_pages=n, retained_named_destinations=len(dest(original)))


def build(output, replay):
    if output.exists() or replay.exists() or output.resolve() == SOURCE.resolve():
        raise FileExistsError('Refusing to overwrite a submission artifact')
    if digest(SOURCE) != SOURCE_SHA:
        raise ValueError('The existing supplement is not the verified guide edition')
    e = load()
    s = scalar_snapshot(e)
    original = PdfReader(SOURCE)
    w = PdfWriter(clone_from=original)
    for page in PdfReader(pages(s, e, len(original.pages)+1)).pages:
        w.add_page(page)
    w.add_outline_item('Matched-budget diagnostics', len(original.pages))
    w.add_metadata({'/Title': 'Supplementary material', '/Author': '', '/Subject': ''})
    tmp = output.with_suffix('.pending')
    w.write(tmp)
    receipt = verify(original, PdfReader(tmp))
    tmp.replace(output)
    verify_script = Path(__file__).with_name('verify_m3w_budget_addendum_scalars.py').read_bytes()
    readme = ('Run: python -I -B -S verify.py\n\nThis bundle replays the appended budget tables from anonymous '
              'source summaries and the original resampling indices. It does not rerun upstream inference, '
              'train models, or contain trajectories, checkpoints, or reserved outcomes.\n').encode()
    with ZipFile(replay, 'x', compression=ZIP_DEFLATED) as archive:
        for name, content in [('README.txt', readme), ('scalars.json', json.dumps(s, sort_keys=True).encode()),
                              ('verify.py', verify_script)]:
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, content)
    return dict(**receipt, supplement_sha256=digest(output), scalar_archive_sha256=digest(replay),
                original_sha256=SOURCE_SHA, upload_performed=False, submission_ready=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('replay', type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.output, a.replay), indent=2))
