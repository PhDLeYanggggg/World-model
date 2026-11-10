"""Build a standalone anonymous supplement paired with the verified main PDF."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.m3w_reader_supplement.verify import check_risk
from scripts import build_m3w_vehicle_supplement as layout

PUBLIC = ROOT/'outputs/publication_readiness_2026_09'
LOCAL = ROOT/'data/stage_cvpr2027_experiments'
OUT = PUBLIC/'reader_supplement_v1'
MAIN = 'evidence_manuscript_v208_shared_seed'
ADMIN = LOCAL/'submission_admin'
TEMPLATE = ROOT/'reproducibility/m3w_reader_supplement'
SOURCES = dict(moving='moving_completion_envelope_v1/receipt.json',
               opendd='opendd_direct_completion_signs_v1/readout.json',
               wep='wepdtof_missing_label_interval_v1/receipt.json',
               songdo='songdo_calibrated_20261110/confirmation_receipt.json',
               shared='songdo_shared_seed_completion_v1/readout.json',
               risk='songdo_full_grid_all_risk_v1/readout.json',
               cases='songdo_failure_cases_v1/review_receipt.json')
PRIVATE = re.compile(r'/Users/|/home/|/cephfs/|https?://|github\.com|kcl\.ac\.uk|yangyue|Le\s+Yang|rdb[1-7]_', re.I)


def digest(value):
    return sha256(value).hexdigest()


def checked(path, expect):
    raw = path.read_bytes()
    if digest(raw) != expect:
        raise ValueError(f'Artifact changed: {path.name}')
    return raw


def load():
    raw = {k: (PUBLIC/v).read_bytes() for k, v in SOURCES.items()}
    d = {k: json.loads(v) for k, v in raw.items()}
    a, b = d['songdo'], d['risk']
    if (not a['no_fitting_on_confirmation'] or not b['original_2pct_unchanged']
            or not b['original_scored_outcomes_reused'] or b['new_external_cohort_opened']
            or any(b[k] for k in ('calibration_outcomes_read', 'deployment_changed', 'fit'))
            or any(b['binding'][k] != a['binding'][k] for k in ('primary_score_sha256', 'flight_score_hash_root'))):
        raise ValueError('Wrong retrospective Songdo sensitivity')
    for arm, (match, _, releases) in check_risk(b).items():
        if match != a['matched'][arm]['risk']['original_risk_pass'] or releases != a['released'][arm]['risk']['edited']:
            raise ValueError('Original pass and release counts disagree')
    return d, {k: digest(v) for k, v in raw.items()}


def finish(pdf, page, y):
    if y < 63:
        raise ValueError('Content runs into the footer')
    pdf.setStrokeColor(layout.RULE)
    pdf.line(layout.LEFT, 49, layout.LEFT+layout.WIDTH, 49)
    pdf.setFont('Helvetica', 9)
    pdf.setFillColor(layout.MUTED)
    pdf.drawString(layout.LEFT, 34, 'Forecast replacement | Supplement')
    pdf.drawRightString(layout.LEFT+layout.WIDTH, 34, str(page))
    pdf.showPage()


def render(d, case_pdf):
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(612, 792), invariant=1)
    pdf.setAuthor('Anonymous')
    pdf.setTitle('When Should a Neural Forecast Replace a Motion Baseline? Supplement')
    para, table, header = layout.para, layout.table, layout.header
    y = header(pdf, 'SUPPLEMENTARY MATERIAL', 'Reading the replacement experiments')
    y = para(pdf, 'Each query contains eight observed and twelve forecast steps. We compare two decisions: '
             'which fixed forecasts to propose at a common budget, and whether to release a proposal under '
             'the original recording-level 2% checks. No-edit CV remains a reference, not a passed risk check.', y)
    y = table(pdf, [['Dataset', 'Role', 'Independent grouping'],
                    ['MovingDroneCrowd++', 'Development and capacity', '14 source groups on main roster'],
                    ['OpenDD', 'Development and matched controls', '7 held locations'],
                    ['DroneCrowd', 'Calibration only', '15 exclusion groups'],
                    ['WEPDTOF', 'Frozen evaluation', '13 sources, 10 estimable for paired utility'],
                    ['Songdo Traffic', '40 calibration / 160 confirmation flights', '14 confirmation source components']],
              [124, 185, 191], y)
    y = para(pdf, '<b>Matched comparisons.</b> Candidate forecasts, queries and counts are held fixed within a comparison. '
             'Displacement is matched in the specified sensitivity analyses. Independent ranks candidate changes '
             'by actor; same-family constrains the edit family; additive drops the relative-path cross term; '
             'joint includes it. Calibration and held outcomes are never mixed.', y)
    y = para(pdf, '<b>Missing labels.</b> Let K be the paired gain on observed steps. Let D be the sum of weighted '
             'forecast disagreements on missing steps, using the same observed CV denominator. The relaxed '
             'sample envelope is [K - D, K + D]. It does not impute trajectories. The bootstrap varies '
             'source composition, not missing positions. Seeds and overlapping windows do not enlarge '
             'the independent source count.', y)
    finish(pdf, 1, y)

    y = header(pdf, 'DEVELOPMENT EVIDENCE', 'Capacity, ranking and label support')
    m = d['moving']['completion_envelope']['record']
    y = para(pdf, 'Allowing arbitrary trajectory shapes under the observed displacement budget tests '
             'whether the bottleneck is the forecast bank. Pooling movement over an entire recording '
             'relaxes the query-level allocation.', y)
    y = table(pdf, [['Moving pooled displacement score', 'Views unable to beat OLS4'],
                    *[[label, f"{m[k]['excluded_views']}/72"] for k, label in (
                        ('unfavorable_completion', 'Conservative missing-label endpoint'),
                        ('observed_steps_only', 'Observed steps alone'),
                        ('favorable_completion', 'Most favorable missing positions'))]], [315, 185], y)
    r = d['opendd']['results']
    rows = [['OpenDD contrast', 'Observed', 'Completion envelope']]
    for key, label in (('support_count', 'Support vs original; count'),
                       ('support_count_movement', 'Support vs original; count and movement'),
                       ('learned_minus_profile', 'Learned vs average missing-label penalty')):
        s = r[key]['summary']
        o, l, u = (1000*s[x]['point'] for x in ('observed', 'lower', 'upper'))
        rows.append([label, f'{o:+.3f}', f'[{l:+.3f}, {u:+.3f}]'])
    y = table(pdf, rows, [260, 88, 152], y)
    y = para(pdf, 'OpenDD units are 0.001 percentage points of the observed CV error. All sample envelopes '
             'span zero after dropping any one of the seven locations. The conservative criterion does not '
             'favor these switches; their unrestricted complete-label signs are not identified. The original '
             'edit budget, model seeds, and role assignments are unchanged.', y)
    finish(pdf, 2, y)

    y = header(pdf, 'EXTERNAL EVIDENCE', 'Frozen comparisons and actual releases')
    w = d['wep']['comparisons']['fixed_independent']['identification_envelope_pp']
    s = d['shared']['comparisons']['independent']['endpoints']
    y = para(pdf, f'WEPDTOF: joint minus independent has a sample envelope [{w[0]:+.5f}, {w[1]:+.5f}] pp '
             'on the ten estimable sources. The fixed joint rule gives 0/45 dual passes. On Songdo, '
             f'the shared-seed upper endpoint is {s["shared_upper"]["point"]:+.6f} pp; its source '
             'interval includes zero. The observed sample and extrapolation to new sources are distinct.', y)
    y = table(pdf, [['Songdo arm', 'Original three checks / 480', 'Also passes full-grid all-edited', 'Actual releases'],
                    *[[a.replace('_', ' ').capitalize(), *map(str, v)] for a, v in check_risk(d['risk']).items()]],
              [102, 136, 169, 93], y)
    y = para(pdf, '<b>Full-grid sensitivity, not a replacement rule.</b> The original all-edited risk ratio '
             'uses mean observed-step errors per track and charges maximum disagreement on wholly unknown '
             'tracks. For the twelve-step diagnostic, sum deterioration over steps first, take its positive '
             'part per edited track, then divide by total edited CV error. A missing target at CV attains '
             'both minimum CV error and maximum deterioration. Thus it attains the worst-case ratio '
             'whenever the observed denominator is positive. The easy subset and its rules are unchanged.', y)
    y = para(pdf, 'All 15 actual independent-policy releases have complete edited labels. Five satisfy '
             'all three checks; four also improve over both motion references. Joint releases none. '
             'The sensitivity cannot turn a matched proposal or an abstention into an actual release.', y)
    y = para(pdf, 'The final page shows three lower-median retrospective Songdo cases, using one seed '
             'and no identifying coordinates or vehicle labels. The shaded range is due to missing targets, '
             'not a confidence interval. In the accompanying archive, run <b>python verify.py</b> for '
             'offline aggregate replay. It does not retrain or read source trajectories.', y)
    finish(pdf, 3, y)
    pdf.save()
    writer = PdfWriter()
    for raw in (buf.getvalue(), case_pdf):
        writer.append(PdfReader(BytesIO(raw)))
    writer.add_metadata({'/Author': 'Anonymous', '/Title': 'Forecast replacement: supplementary evidence'})
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def archive_bytes(parts):
    buf = BytesIO()
    with ZipFile(buf, 'w', compression=ZIP_DEFLATED, compresslevel=9) as z:
        for k, v in sorted(parts.items()):
            z.writestr(ZipInfo(k), v, compress_type=ZIP_DEFLATED, compresslevel=9)
    return buf.getvalue()


def as_json(v):
    return (json.dumps(v, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def main():
    d, hashes = load()
    previous = json.loads((PUBLIC/MAIN/'review_receipt.json').read_bytes())
    checked(LOCAL/MAIN/'manuscript.pdf', previous['companion_pdf_sha256'])
    old = checked(ADMIN/'review_supplement_v40.zip', previous['archive_sha256'])
    case = checked(LOCAL/'songdo_failure_cases_v1/failure_cases.pdf', d['cases']['figure_sha256'])
    pdf = render(d, case)
    reader = PdfReader(BytesIO(pdf))
    extracted = '\n'.join(p.extract_text() for p in reader.pages)
    if (len(reader.pages) != 4 or any(p.get('/Annots') for p in reader.pages)
            or PRIVATE.search(extracted) or PRIVATE.search(str(reader.metadata))
            or any(x not in extracted for x in ('179', '176', 'Full-grid sensitivity', 'Incomplete future labels'))):
        raise ValueError('Unexpected text, links or identity in the reader PDF')
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'supplement.pdf': pdf, 'risk.json': as_json(d['risk']), 'prior_evidence.zip': old})
    if PRIVATE.search(parts['risk.json'].decode()):
        raise ValueError('Identifying content in risk readout')
    parts['manifest.json'] = as_json({k: digest(v) for k, v in parts.items()})
    payload = archive_bytes(parts)
    if payload != archive_bytes(parts):
        raise ValueError('Nondeterministic supplement')
    with TemporaryDirectory(prefix='reader-supplement-') as tmp:
        tmp = Path(tmp)
        p = tmp/'review.zip'
        p.write_bytes(payload)
        unpack(p, tmp/'unpacked')
        if run_replay(tmp/'unpacked')['full_grid_all_risk'] != check_risk(d['risk']):
            raise ValueError('Risk table did not replay')
    receipt = dict(companion_pdf_sha256=previous['companion_pdf_sha256'], pdf_sha256=digest(pdf), pages=4,
                   archive_sha256=digest(payload), prior_archive_sha256=previous['archive_sha256'],
                   archive_bytes=len(payload), input_sha256=hashes, original_2pct=True, uploaded=False,
                   submission_ready=False, includes_raw_labels=False, includes_checkpoints=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for p, raw in ((ADMIN/'review_supplement_v41.pdf', pdf), (ADMIN/'review_supplement_v41.zip', payload),
                   (OUT/'receipt.json', as_json(receipt))):
        if p.exists() and p.read_bytes() != raw:
            raise FileExistsError('Refusing to overwrite a frozen supplement')
        if not p.exists():
            with p.open('xb') as f:
                f.write(raw)
    print(receipt)


if __name__ == '__main__':
    main()
