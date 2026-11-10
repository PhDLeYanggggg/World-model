"""Package the focused paper's anonymous proof, readouts and offline replay."""

import argparse
from io import BytesIO
import json
from math import isclose
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement import moving_completion, verify
from scripts import audit_m3w_moving_completion_envelope as moving
from scripts import build_m3w_vehicle_supplement as layout
from scripts import package_m3w_combined_review_supplement as supplement


main, archive = supplement.main, supplement.archive
TEMPLATE = archive.ROOT/'reproducibility/focused_review_supplement'
OUT = archive.PUBLIC/'focused_review_supplement_v1'
DEST = supplement.LOCAL/'review_supplement_v3.zip'
DRAFT = main.PRIVATE/'evidence_manuscript_v172_focused/manuscript.pdf'


def original_members():
    previous = supplement.read_receipt('songdo_review_supplement_v1')
    v1 = json.loads((supplement.OUT/'verification.json').read_text())
    with ZipFile(BytesIO(supplement.checked(supplement.LOCAL/'review_supplement_v2.zip', previous['archive_sha256']))) as z:
        songdo = {k: z.read(k) for k in ('songdo_confirmation.pdf', 'songdo_replay.zip')}
        v1_bytes = z.read('original_evidence.zip')
    if main.digest(v1_bytes) != v1['sha256'] or previous['prior_bundle_sha256'] != v1['sha256']:
        raise ValueError('Original evidence lineage changed')
    if main.digest(songdo['songdo_confirmation.pdf']) != previous['pdf_sha256']:
        raise ValueError('Songdo addendum differs from the verified build')
    with ZipFile(BytesIO(v1_bytes)) as z:
        members = {k: z.read(k) for k in ('technical_appendix.pdf', 'reviewer_evidence.zip', 'moving_peak_certificate.zip')}
    for k, report in (('moving_peak_certificate.zip', 'moving_peak_certificate_v1'),
                      ('reviewer_evidence.zip', 'reviewer_evidence_v1')):
        if main.digest(members[k]) != supplement.read_receipt(report)['archive_sha256']:
            raise ValueError('Inherited scalar evidence changed: '+k)
    return members | songdo, previous, v1['headline']


def moving_extension(archived):
    with ZipFile(BytesIO(archived)) as z:
        raw = z.read('evidence.json')
    original = json.loads(raw)
    payload, expected, hashes = moving.prior.export(aggregate_view=moving.aggregate)
    fields = ('recording', 'source', 'seed', 'completion_queries')
    extensions, bases = [], []
    for row in payload['views']:
        bases.append({k: v for k, v in row.items() if k != 'completion_queries'})
        extra = {k: row[k] for k in fields}
        extra['completion_queries'] = sorted(extra['completion_queries'], key=lambda q: (
            q['actors'], q['budget'], q['favorable_ray'], q['favorable_lengths']))
        extensions.append(extra)
    if bases != original['views']:
        raise ValueError('Re-aggregated development views differ from the prior certificate')
    previous = moving.verify.replay(original)
    moving.prior.check(previous, expected)
    extra = dict(original_evidence_sha256=main.digest(raw), views=extensions)
    result = moving_completion.report(moving_completion.attach(original, extra), previous, moving.verify)
    stored = supplement.read_receipt('moving_completion_envelope_v1')
    if stored['archive_input_sha256'] != hashes or any(result[k] != stored[k] for k in result):
        raise ValueError('Anonymous completion replay disagrees with the verified result')
    return (json.dumps(extra, separators=(',', ':'), allow_nan=False)+'\n').encode(), result


def reading_guide(result):
    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=(612, 792), invariant=1)
    pdf.setAuthor('')
    pdf.setCreator('Anonymous')
    pdf.setTitle('Forecast replacement: evidence and replay')
    y = layout.header(pdf, 'SUPPLEMENTARY MATERIAL', 'Forecast replacement: evidence and replay')
    y = layout.para(pdf, 'The matched controls ask which forecasts to replace, not how many. '
                    'A useful candidate in hindsight, a proposal chosen from the past and a released '
                    'forecast are three different results.', y)
    y = layout.para(pdf, '1. What does the displacement budget permit?', y, bold=True)
    m = result['moving']['completion_envelope']
    n = result['moving']['views']
    rows = [['Moving development criterion', 'Fixed query budget', 'Pooled recording budget']]
    for key, label in zip(moving_completion.KINDS, ('Conservative stepwise score', 'Observed steps alone',
                                                  'Most favorable missing labels'), strict=True):
        rows.append([label, *[f"{m[s][key]['excluded_views']}/{n}" for s in ('query', 'record')]])
    y = layout.table(pdf, rows, [248, 111, 141], y)
    y = layout.para(pdf, 'Counts exclude positive gain over OLS4. The budget caps each agent\'s peak change '
                    'from CV, summed across agents. At a labeled step, radius r can recover at most '
                    'min(r, CV error). At an unlabeled step, distance d from CV to OLS4 gives the best '
                    'conservative contribution -max(d-r, 0) or the favorable upper contribution d+r. '
                    'Allocate budget in descending order of query-weighted marginal gains. This bound '
                    'drops edit counts and risk constraints; the 72 views repeat 24 recordings.', y, size=10, lead=13)
    y = layout.para(pdf, '2. Can a past-only rule find the useful replacements?', y, bold=True)
    o = result['original']['headline']['opendd']
    y = layout.para(pdf, f"On OpenDD development, {o['hindsight_passes']}/{o['views']} views have a verified "
                    f"label-aware plan under the restricted menu and chronological budget. The past-only count "
                    f"controller passes {o['past_only_passes']}/{o['views']}. The same 202 recordings are repeated "
                    'across three seeds; the difference is not independent replication.', y, size=10, lead=13)
    y = layout.para(pdf, '3. Does the release rule transfer?', y, bold=True)
    w = result['wepdtof']['contrasts']['fixed_independent']['identification_envelope_pp']
    s = result['songdo']['actual_risk']
    y = layout.para(pdf, f'On WEPDTOF the missing-label envelope is [{w[0]:+.5f}, {w[1]:+.5f}] pp '
                    'for joint minus independent, with 0/45 dual passes. The envelope is not a sampling '
                    f"interval. On Songdo, joint released {s['joint']['edited']}/480 views. Independent released "
                    f"{s['independent']['edited']}; {s['independent']['original_risk_pass']} met all original risk checks "
                    f"and {s['independent']['dual_pass']} also beat both motion references. Zero edits do not certify risk.",
                    y, size=10, lead=13)
    y = layout.para(pdf, 'Replay: extract the archive and run <font face="Courier">python3 -I -B -S verify.py</font>. '
                    'The technical appendix and Songdo sheet retain the original scoring and protocol details.',
                    y, size=9.5, lead=12, color=layout.MUTED)
    if y < 55:
        raise ValueError('Reading guide exceeds one page')
    pdf.showPage()
    pdf.save()
    data = buf.getvalue()
    text = supplement.check_pdf(data, 1)
    if not all(k in text for k in ('67/72', '36/72', '9/72', '330/606', '57/606', '0/45', '0/480')):
        raise ValueError('The guide lost a headline count')
    return data


def package(destination):
    inherited, prior_songdo, headline = original_members()
    draft = supplement.read_receipt('evidence_manuscript_v172_focused')
    sha = draft['artifacts']['manuscript.pdf']
    supplement.check_pdf(supplement.checked(DRAFT, sha), draft['pages'])
    supplement.check_pdf(inherited['technical_appendix.pdf'], 50, {supplement.MANTRA})
    supplement.check_pdf(inherited['songdo_confirmation.pdf'], 1)
    extra, expected_moving = moving_extension(inherited['moving_peak_certificate.zip'])
    items = inherited | {k: (TEMPLATE/k).read_bytes() for k in (
        'README.md', 'verify.py', 'moving_completion.py', 'wepdtof_completion.py')}
    items['moving_completion.json'] = extra
    if any(archive.PRIVATE_MARKERS.search(raw) or b'moving_source_component_' in raw
           for name, raw in items.items() if Path(name).suffix in ('.md', '.py', '.json')):
        raise ValueError('Identifying data in the reviewer bundle')
    with TemporaryDirectory(prefix='focused-review-') as temp:
        root = Path(temp)
        # Bootstrap the guide from the offline replay, not a separate headline table.
        items['reading_guide.pdf'] = b''

        def assemble(path):
            if set(items) != verify.MEMBERS:
                raise ValueError('Unexpected review members')
            archive.encode(path, items | {'manifest.json': main.as_json({k: main.digest(v) for k, v in items.items()})})

        first = root/'draft.zip'
        assemble(first)
        verify.unpack(first, root/'draft')
        result = verify.run_replay(root/'draft')
        if (result['moving'] != expected_moving or result['original']['headline'] != headline
                or result['songdo']['actual_risk'] != prior_songdo['original_released_risk']
                or result['songdo']['contrasts']['independent']['lower_pp'] != prior_songdo['joint_minus_independent_lower_pp']):
            raise ValueError('Offline replay changed the verified headline, Moving or Songdo results')
        expected_wep = supplement.read_receipt('wepdtof_missing_label_interval_v1')['comparisons']
        for arm, row in result['wepdtof']['contrasts'].items():
            for k in ('identification_envelope_pp', 'empirical_bootstrap_outer_pp'):
                if not all(isclose(a, b, abs_tol=1e-12) for a, b in zip(row[k], expected_wep[arm][k], strict=True)):
                    raise ValueError('Offline WEPDTOF completion result differs from the verified receipt')
        items['reading_guide.pdf'] = reading_guide(result)
        a, b = root/'final.zip', root/'rebuilt.zip'
        assemble(a)
        assemble(b)
        raw = a.read_bytes()
        if raw != b.read_bytes():
            raise ValueError('Review package was not deterministic')
        verify.unpack(a, root/'final')
        if verify.run_replay(root/'final') != result:
            raise ValueError('Delivered archive does not replay')
    receipt = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), main_pdf_sha256=sha,
                   guide_sha256=main.digest(items['reading_guide.pdf']), prior_bundle_sha256=prior_songdo['archive_sha256'],
                   moving=result['moving']['completion_envelope'], wepdtof=result['wepdtof'], headline=headline,
                   songdo_actual_release=result['songdo']['actual_risk'], clean_stdlib_replay=True, deterministic=True,
                   external_labels_reopened=False, original_2pct_unchanged=True, uploaded=False, submission_ready=False)
    files = {destination: raw, destination.with_name('focused_review_guide_v1.pdf'): items['reading_guide.pdf'],
             OUT/'receipt.json': main.as_json(receipt)}
    if any(p.exists() and p.read_bytes() != v for p, v in files.items()):
        raise FileExistsError('Refusing to overwrite a different reviewer artifact')
    for p, v in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(json.dumps({k: receipt[k] for k in ('archive_sha256', 'archive_bytes', 'guide_sha256', 'clean_stdlib_replay')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEST)
    moving.prior.audit.torch.set_num_threads(4)
    moving.prior.audit.torch.set_num_interop_threads(1)
    package(p.parse_args().output)
