"""Build a separate manuscript candidate with the fixed-recipient vehicle evidence."""

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_verified_references_candidate as prior


main, supplement, focused, scope = prior.main, prior.supplement, prior.focused, prior.scope
OUT = prior.EXCHANGE_OUT.parent/'evidence_manuscript_v180_vehicle_direction'
RECEIPT = main.PUBLIC/OUT.name/'receipt.json'
FILES = ('opendd_vehicle_angular_pair_v1', 'opendd_vehicle_single_rotation_v1')
ANCHOR = 'the sign. Neither result shows a benefit from adding the cross term.'


def inputs():
    readout, hashes = [], {}
    for name in FILES:
        directory = main.PUBLIC/name
        raw = (directory/'readout.json').read_bytes()
        d = json.loads(raw)
        if (d['registration'] != main.digest((directory/'registration.json').read_bytes()) or
                d['external_outcomes_read'] or d['independent_confirmation'] or not d['original_2pct']):
            raise ValueError('Unexpected vehicle experiment roles')
        readout.append(d)
        hashes[name] = main.digest(raw)
    return readout, hashes


def revise(source, rows):
    forced, selective = rows
    a, b = forced['summary'], selective['summary']
    if (a['counts']['views'], b['counts']['views'], forced['unique_recordings'], selective['unique_recordings']) != (8874, 8874, 493, 493):
        raise ValueError('Vehicle rosters do not match')
    if a['arms']['original']['dual'] != b['arms']['original']['dual'] or source.count(ANCHOR) != 1:
        raise ValueError('Original forecast control or manuscript anchor changed')
    c = a['counts']
    d = a['paired']['coupled_minus_independent']['pair_lower']
    e = b['paired']['interaction_minus_independent_at_interaction']['pair_lower']
    n = b['arms']['interaction']['rotated_views']
    if n != b['arms']['independent_at_interaction']['rotated_views']:
        raise ValueError('Selective comparison is not rotation-matched')
    text = rf'''

\paragraph{{Coordinating direction without changing recipients.}}
In {forced['unique_recordings']} OpenDD vehicle recordings ({c['views']:,} repeated views),
we keep the two additive-selected actors and their per-step displacement
from CV. Among {c['any_relative_win']:,} views with a hindsight direction change
that passes risk, preserves trajectory utility, and improves relative
motion, {c['single_rotation_relative_win']:,} allow changing just one actor.
Forcing two nonzero angles, the causal coupling term loses
\({abs(d['point']):.4f}\) native units of conservative relative-motion gain
against independent angles (95\% location interval
[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]). Allowing no rotation or one
rotation reduces the difference, but it remains negative at the same
{n} chosen actors (\({e['point']:.6f}\) native units). The extra freedom in the
two-angle search is therefore not evidence that a coupled decision
improves the forecast from the past.
'''
    return source.replace(ANCHOR, ANCHOR+text, 1), (f"{c['any_relative_win']:,}", f"{c['single_rotation_relative_win']:,}",
                                                    f"{abs(d['point']):.4f}", f'{n} chosen actors')


def build(destination, compiler, receipt, *, parent_dir=prior.EXCHANGE_OUT, read_inputs=inputs, revise_source=revise,
          extra_assets=None):
    parent = supplement.read_receipt(parent_dir.name)
    raw = {k: supplement.checked(parent_dir/k, v) for k, v in parent['artifacts'].items()}
    with ZipFile(BytesIO(raw['main_source.zip'])) as z:
        files = {k: z.read(k) for k in z.namelist()}
        if len(files) != len(z.namelist()) or any(Path(k).name != k for k in files):
            raise ValueError('Expected a flat archive')
    if files['manuscript.tex'] != raw['manuscript.tex']:
        raise ValueError('Parent archive and manuscript differ')
    original = files['manuscript.tex'].decode('ascii')
    citations = scope.citation.used_citations(files)
    rows, hashes = read_inputs()
    revised, markers = revise_source(original, rows)
    files['manuscript.tex'] = revised.encode('ascii')
    extra_assets = extra_assets or {}
    if set(extra_assets) & files.keys() or any(Path(k).name != k or not k.endswith('.pdf') for k in extra_assets):
        raise ValueError('Extra assets must be new flat PDF members')
    files.update(extra_assets)
    if scope.citation.used_citations(files) != citations:
        raise ValueError('Bibliography changed unexpectedly')
    style = supplement.checked(scope.citation.main.DRAFT/'cvpr.sty', scope.citation.STYLE_SHA)
    with TemporaryDirectory(prefix='vehicle-direction-paper-') as directory:
        root = Path(directory)
        for k, data in {**files, 'cvpr.sty': style}.items():
            (root/k).write_bytes(data)
        p = subprocess.run([str(compiler), '-X', 'compile', '--untrusted', '--only-cached', '--keep-logs', 'manuscript.tex'],
            cwd=root, env={**os.environ, 'SOURCE_DATE_EPOCH': '0'}, capture_output=True, text=True, timeout=120)
        if p.returncode:
            raise RuntimeError(p.stdout+'\n'+p.stderr)
        if re.search(r'(?i)undefined (citation|reference)|Citation .* undefined|Overfull \\hbox',
                     (root/'manuscript.log').read_text()):
            raise ValueError('Unresolved citation, reference, or overfull line')
        pdf = (root/'manuscript.pdf').read_bytes()
        pages = focused.check_render(pdf, raw['manuscript.pdf'], ('181/606', '197/480', *markers))
        scope.citation.archive.encode(root/'main_source.zip', files)
        output = {'manuscript.tex': files['manuscript.tex'], 'manuscript.pdf': pdf,
                  'main_source.zip': (root/'main_source.zip').read_bytes()}
    result = dict(pages=pages, prior_manuscript_sha256=parent['artifacts']['manuscript.pdf'], source_readout_sha256=hashes,
        artifacts={k: main.digest(v) for k, v in output.items()}, only_changed_archive_member='manuscript.tex',
        title_abstract_disclosure_unchanged=True, external_outcomes_unchanged=True, uploaded=False, submission_ready=False)
    if extra_assets:
        result['added_archive_assets_sha256'] = {k: main.digest(v) for k, v in extra_assets.items()}
    deliver = {**{destination/k: v for k, v in output.items()}, receipt: main.as_json(result)}
    if any(k.exists() and k.read_bytes() != v for k, v in deliver.items()):
        raise FileExistsError('Refusing to replace a different candidate artifact')
    for k, v in deliver.items():
        k.parent.mkdir(parents=True, exist_ok=True)
        if not k.exists():
            with k.open('xb') as f:
                f.write(v)
    print(result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--receipt', type=Path, default=RECEIPT)
    args = p.parse_args()
    build(args.output, args.compiler.resolve(), args.receipt)
