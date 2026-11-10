"""Build the condensed v207 results and the two-sided completion replay."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproducibility.focused_review_supplement.verify import run_replay, unpack
from reproducibility.opendd_direct_completion_supplement import verify
from scripts import audit_m3w_opendd_completion_signs as study
from scripts import build_m3w_step_visibility_candidate as prior

archive, bundle, main, builder = prior.archive, prior.bundle, prior.main, prior.builder
OUT = prior.OUT.parent/'evidence_manuscript_v207_completion_signs'
PUBLIC = main.PUBLIC/OUT.name
TEMPLATE = archive.ROOT/'reproducibility/opendd_direct_completion_supplement'
DEST = bundle.LOCAL/'review_supplement_v39.zip'


def inputs():
    d, result, h = study.data()
    r = (study.OUT/'readout.json').read_bytes()
    v = json.loads(r)
    if (v['results'] != result or v['hashes'] != h or not v['original_2pct']
            or any(v[k] for k in ('fit', 'raw_labels_read', 'new_external_labels_opened', 'deployment_changed'))):
        raise ValueError('Expected the verified two-sided result')
    return d, result, dict(**h, readout=main.digest(r), protocol=main.digest((study.OUT/'protocol.md').read_bytes()))


def revise(text, rows):
    start, stop = (r'\paragraph{Changing the recipient at the same budget.}',
                   r'\section{What Transfers to Other Recordings?}')
    proof = 'intervals. Source resampling measures a different uncertainty.'
    if any(text.count(s) != 1 for s in (start, stop, proof)):
        raise ValueError('Require the original v206 results and derivation')
    tab = []
    for label, key in (('Support: count', 'support_count'), ('Support: count and movement', 'support_count_movement'),
                       ('Learned vs. step profile', 'learned_minus_profile')):
        s = rows[key]['summary']
        o, l, u = (1000*s[k]['point'] for k in ('observed', 'lower', 'upper'))
        tab.append(f'{label} & {o:+.3f} & [{l:+.3f}, {u:+.3f}] '+r'\\')
    block = r'''\paragraph{Choosing whom to edit.}
In the matched OpenDD swap study, only 145 of the 2,624 views with a
qualifying hindsight swap are in the frozen 353-swap mask. Moving the
same quota toward predicted opportunities raises qualifying swaps from
54 to 68, but conservative relative motion worsens. The same nominations
allow 190 qualifying swaps in hindsight. The supplement separates the
missing opportunities from ranking errors and reports the frozen
release comparisons.

\paragraph{Does predicting label support help?}
Holding both paths fixed per recording, support-weighted merit yields
752 dual passes versus 710 for original merit at 2,196 selected views
each. The location interval for the rate change is [-0.67, +4.33] pp.
After also matching total and query-weighted movement to within 1\%,
support-weighted merit improves observed ADE but loses under the
conservative score. A learned stepwise missing-label model reduces
Brier score by 0.00515 and debit MAE by 0.00527 versus training-site
step averages. Replacing the step-average penalty with the learned
penalty nevertheless reduces conservative gain at the same budgets.

\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrr@{}}
\toprule
OpenDD contrast & Observed & Completion envelope \\
\midrule
'''+ '\n'.join(tab)+r'''
\bottomrule
\end{tabular}
\caption{Direct trajectory gains in units of 0.001 pp. The support rows
compare to original merit. The final row matches both budgets.
Envelopes vary missing labels within the sample; they are not confidence intervals.}
\label{tab:opendd-completion}
\end{table}

\paragraph{Does the missing part change the conclusion?}
All three sample envelopes in Table~\ref{tab:opendd-completion} span
zero, even after dropping any one of the seven locations. For the
movement-matched support comparison, the lower endpoint's location
interval is $[-0.798,-0.072]$ and the upper endpoint's is
$[+0.402,+3.422]$ in $10^{-3}$ pp. Thus the conservative criterion
does not favor replacing the original rule, but the sign of the
unrestricted full-grid effect remains unidentified.

'''
    text = text[:text.index(start)]+block+text[text.index(stop):]
    text = text.replace(proof, proof+r'''
Both endpoints are attainable if missing positions are allowed to
vary independently: set each unknown target to the treatment forecast
for the upper endpoint or to the reference for the lower endpoint.
This sharpness does not require a physically coherent completion.''')
    return text, ('1.570', 'Completion envelope', '0.00515')


def package():
    data, result, hashes = inputs()
    paper = bundle.read_receipt(OUT.name)
    pdf = paper['artifacts']['manuscript.pdf']
    parent = json.loads((prior.PUBLIC/'review_receipt.json').read_bytes())
    if paper['source_readout_sha256'] != hashes or paper['prior_manuscript_sha256'] != parent['companion_pdf_sha256']:
        raise ValueError('Paper and inherited replay mismatch')
    bundle.checked(OUT/'manuscript.pdf', pdf)
    parts = {k: (TEMPLATE/k).read_bytes() for k in ('README.md', 'verify.py')}
    parts.update({'protocol.md': bundle.checked(study.OUT/'protocol.md', hashes['protocol']),
                  'completion.py': Path(study.completion.__code__.co_filename).read_bytes(),
                  'points.json': main.as_json(data)})
    if any(archive.PRIVATE_MARKERS.search(v) for v in parts.values()):
        raise ValueError('Identifying material in anonymous addendum')
    parts['prior_evidence.zip'] = bundle.checked(prior.DEST, parent['archive_sha256'])
    if set(parts) != verify.MEMBERS:
        raise ValueError('Unexpected replay contents')
    parts['manifest.json'] = main.as_json({k: main.digest(v) for k, v in parts.items()})
    with TemporaryDirectory(prefix='opendd-completion-review-') as tmp:
        root = Path(tmp)
        for n in ('first.zip', 'second.zip'):
            archive.encode(root/n, parts)
        raw = (root/'first.zip').read_bytes()
        if raw != (root/'second.zip').read_bytes():
            raise ValueError('Nondeterministic archive')
        unpack(root/'first.zip', root/'unpacked')
        if run_replay(root/'unpacked')['opendd_completion_signs'] != result:
            raise ValueError('Anonymous completion replay disagrees with the readout')
    receipt = dict(archive_sha256=main.digest(raw), archive_bytes=len(raw), companion_pdf_sha256=pdf,
        prior_bundle_sha256=parent['archive_sha256'], source_readout_sha256=hashes['readout'],
        standard_library_replay=True, includes_raw_labels=False, uploaded=False, submission_ready=False)
    for p, v in ((DEST, raw), (PUBLIC/'review_receipt.json', main.as_json(receipt))):
        if p.exists() and p.read_bytes() != v:
            raise FileExistsError('Refusing to replace a frozen review artifact')
        if not p.exists():
            with p.open('xb') as f:
                f.write(v)
    print(receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    a = p.parse_args()
    builder.build(OUT, a.compiler.resolve(), PUBLIC/'receipt.json', parent_dir=prior.OUT,
                  read_inputs=lambda: inputs()[1:], revise_source=revise)
    package()
