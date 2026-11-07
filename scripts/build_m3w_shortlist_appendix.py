"""Build a short evidence appendix from the verified frozen shortlist studies."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_m3w_native_paper_v10 import latex_document,sha,write_json
from scripts.build_m3w_utility_tradeoff_appendix import LEDGER,OUT

BUILD = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/shortlist_build'


def read(p):
    return json.loads(p.read_text())


def main():
    hashes = {}
    def checked(folder,name='analysis.json'):
        root = LEDGER/folder
        src = root/'analysis.json'
        proof = root/'replay_execution.json'
        if read(proof)['artifact']['sha256']!=sha(src):
            raise ValueError('Require replayed frozen results')
        p = root/name
        data = read(p)
        if name!='analysis.json' and data['analysis_sha256']!=sha(src):
            raise ValueError('Diagnostic must bind to the registered readout')
        hashes.update({str(v.relative_to(ROOT)):sha(v) for v in (src,proof,p)})
        return data
    ranked = checked('opening_pair_rank')
    result = checked('shortlist_cost')
    diagnostic = checked('shortlist_cost','event_choice_diagnostic.json')
    previous = checked('opening_stop_wait','rank_recall.json')
    first = result['recall']['first_two']
    table = []
    for name,n in [('Whole bank',previous['counts']['first_two']['bank']),
                   ('Flat previous shortlist',ranked['old_recall']['first_two']['joint_top8']),
                   ('Family-diverse shortlist',first['shortlist']),
                   ('Gain-only winner',first['gain']),('Risk-penalized winner',first['risk']),
                   ('Event-score winner',diagnostic['recall']['first_two'])]:
        table.append(f'| {name} | {n} / 18 |')
    risk = result['risk']
    v = result['contrasts']['diagnostic_risk_vs_gain']
    lo,hi = v['interval95']
    text = f'''# From Candidate Availability to Observed Intervention

## Controlled Setup

We keep the forecast bank, direct policy, and original 2% recording risk
and reference-support rules fixed on exposed Moving A. Each of 42 cells
uses seven fitting sources, six calibration sources and one held source.
The 102 views repeat 34 recordings over three forecaster seeds. The task
uses eight observed and twelve forecast annotation steps in image coordinates.

### Table 1. Positive supported actions in the first two eligible queries

| Available actions or final choice | Zero-direct views hit |
| --- | ---: |
''' + '\n'.join(table) + f'''

These are descriptive counts over the 18 original zero-direct views.
A hit requires a strictly positive ADE gain and a standalone pass under
the unchanged observed rule. A shortlist hit still needs a decision
rule to choose the action. The event-score winner is a posthoc readout
of the same fitted model, not an additional calibrated policy.

## Choice At Fixed Coverage

The fixed shortlist puts the highest original-score candidate from
each family first, then fills the remaining slots up to eight. A ridge
model trained on donor-excluded fitting shortlists estimates utility,
risk costs and the useful-action event. The gain and risk choices see
the same shortlist and predicted outputs. In the unthresholded
first-query control, both make {risk['diagnostic_risk']['switches']:,}
switches at identical queries. Risk minus gain changes paired
completion-lower utility by {v['point']:+.5f} percentage points
(95% source interval [{lo:+.5f}, {hi:+.5f}]). Both have
{risk['diagnostic_risk']['passes']}/102 complete passes in this diagnostic.
With donor-only threshold selection, the gain and risk arms have
{risk['gain']['passes']} and {risk['risk']['passes']} complete passes,
versus {risk['direct']['passes']} for direct. These do not support
improved calibrated intervention.

## Relation To Deferral

[Mao et al. (ICML 2024)](https://proceedings.mlr.press/v235/mao24d.html)
already study multi-expert regression deferral with fixed predictors.
Here we isolate candidate availability, an eight-slot shortlist, and
final choice under the original trajectory-risk contract. We do not
implement their surrogate or claim their consistency guarantees.
The gap between a useful action being available and being chosen
remains the central empirical limitation. No independent confirmation
or deployment follows from these results.
'''
    tex = latex_document(text).replace('[10pt,twocolumn,letterpaper]','[10pt,onecolumn,letterpaper]')
    BUILD.mkdir(parents=True,exist_ok=True)
    (OUT/'shortlist_appendix.md').write_text(text)
    (OUT/'shortlist_appendix.tex').write_text(tex)
    (BUILD/'main.tex').write_text(tex)
    shutil.copyfile(BUILD.parent/'paper_build/cvpr.sty',BUILD/'cvpr.sty')
    write_json(OUT/'shortlist_appendix_bindings.json',dict(source_sha256=hashes,
        builder_sha256=sha(Path(__file__)),tex_sha256=sha(OUT/'shortlist_appendix.tex'),
        independent_confirmation=False,submission_ready=False,new_outcome_readout=False))


if __name__=='__main__':
    main()
