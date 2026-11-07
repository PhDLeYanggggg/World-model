"""Build the development-only insertion evidence appendix from replayed aggregates."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_m3w_native_paper_v10 import latex_document, sha, write_json
from scripts.build_m3w_utility_tradeoff_appendix import LEDGER, OUT

BUILD = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/insertion_appendix_build'


def main():
    sources = {}

    def read(p):
        sources[str(p.relative_to(ROOT))] = sha(p)
        return json.loads(p.read_text())

    def checked(name):
        a = LEDGER/name/'analysis.json'
        v = read(a)
        for phase in ('score', 'replay'):
            if read(a.with_name(f'{phase}_execution.json'))['artifact']['sha256'] != sha(a):
                raise ValueError('Require exact frozen score replay')
        if v['independent_confirmation'] or v['deployment_changed'] or not v['original_rules_unchanged']:
            raise ValueError('Require exposed development without policy advancement')
        return v

    hurdle, rescue, swap, optional, boxes = map(checked, (
        'reference_hurdle', 'record_rescue', 'atomic_swap', 'optional_withdrawal', 'box_visibility'))
    base = boxes['risk']['direct']
    if any(any(v['risk']['direct'][k] != base[k] for k in ('passes', 'switches', 'undefined_easy', 'easy_net_over_2'))
           for v in (hurdle, rescue, swap, optional)):
        raise ValueError('Incompatible direct policies')
    diagnostic = read(LEDGER/'box_visibility/probe_diagnostic.json')
    if diagnostic['analysis_sha256'] != sha(LEDGER/'box_visibility/analysis.json'):
        raise ValueError('Changed box probe diagnostic')
    cap = read(LEDGER/'atomic_swap/ordered_capacity.json')['views']
    bounds = checked('support_acquisition')['groups']['all_missing']['modes']
    n = cap['missing_reference_views']
    table = []
    for name, v, left, right, key in (
        ('Factorized value', hurdle, 'direct_value', 'hurdle', 'first_nonnegative_supported_gain'),
        ('Whole-record target', rescue, 'local', 'record', 'first_rescue'),
        ('Joint vs additive exchange', swap, 'additive', 'joint', 'first_rescues'),
        ('Optional withdrawal', optional, 'matched_insert', 'optional', None),
        ('Observed head boxes', boxes, 'trajectory', 'trajectory_boxes', 'all_rescue'),
    ):
        probe = (f"{v['probe'][left][key]} to {v['probe'][right][key]}"
                 if key else f"{v['risk'][right]['withdrawals']} withdrawals")
        table.append(f"| {name} | {probe} | {v['risk'][left]['passes']} to {v['risk'][right]['passes']} | "
                     f"{v['risk'][left]['easy_net_over_2']} to {v['risk'][right]['easy_net_over_2']} |")
    conf = boxes['contrasts']['first_rescue']
    lo, hi = conf['interval95']
    text = f'''# From Insertion Retrieval to Whole-Recording Repair

These experiments use the same exposed native image-coordinate
cohort: 34 recordings, 14 source groups and three frozen forecast
seeds (102 repeated views). Observation and prediction lengths are
eight and twelve annotation steps. None is an independent test.
The unchanged direct policy has {base["passes"]} recording passes,
{base["undefined_easy"]} undefined easy-reference views and
{base["easy_net_over_2"]} signed easy violations.

## Matched Model Comparisons

The component column shows first eligible-query successes without
thresholding, except for optional withdrawal, which reports the
number of actual withdrawals after the same frozen insertion.
Factorized value measures observed easy reference with nonnegative
local gain; the other probes require non-loss whole-record rescue.
Recording columns use the original full policy with donor-only
calibration. An empty intervention does not establish safety.

### Table 1. Paired development results, control to treatment

| Comparison | Component probe | Recording passes | Signed easy violations |
| --- | ---: | ---: | ---: |
''' + '\n'.join(table) + f'''

The box ablation aligns all eight observed head-box centers exactly
to the original trajectories. Its first-query rescue contrast is
{100*conf["point"]:+.2f} pp (paired source-bootstrap 95% interval
[{100*lo:+.2f}, {100*hi:+.2f}]). Both feasible probes come from
{diagnostic["rescue_distinct_recordings"]} recording and neither has a donor-selected cutoff.
Box geometry is not an occlusion label.

## What the Action Set Permits

Among the {n} views without easy reference, {cap["zero_direct_views"]}
have no action to withdraw. With observed labels, same-query
one-for-one exchange repairs {cap["same_query"]}/{n}; allowing a later
withdrawal at the same recording count raises this to {cap["ordered_one_pair"]}/{n}.
Insertion alone permits {bounds["insert_only"]["checked_feasible"]}/{n};
an insertion with optional ordered deletions permits
{bounds["ordered_joint"]["checked_feasible"]}/{n}. These are different
action sets and overlapping counts, not results to add or
learned-policy passes. Verified witnesses preserve the original
2% and reference-support checks. More flexible action sets do not
resolve the problem of retrieving the right action from past data.
No deployment, physical-safety or metric claim follows.
'''
    tex = latex_document(text).replace('[10pt,twocolumn,letterpaper]', '[10pt,onecolumn,letterpaper]')
    BUILD.mkdir(parents=True, exist_ok=True)
    for p, body in ((OUT/'insertion_appendix.md', text), (OUT/'insertion_appendix.tex', tex),
                    (BUILD/'main.tex', tex)):
        p.write_text(body)
    shutil.copyfile(BUILD.parent/'paper_build/cvpr.sty', BUILD/'cvpr.sty')
    write_json(OUT/'insertion_appendix_bindings.json', dict(source_sha256=sources,
        builder_sha256=sha(Path(__file__)), tex_sha256=sha(OUT/'insertion_appendix.tex'),
        independent_confirmation=False, submission_ready=False, new_outcome_readout=False))


if __name__ == '__main__':
    main()
