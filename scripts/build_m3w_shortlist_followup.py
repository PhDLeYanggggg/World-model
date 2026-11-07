"""Assemble the shortlist follow-up appendix from exact replay receipts."""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_m3w_native_paper_v10 import latex_document, sha, write_json
from scripts.build_m3w_utility_tradeoff_appendix import LEDGER, OUT

BUILD = ROOT/'data/stage_cvpr2027_experiments/evidence_manuscript_v10/shortlist_followup_build'


def main():
    sources = {}

    def checked(folder):
        a = LEDGER/folder/'analysis.json'
        replay = a.with_name('replay_execution.json')
        if json.loads(replay.read_text())['artifact']['sha256'] != sha(a):
            raise ValueError('Require exact score replay')
        for path in (a, replay):
            sources[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(a.read_text())

    h, i, p = map(checked, ('shortlist_history', 'shortlist_interaction', 'shortlist_portfolio'))
    for v in (h, i, p):
        if v['views'] != 102 or not v['original_rules_unchanged'] or v['independent_confirmation']:
            raise ValueError('Incompatible study scope')
        if v['risk']['base'] != h['risk']['base'] or v['risk']['matched_base'] != h['risk']['matched_base']:
            raise ValueError('Incompatible controls')
    matched = h['risk']['matched_base']['switches']
    rows = []
    for name, d, arm, key, hits in (
        ('Short observed-window errors', h, 'backcast', 'matched_backcast_vs_base', h['recall']['first_two']['backcast']),
        ('Resolved forecast errors', h, 'resolved', 'matched_resolved_vs_base', h['recall']['first_two']['resolved']),
        ('Observed neighbor motion', i, 'observed', 'matched_observed', i['first_two_recall']['observed']),
        ('Motion and rollout proximity', i, 'rollout', 'matched_rollout', i['first_two_recall']['rollout']),
        ('Portfolio, risk winner', p, 'risk', 'matched_risk', p['first_two_recall']['risk']),
        ('Portfolio, event winner', p, 'event', 'matched_event', p['first_two_recall']['event']),
    ):
        if d['risk']['matched_'+arm]['switches'] != matched:
            raise ValueError('Require same diagnostic switch budget')
        risk = d['risk'][arm]
        v = d['contrasts'][key]
        lo, hi = v['interval95']
        rows.append(f"| {name} | {hits} | {risk['passes']} | {risk['easy_net_over_2']} | "
                    f"{v['point']:+.5f} [{lo:+.5f}, {hi:+.5f}] |")
    hit = p['first_two_recall']
    text = f'''# What Limits Early Intervention At A Fixed Action Budget?

The exposed Moving A development task uses eight observed and twelve
forecast annotation steps in native image coordinates. We keep the
bank, direct actions, eight candidate slots, and the original 2%
risk and reference-support rules fixed. Each of 42 source/seed cells
uses seven fitting, six calibration, and one held source. The views
are 34 recordings repeated across three forecaster seeds.

## Retrieval And Choice Are Separate Bottlenecks

In the first two eligible queries of 18 original zero-direct views,
the full bank has a positive supported action in 15. The family-diverse
list retains one in {hit['base_list']}. The new cost portfolio keeps the
old top action and cycles through support, gain, risk-adjusted gain,
and low excess, preferring different agents. It retains one in
{hit['portfolio']}: {p['recall_retained']} shared with the old list,
{p['recall_added']} new and {p['recall_lost']} lost. Thus diversifying
cost rankings is not a replacement for family diversity.

## Refit Results

Early hits are final choices in the first two eligible queries, out of
18 original zero-direct views, without thresholding. Pass and easy
columns are complete calibrated policy counts out of 102 views.
The unchanged selector has zero early hits, two passes and eight
signed easy violations. Matched effects compare exactly {matched:,}
switches on identical queries to that selector, in percentage points
of completion-lower utility. Intervals resample 14 sources 3,000 times,
averaging the three seeds within source.

### Table 1. Early choice, observed policy passes, and matched utility

| Added input or changed retrieval | Early hits | Passes | Easy >2% | Matched effect [95% interval] |
| --- | ---: | ---: | ---: | ---: |
''' + '\n'.join(rows) + '''

No tested model adds a complete observed pass. The short-history
backtest improves matched utility without solving early choice.
Already-resolved feedback is absent in those early queries.
Neighbor summaries do not recover a useful choice; rollout proximity
adds a calibrated easy violation. The portfolio event model finds
one early choice but loses matched utility and complete passes at
fixed coverage. None provides independent confirmation or changes
deployment. The units are not seconds, meters or physical safety.
'''
    tex = latex_document(text).replace('[10pt,twocolumn,letterpaper]', '[10pt,onecolumn,letterpaper]')
    tex = (tex.replace(r'\begin{table*}[t]\centering\small', r'\begin{center}\small')
              .replace(r'\caption{', r'\captionof{table}{')
              .replace(r'\end{table*}', r'\end{center}')
              .replace(r'>{\raggedright\arraybackslash}X' * 5, 'Xrrrl'))
    BUILD.mkdir(parents=True, exist_ok=True)
    for path, content in ((OUT/'shortlist_followup.md', text), (OUT/'shortlist_followup.tex', tex),
                          (BUILD/'main.tex', tex)):
        path.write_text(content)
    shutil.copyfile(BUILD.parent/'paper_build/cvpr.sty', BUILD/'cvpr.sty')
    write_json(OUT/'shortlist_followup_bindings.json', dict(source_sha256=sources,
        builder_sha256=sha(Path(__file__)), tex_sha256=sha(OUT/'shortlist_followup.tex'),
        independent_confirmation=False, submission_ready=False, new_outcome_readout=False))


if __name__ == '__main__':
    main()
