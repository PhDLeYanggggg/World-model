"""Presentation-only legend placement; frozen scientific artifacts stay unchanged."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import report_m3w_budget_score_cost as report


def render():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    run = report.run
    paths = [run.PUBLIC / p for p in ('summary.json', 'complete.json', 'verification.json',
        'evidence.json', 'training_traces.json', 'reference_support.json')]
    frozen = {p.name: run.sha(p) for p in paths}
    original = Axes.legend
    calls = []
    def legend(ax, *args, **kwargs):
        if ax.get_title() == 'Complete-policy known ADE utility':
            kwargs.update(loc='upper center', bbox_to_anchor=(.70, 1.))
            calls.append(ax.get_title())
        return original(ax, *args, **kwargs)
    Axes.legend = legend
    try:
        report.figures()
    finally:
        Axes.legend = original
    if calls != ['Complete-policy known ADE utility'] or any(run.sha(p) != frozen[p.name] for p in paths):
        raise ValueError('Exactly one presentation adjustment; no scientific artifact change allowed')
    value = dict(presentation_only=True, change='move_first_panel_legend_between_candidate_columns',
        scientific_artifacts_sha256=frozen, estimates_intervals_traces_and_risk_rules_unchanged=True,
        code_sha256=run.sha(Path(__file__)),
        figures={p: run.sha(run.PUBLIC / 'figures' / p) for p in ('summary.svg', 'neural_use.svg')})
    run.once(run.PUBLIC / 'figure_presentation.json', value)
    return value


if __name__ == '__main__':
    print(json.dumps(render()))
