from scripts import report_m3w_selection_conditioned_analogue as report
from tests.test_m3w_selection_conditioned_verifier import fixture


def test_population_and_slices_preserved():
    rows, _, _ = fixture(); data, text = report.describe(rows)
    assert data['all']['selected_nearest32']['rows'] == 216
    assert data['all']['selected_nearest32']['unknown'] == 72
    assert data['decomposition']['selected']['easy']['known'] == 144
    assert data['decomposition']['selected']['easy']['events'] == 72
    assert data['decomposition']['selected']['easy']['brier'] == .25
    assert len(data['context']) == 6 and len(data['seed']) == 3 and len(data['source']) == 4
    assert 'undefined' not in text.split('## Event')[0]
    assert not data['screen_changed'] and not data['deployment_changed']
