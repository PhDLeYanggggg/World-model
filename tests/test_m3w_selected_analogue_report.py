from scripts.report_m3w_selected_analogue import render
from tests.test_m3w_selected_analogue_verifier import example


def test_descriptive_slices_do_not_drop_unknown_queries_or_undefined_views():
    rows, _, _ = example()
    rows[0]['scores']['nearest32']['easy']['AUROC'] = None
    result, table = render(rows, 'completion', 'reporter')
    assert result['all']['nearest32']['unknown'] == 72
    assert result['all']['nearest32']['easy']['AUROC'] is None
    assert result['all']['nearest32']['easy']['undefined_AUROC_views'] == 1
    assert len(result['context']) == 6 and len(result['seed']) == 3
    assert len(result['source']) == 4
    assert result['whole_pipeline_OOF'] is False and result['screen_changed'] is False
    assert table.count('| context:') == 6
