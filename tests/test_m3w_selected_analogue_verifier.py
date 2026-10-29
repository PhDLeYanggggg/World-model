import copy
import numpy as np
import pytest
from scripts.verify_m3w_selected_analogue import verify, ARMS
from src.evaluation import m3w_selected_analogue as api


def example():
    y = np.zeros((3, 5)); y[:, 2:4] = 1; y[0, [1,4]] = .1; y[2] = np.nan
    p = np.zeros_like(y); p[:, 2:4] = 1; p[:, [1,4]] = .02
    scores = {a: api.score(y, p, np.ones(3), 1., np.ones(3, bool)) for a in ARMS}
    rows = [dict(context=f'c{c}', source=f's{s}', seed=seed, scope='fit',
        scores=copy.deepcopy(scores), prior_predictions_exact=True, whole_pipeline_OOF=False,
        held_inference=False, independent_roles_read=False, deployment_changed=False,
        references=[dict(recording='query', query_rows=3, causal_donors=40, known_donors=32,
            donor_recordings=2, same_recording_donors=0, supported=True, mean_neighbor_distance=.1)])
        for c in range(6) for s in range(4) for seed in (17,29,43)]
    expected = {(r['context'],r['source'],r['seed']) for r in rows}
    return rows, api.summarize(rows, expected), expected


def test_independent_summary_and_preserved_unknown_counts():
    result = verify(*example())
    assert result['checks'] == dict(views=72,arm_targets=720,recording_banks=72)
    assert not result['information_screen_pass']


@pytest.mark.parametrize('mutation', ['unknown', 'support', 'same_recording', 'screen'])
def test_tampered_receipts_are_rejected(mutation):
    rows, summary, expected = example()
    if mutation == 'unknown': rows[0]['scores']['nearest32']['unknown'] = 0
    if mutation == 'support': rows[0]['scores']['nearest32']['unsupported'] = 1
    if mutation == 'same_recording': rows[0]['references'][0]['same_recording_donors'] = 1
    if mutation == 'screen': summary['conditional_signal_screen_pass'] = True
    with pytest.raises(ValueError): verify(rows, summary, expected)
