import json

from scripts import build_m3w_focused_evidence_v202 as draft


def test_backbone_table_keeps_direct_paired_gain_distinct_from_cv_gains():
    root = draft.prior.main.PUBLIC
    paths = dict(social='opendd_vehicle_stgcnn_v1/readout.json',
                 eq='opendd_vehicle_eqmotion_v1/readout.json',
                 pairs='opendd_vehicle_stgcnn_v1/backbone_pairs.json')
    table, markers = draft.backbone_table({k: json.loads((root/v).read_text()) for k, v in paths.items()})
    assert r'EqMotion~\cite{eqmotion} & 4.510 & 13.914 & 10.087 & +6.19 & 5' in table
    assert r'Social-STGCNN~\cite{socialstgcnn} & 4.701 & 14.620 & 11.292 & +2.33 & 3' in table
    assert 'matched EqMotion model is\n-16.66 pp [-18.96, -14.31]' in table
    assert '-16.66' in markers


def test_revision_keeps_title_abstract_and_llm_disclosure(monkeypatch):
    monkeypatch.setattr(draft, 'backbone_table', lambda rows: ('matched graph controls\n', ()))
    start = '\\title{Original title}\n\\begin{abstract}Original abstract\\end{abstract}\n'
    tail = '\\section*{LLM Usage Statement}\nOriginal disclosure.\n'
    original = (start+draft.BEFORE+'\nold body\n'+draft.AFTER+r'\label{tab:matched}'
                '\nThe intervals resample sources. Neither check changes release.\n'+tail+r'\end{thebibliography}')
    revised, _ = draft.revise(original, {})
    assert revised.startswith(start)
    assert tail+draft.SOCIAL_BIB in revised
    assert 'old body' not in revised
    assert 'Three retrospective examples' in revised
