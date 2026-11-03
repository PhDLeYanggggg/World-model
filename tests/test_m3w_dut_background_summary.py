from scripts import audit_m3w_dut_background_bridge as audit


def test_real_summary_assembly_preserves_false_role_without_duplicate_keywords(tmp_path,monkeypatch):
    (tmp_path/'registration.json').write_text('{}')
    monkeypatch.setattr(audit,'PUBLIC',tmp_path)
    result=audit.summarize([],dict(conservative_exclusion_components=[['moving_a'],['drone_b']]),
                           ['dut_intersection_01'],{})
    assert result['clip_pairs']==0 and result['frame_pairs_attempted']==0
    assert result['scientific_roles_assigned'] is False
    assert result['no_model_training'] is True
    assert len(result['conservative_components'])==3
