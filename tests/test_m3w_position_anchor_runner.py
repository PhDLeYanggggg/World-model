from scripts import run_m3w_position_anchor as run


def test_original_loader_registry_uses_contexts_not_new_registration_source_contexts(monkeypatch):
    contexts = {f'c{i}': [dict(packet=dict(identity=dict(source=f's{j}'))) for j in range(4)] for i in range(6)}
    # The original loader returns its ancestor registry,not its own serialized registration.
    monkeypatch.setattr(run.base, 'admission', lambda: (None, dict(contexts=contexts), None, None, None, None))
    written = {}
    monkeypatch.setattr(run, 'once', lambda path, value: written.update(value))
    run.admission(True)
    assert written['contexts'] == {c: [f's{j}' for j in range(4)] for c in contexts}
    assert written['groups'] == 24 and not written['own_outcomes_read_at_registration']
    assert not written['independent_roles_opened'] and written['optimizer_updates'] == 0
