from scripts.run_m3w_moving_supported_global_motion_safe import prepare_directories


def test_runtime_directory_initialization_is_idempotent_and_preserves_weights(tmp_path):
    private=tmp_path/'private';private.mkdir()
    weights=private/'model.pt.gz';weights.write_bytes(b'unchanged-frozen-checkpoint')
    path=prepare_directories(private)
    assert path==private/'allocation' and path.is_dir()
    assert prepare_directories(private)==path
    assert weights.read_bytes()==b'unchanged-frozen-checkpoint'
    assert list(path.iterdir())==[]
