from src.final_model_pipeline import load_checkpoint, train_final_model


def test_final_checkpoint_forbids_leakage_inputs():
    train_final_model(quick=True)
    checkpoint = load_checkpoint()
    assert checkpoint["no_future_endpoint_input"] is True
    assert checkpoint["no_test_endpoint_goals"] is True
    assert checkpoint["no_central_velocity"] is True
    assert checkpoint["oracle_labels_are_training_supervision_only"] is True

