from pathlib import Path

from src.final_model_pipeline import FINAL_DIR, run_inference_demo, train_final_model


def test_final_inference_demo_runs():
    train_final_model(quick=True)
    out = run_inference_demo()
    assert "predicted_trajectories" in out
    assert (FINAL_DIR / "inference_demo_output.json").exists()
    assert Path("outputs/final_model/final_selected_checkpoint.pt").exists()

