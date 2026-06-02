from __future__ import annotations

import numpy as np
import torch

from src import stage43_latent_transition_adapter_repair as m


def _payload(*, raw_gain: float = 0.4, calibrated_gain: float = 0.2, ci_low: float = 0.05) -> dict:
    return {
        "stage43_m_checkpoint_replayed": True,
        "stage43_by_precondition_seen": True,
        "training_history": [{"epoch": 1, "train_loss": 1.0}],
        "no_leakage": {
            "future_target_latent_label_eval_only": True,
            "future_target_latent_input": False,
            "test_statistics_normalization": False,
        },
        "adapter_latent_stats": {"adapter_min_variance": 0.02, "noncollapse_threshold": 0.01},
        "adapter_overall": {
            "transition_gain_vs_identity": raw_gain,
            "transition_gain_vs_train_centroid": 0.1,
            "mse_next_to_target": 0.4,
        },
        "stage43_by_reference": {
            "overall": {"mse_next_to_target": 0.5},
            "calibrated_readout_overall": {},
        },
        "adapter_calibrated_readout_overall": {"transition_gain_vs_identity": calibrated_gain},
        "adapter_calibrated_readout_bootstrap": {
            "transition_gain_vs_identity": {"low": ci_low},
        },
        "adapter_domain_breakdown": {"UCY": {"rows": 10}},
        "adapter_horizon_breakdown": {"50": {"rows": 10}},
        "weak_adapter_slices": [],
        "weak_calibrated_adapter_slices": [],
        "claim_boundary": {
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }


def test_adapter_forward_preserves_shape_and_finite_values() -> None:
    model = m.LatentTransitionAdapter(feature_dim=5, latent_dim=3, hidden_dim=7, delta_clip=2.0)
    x = torch.zeros((4, 5), dtype=torch.float32)
    z = torch.randn((4, 3), dtype=torch.float32)
    out = model(x, z)
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_gate_passes_when_adapter_beats_identity_with_bootstrap_support() -> None:
    gate = m._gate(_payload(raw_gain=0.4, calibrated_gain=0.2, ci_low=0.03))
    assert gate["passed"] == gate["total"]
    assert gate["verdict"] == "stage43_bz_latent_transition_adapter_repair_pass"


def test_gate_reports_readout_caveat_when_raw_repair_works_but_calibrated_identity_remains_stronger() -> None:
    gate = m._gate(_payload(raw_gain=0.4, calibrated_gain=-0.02, ci_low=-0.05))
    assert gate["gates"]["raw_adapter_beats_identity"] is True
    assert gate["gates"]["calibrated_adapter_beats_identity"] is False
    assert gate["verdict"] == "stage43_bz_latent_transition_adapter_repair_pass_with_readout_caveat"


def test_limit_split_subsamples_aligned_arrays() -> None:
    class Dummy:
        pass

    ds = Dummy()
    ds.x = np.arange(20, dtype=np.float32).reshape(10, 2)
    ds.y = np.arange(10, dtype=np.float32)
    ds.name = "keep"
    out = m._limit_split(ds, 4, seed=1)
    assert len(out.x) == 4
    assert len(out.y) == 4
    assert out.name == "keep"
