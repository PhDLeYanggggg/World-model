from __future__ import annotations

from src.stage42_full_waypoint_loss_family_replay import run_stage42_full_waypoint_loss_family_replay


if __name__ == "__main__":
    payload = run_stage42_full_waypoint_loss_family_replay()
    gate = payload["stage42_dx_gate"]
    print(f"Stage42-DX full-waypoint loss-family replay: {gate['passed']}/{gate['total']} {gate['verdict']}")
