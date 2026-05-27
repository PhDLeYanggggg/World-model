from src.stage42_safety_aware_joint_objective_training import run_stage42_safety_aware_joint_objective_training


if __name__ == "__main__":
    payload = run_stage42_safety_aware_joint_objective_training()
    gate = payload["stage42_fd_gate"]
    print(f"Stage42-FD gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
