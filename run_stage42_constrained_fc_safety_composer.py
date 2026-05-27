from src.stage42_constrained_fc_safety_composer import run_stage42_constrained_fc_safety_composer


if __name__ == "__main__":
    payload = run_stage42_constrained_fc_safety_composer()
    gate = payload["stage42_fe_gate"]
    print(f"Stage42-FE gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
