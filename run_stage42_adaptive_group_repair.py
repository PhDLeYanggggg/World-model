from src.stage42_adaptive_group_repair import run_stage42_adaptive_group_repair


if __name__ == "__main__":
    payload = run_stage42_adaptive_group_repair()
    gate = payload["stage42_ew_gate"]
    print(f"Stage42-EW adaptive group repair: {gate['passed']}/{gate['total']} {gate['verdict']}")
