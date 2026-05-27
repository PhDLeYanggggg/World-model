from src.stage42_fe_source_robustness_audit import run_stage42_fe_source_robustness_audit


if __name__ == "__main__":
    result = run_stage42_fe_source_robustness_audit()
    gate = result["stage42_fg_gate"]
    print(f"Stage42-FG gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
