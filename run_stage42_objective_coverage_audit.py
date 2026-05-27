from src.stage42_objective_coverage_audit import run_stage42_objective_coverage_audit


if __name__ == "__main__":
    payload = run_stage42_objective_coverage_audit()
    gate = payload["stage42_fx_gate"]
    print(f"Stage42-FX gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
