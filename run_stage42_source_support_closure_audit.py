from src.stage42_source_support_closure_audit import run_stage42_source_support_closure_audit


if __name__ == "__main__":
    result = run_stage42_source_support_closure_audit()
    gate = result["stage42_dd_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
