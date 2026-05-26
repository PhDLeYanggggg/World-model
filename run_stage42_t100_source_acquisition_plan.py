from src.stage42_t100_source_acquisition_plan import run_stage42_t100_source_acquisition_plan


if __name__ == "__main__":
    result = run_stage42_t100_source_acquisition_plan()
    gate = result["stage42_bc_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
