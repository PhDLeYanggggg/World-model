from src.stage42_local_t100_source_inventory import run_stage42_local_t100_source_inventory


if __name__ == "__main__":
    result = run_stage42_local_t100_source_inventory()
    gate = result["stage42_bd_gate"]
    print(f"{gate['verdict']} ({gate['passed']}/{gate['total']})")
