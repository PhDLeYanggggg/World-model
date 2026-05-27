from src.stage42_explicit_physical_consistency_checkpoint import run_stage42_explicit_physical_consistency_checkpoint


if __name__ == "__main__":
    payload = run_stage42_explicit_physical_consistency_checkpoint()
    gate = payload["stage42_dy_gate"]
    print(
        "Stage42-DY explicit physical consistency checkpoint:",
        f"{gate['passed']}/{gate['total']}",
        gate["verdict"],
    )
