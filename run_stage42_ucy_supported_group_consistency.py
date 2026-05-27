from src.stage42_ucy_supported_group_consistency import run_stage42_ucy_supported_group_consistency


if __name__ == "__main__":
    payload = run_stage42_ucy_supported_group_consistency()
    gate = payload["stage42_dz_gate"]
    print(
        "Stage42-DZ UCY-supported group consistency:",
        f"{gate['passed']}/{gate['total']}",
        gate["verdict"],
    )
