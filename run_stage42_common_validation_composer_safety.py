from src.stage42_common_validation_composer_safety import run


if __name__ == "__main__":
    payload = run()
    gate = payload["stage42_cp_gate"]
    print(f"Stage42-CP composer safety: {gate['verdict']} ({gate['passed']}/{gate['total']})")
