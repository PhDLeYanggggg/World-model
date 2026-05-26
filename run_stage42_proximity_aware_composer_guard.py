from src.stage42_proximity_aware_composer_guard import run


if __name__ == "__main__":
    result = run()
    gate = result["stage42_cq_gate"]
    print(f"Stage42-CQ proximity composer guard: {gate['verdict']} ({gate['passed']}/{gate['total']})")
