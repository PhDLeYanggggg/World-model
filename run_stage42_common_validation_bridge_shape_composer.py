from src.stage42_common_validation_bridge_shape_composer import run


if __name__ == "__main__":
    payload = run()
    gate = payload["stage42_co_gate"]
    print(f"Stage42-CO common validation composer: {gate['verdict']} ({gate['passed']}/{gate['total']})")
