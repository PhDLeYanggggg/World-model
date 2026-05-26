from src.stage42_bridge_shape_composer import run


if __name__ == "__main__":
    payload = run()
    gate = payload["stage42_cn_gate"]
    print(f"Stage42-CN bridge/shape composer: {gate['verdict']} ({gate['passed']}/{gate['total']})")
