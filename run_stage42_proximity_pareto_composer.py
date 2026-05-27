from src.stage42_proximity_pareto_composer import run_stage42_proximity_pareto_composer


if __name__ == "__main__":
    payload = run_stage42_proximity_pareto_composer()
    gate = payload["stage42_fb_gate"]
    print(f"Stage42-FB gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
    print(f"Decision: {payload['deployment_decision']['decision']}")
